#!/usr/bin/env python3
"""
钢铁缺陷检测 - YOLOv11训练脚本
使用Ultralytics YOLOv11进行训练
"""

from pathlib import Path
import yaml
import argparse
from datetime import datetime
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


class SteelDefectTrainer:
    def __init__(self, config_path="config/training_config.yaml"):
        """
        初始化训练器
        
        Args:
            config_path: 训练配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # 设置随机种子
        self.set_seed(self.config.get("seed", 42))
        
        # 设备设置
        self.device = self.set_device()
        
        # 输出目录
        self.output_dir = Path(self.config.get("output_dir", "runs/detect"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验名称
        exp_name = self.config.get("experiment_name", "steel_defect")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.output_dir / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"实验目录: {self.exp_dir}")
        
    def load_config(self):
        """加载训练配置"""
        if not self.config_path.exists():
            print(f"警告: 配置文件不存在 {self.config_path}，使用默认配置")
            return self.get_default_config()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_default_config(self):
        """获取默认训练配置"""
        return {
            # 数据配置
            "data_yaml": "data/yolo_format/data.yaml",
            "imgsz": 200,  # 保持原尺寸
            
            # 模型配置
            "model": "yolo11n.pt",  # 从nano模型开始
            "pretrained": True,
            
            # 训练配置
            "epochs": 200,
            "batch_size": 16,
            "workers": 4,
            "device": "0",  # GPU设备
            
            # 优化器配置
            "optimizer": "AdamW",
            "lr0": 0.001,   # 初始学习率
            "lrf": 0.01,    # 最终学习率衰减
            "momentum": 0.937,
            "weight_decay": 0.0005,
            
            # 学习率调度
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,  # 余弦退火
            
            # 损失函数权重
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            
            # 数据增强
            "hsv_h": 0.015,  # 色调增强
            "hsv_s": 0.7,    # 饱和度增强
            "hsv_v": 0.4,    # 明度增强
            "degrees": 10.0, # 旋转角度
            "translate": 0.1, # 平移
            "scale": 0.5,    # 缩放
            "fliplr": 0.5,   # 水平翻转概率
            
            # 正则化
            "dropout": 0.0,
            "label_smoothing": 0.0,
            
            # 其他配置
            "patience": 50,  # 早停耐心值
            "save_period": 10,  # 保存周期
            "seed": 42,
            "verbose": True,
            "exist_ok": True,
            "project": "steel_defect_detection",
            "name": "yolo11n_steel_v1",
        }
    
    def set_seed(self, seed=42):
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"设置随机种子: {seed}")
    
    def set_device(self):
        """设置训练设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"使用GPU: {gpu_name} (数量: {gpu_count})")
            print(f"CUDA版本: {torch.version.cuda}")
        else:
            device = torch.device("cpu")
            print("使用CPU进行训练")
        
        return device
    
    def analyze_dataset(self):
        """分析数据集分布"""
        print("\n" + "=" * 60)
        print("分析数据集分布")
        print("=" * 60)
        
        data_yaml = self.config.get("data_yaml")
        if not Path(data_yaml).exists():
            print(f"错误: 数据配置文件不存在 {data_yaml}")
            return
        
        # 读取数据配置
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 获取类别信息
        class_names = data_config.get("names", [])
        num_classes = len(class_names)
        
        print(f"数据集路径: {data_config.get('path', '未知')}")
        print(f"训练集: {data_config.get('train', '未知')}")
        print(f"验证集: {data_config.get('val', '未知')}")
        print(f"测试集: {data_config.get('test', '未知')}")
        print(f"类别数量: {num_classes}")
        print(f"类别名称: {class_names}")
        
        # 统计每个类别的样本数量
        train_labels_dir = Path(data_config['path']) / "labels" / "train"
        val_labels_dir = Path(data_config['path']) / "labels" / "val"
        
        if train_labels_dir.exists():
            class_counts = {i: 0 for i in range(num_classes)}
            
            # 统计训练集
            train_label_files = list(train_labels_dir.glob("*.txt"))
            for label_file in tqdm(train_label_files, desc="统计训练集类别分布"):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                cls_id = int(line.strip().split()[0])
                                if cls_id in class_counts:
                                    class_counts[cls_id] += 1
                except:
                    continue
            
            # 统计验证集
            val_label_files = list(val_labels_dir.glob("*.txt"))
            for label_file in tqdm(val_label_files, desc="统计验证集类别分布"):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                cls_id = int(line.strip().split()[0])
                                if cls_id in class_counts:
                                    class_counts[cls_id] += 1
                except:
                    continue
            
            # 打印统计结果
            print("\n类别分布统计:")
            total_instances = sum(class_counts.values())
            for cls_id, count in class_counts.items():
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
                percentage = (count / total_instances * 100) if total_instances > 0 else 0
                print(f"  {cls_name} (ID: {cls_id}): {count} 个实例 ({percentage:.1f}%)")
            
            print(f"总实例数: {total_instances}")
            
            # 保存统计结果
            stats_df = pd.DataFrame({
                'class_id': list(class_counts.keys()),
                'class_name': [class_names[i] if i < len(class_names) else f"Class_{i}" 
                              for i in class_counts.keys()],
                'count': list(class_counts.values())
            })
            stats_path = self.exp_dir / "class_distribution.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"类别分布已保存到: {stats_path}")
            
            # 可视化类别分布
            self.plot_class_distribution(stats_df, class_names)
    
    def plot_class_distribution(self, stats_df, class_names):
        """可视化类别分布"""
        plt.figure(figsize=(12, 6))
        
        # 柱状图
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(stats_df)), stats_df['count'])
        plt.xlabel('类别ID')
        plt.ylabel('实例数量')
        plt.title('钢铁缺陷类别分布')
        plt.xticks(range(len(stats_df)), [f'{i}\n{name[:10]}' for i, name in enumerate(class_names)])
        
        # 添加数值标签
        for bar, count in zip(bars, stats_df['count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    str(count), ha='center', va='bottom')
        
        # 饼图
        plt.subplot(1, 2, 2)
        plt.pie(stats_df['count'], labels=class_names, autopct='%1.1f%%')
        plt.title('类别比例分布')
        
        plt.tight_layout()
        plot_path = self.exp_dir / "class_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别分布图已保存到: {plot_path}")
    
    def train_model(self):
        """训练YOLOv11模型"""
        print("\n" + "=" * 60)
        print("开始训练YOLOv11模型")
        print("=" * 60)
        
        # 获取训练参数
        train_args = {
            'data': self.config.get("data_yaml"),
            'epochs': self.config.get("epochs", 200),
            'imgsz': self.config.get("imgsz", 200),
            'batch': self.config.get("batch_size", 16),
            'workers': self.config.get("workers", 4),
            'device': self.config.get("device", "0"),
            'project': self.config.get("project", "steel_defect_detection"),
            'name': self.config.get("name", "yolo11n_steel_v1"),
            'exist_ok': self.config.get("exist_ok", True),
            'pretrained': self.config.get("pretrained", True),
            'optimizer': self.config.get("optimizer", "AdamW"),
            'lr0': self.config.get("lr0", 0.001),
            'cos_lr': self.config.get("cos_lr", True),
            'patience': self.config.get("patience", 50),
            'save': True,
            'save_period': self.config.get("save_period", 10),
            'visualize': False,
            'verbose': self.config.get("verbose", True),
            
            # 数据增强参数
            'hsv_h': self.config.get("hsv_h", 0.015),
            'hsv_s': self.config.get("hsv_s", 0.7),
            'hsv_v': self.config.get("hsv_v", 0.4),
            'degrees': self.config.get("degrees", 10.0),
            'translate': self.config.get("translate", 0.1),
            'scale': self.config.get("scale", 0.5),
            'fliplr': self.config.get("fliplr", 0.5),
            
            # 损失函数权重
            'box': self.config.get("box", 7.5),
            'cls': self.config.get("cls", 0.5),
            'dfl': self.config.get("dfl", 1.5),
            
            # 学习率调度
            'warmup_epochs': self.config.get("warmup_epochs", 3),
            'warmup_momentum': self.config.get("warmup_momentum", 0.8),
            'warmup_bias_lr': self.config.get("warmup_bias_lr", 0.1),
        }
        
        # 打印训练配置
        print("训练配置:")
        for key, value in train_args.items():
            if key not in ['data', 'project', 'name']:
                print(f"  {key}: {value}")
        
        # 加载模型
        model_name = self.config.get("model", "yolo11n.pt")
        print(f"\n加载模型: {model_name}")
        
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"错误: 加载模型失败: {e}")
            print("尝试使用yolov8n.pt作为替代")
            model = YOLO("yolov8n.pt")
        
        # 开始训练
        print("\n开始训练...")
        try:
            results = model.train(**train_args)
            print("训练完成!")
            
            # 保存训练结果
            self.save_training_results(results, model)
            
            return model, results
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_training_results(self, results, model):
        """保存训练结果"""
        print("\n" + "=" * 60)
        print("保存训练结果")
        print("=" * 60)
        
        # 保存训练配置
        config_path = self.exp_dir / "training_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        print(f"训练配置已保存到: {config_path}")
        
        # 保存训练日志
        if hasattr(results, 'csv'):
            log_path = self.exp_dir / "training_log.csv"
            results.csv(log_path)
            print(f"训练日志已保存到: {log_path}")
        
        # 保存最佳模型路径
        best_model_path = Path(model.trainer.best) if hasattr(model.trainer, 'best') else None
        if best_model_path and best_model_path.exists():
            model_info = {
                'best_model': str(best_model_path),
                'experiment_dir': str(self.exp_dir),
                'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': self.config
            }
            
            info_path = self.exp_dir / "model_info.yaml"
            with open(info_path, 'w', encoding='utf-8') as f:
                yaml.dump(model_info, f, default_flow_style=False, allow_unicode=True)
            print(f"模型信息已保存到: {info_path}")
    
    def evaluate_model(self, model_path=None):
        """评估训练好的模型"""
        print("\n" + "=" * 60)
        print("评估模型性能")
        print("=" * 60)
        
        if model_path is None:
            # 查找最佳模型
            model_files = list(self.exp_dir.glob("**/best.pt"))
            if not model_files:
                print("错误: 未找到训练好的模型")
                return
            
            model_path = model_files[0]
        
        print(f"加载模型: {model_path}")
        
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"错误: 加载模型失败: {e}")
            return
        
        # 在验证集上评估
        data_yaml = self.config.get("data_yaml")
        print(f"使用数据集: {data_yaml}")
        
        try:
            metrics = model.val(
                data=data_yaml,
                imgsz=self.config.get("imgsz", 200),
                batch=self.config.get("batch_size", 16),
                conf=0.25,      # 置信度阈值
                iou=0.45,       # IoU阈值
                device=self.config.get("device", "0"),
                plots=True,     # 生成评估图表
                save_json=True, # 保存JSON结果
            )
            
            # 打印关键指标
            print("\n评估结果:")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  精确率: {metrics.box.p:.4f}")
            print(f"  召回率: {metrics.box.r:.4f}")
            print(f"  F1分数: {2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-16):.4f}")
            
            # 保存评估结果
            eval_results = {
                'mAP50-95': float(metrics.box.map),
                'mAP50': float(metrics.box.map50),
                'precision': float(metrics.box.p),
                'recall': float(metrics.box.r),
                'f1_score': float(2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-16)),
                'eval_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            eval_path = self.exp_dir / "evaluation_results.yaml"
            with open(eval_path, 'w', encoding='utf-8') as f:
                yaml.dump(eval_results, f, default_flow_style=False, allow_unicode=True)
            print(f"评估结果已保存到: {eval_path}")
            
            return metrics
            
        except Exception as e:
            print(f"评估过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def export_model(self, model_path=None, format="onnx"):
        """导出模型为其他格式"""
        print("\n" + "=" * 60)
        print(f"导出模型为 {format.upper()} 格式")
        print("=" * 60)
        
        if model_path is None:
            model_files = list(self.exp_dir.glob("**/best.pt"))
            if not model_files:
                print("错误: 未找到训练好的模型")
                return
            
            model_path = model_files[0]
        
        print(f"加载模型: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            # 导出模型
            export_args = {
                'format': format,
                'imgsz': self.config.get("imgsz", 200),
                'device': self.config.get("device", "0"),
                'simplify': True,  # 简化模型
                'opset': 12,       # ONNX opset版本
            }
            
            if format == "engine":  # TensorRT
                export_args['workspace'] = 4
            
            exported_path = model.export(**export_args)
            print(f"模型已导出到: {exported_path}")
            
            # 保存导出信息
            export_info = {
                'original_model': str(model_path),
                'exported_model': str(exported_path),
                'export_format': format,
                'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'export_args': export_args
            }
            
            info_path = self.exp_dir / f"export_info_{format}.yaml"
            with open(info_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_info, f, default_flow_style=False, allow_unicode=True)
            print(f"导出信息已保存到: {info_path}")
            
            return exported_path
            
        except Exception as e:
            print(f"导出过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_pipeline(self):
        """运行完整的训练流程"""
        print("钢铁缺陷检测 - YOLOv11训练流程")
        print("=" * 60)
        
        # 1. 分析数据集
        self.analyze_dataset()
        
        # 2. 训练模型
        model, results = self.train_model()
        
        if model is None:
            print("训练失败，退出流程")
            return
        
        # 3. 评估模型
        metrics = self.evaluate_model()
        
        # 4. 导出模型（可选）
        export_format = self.config.get("export_format", "onnx")
        if export_format:
            self.export_model(format=export_format)
        
        print("\n" + "=" * 60)
        print("训练流程完成!")
        print(f"实验目录: {self.exp_dir}")
        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="钢铁缺陷检测YOLOv11训练脚本")
    parser.add_argument("--config", default="config/training_config.yaml", 
                       help="训练配置文件路径")
    parser.add_argument("--analyze-only", action="store_true",
                       help="仅分析数据集，不训练")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="仅评估模型，不训练")
    parser.add_argument("--export-only", action="store_true",
                       help="仅导出模型，不训练")
    parser.add_argument("--model-path", help="要评估或导出的模型路径")
    parser.add_argument("--export-format", default="onnx",
                       choices=["onnx", "engine", "torchscript", "tflite"],
                       help="导出模型格式")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = SteelDefectTrainer(args.config)
    
    if args.analyze_only:
        # 仅分析数据集
        trainer.analyze_dataset()
    elif args.evaluate_only:
        # 仅评估模型
        trainer.evaluate_model(args.model_path)
    elif args.export_only:
        # 仅导出模型
        trainer.export_model(args.model_path, args.export_format)
    else:
        # 运行完整流程
        trainer.run_pipeline()


if __name__ == "__main__":
    main()