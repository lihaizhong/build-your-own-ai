#!/usr/bin/env python3
"""
将PASCAL VOC XML格式转换为YOLO格式
用于钢铁缺陷检测数据集
"""

import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import random
from tqdm import tqdm


class SteelDefectConverter:
    def __init__(self, data_root="data"):
        """
        初始化转换器
        
        Args:
            data_root: 数据根目录
        """
        self.data_root = Path(data_root)
        self.train_images_dir = self.data_root / "train" / "IMAGES"
        self.train_annotations_dir = self.data_root / "train" / "ANNOTATIONS"
        self.test_images_dir = self.data_root / "test" / "IMAGES"
        
        # 缺陷类别映射
        self.class_map = {
            "rolled-in_scale": 0,
            "patches": 1,
            "scratches": 2,
            "inclusion": 3,
            "crazing": 4,
            "pitted_surface": 5
        }
        
        # 反向映射（用于验证）
        self.id_to_class = {v: k for k, v in self.class_map.items()}
        
        # 输出目录
        self.output_dir = self.data_root / "yolo_format"
        self.yolo_images_dir = self.output_dir / "images"
        self.yolo_labels_dir = self.output_dir / "labels"
        
    def create_directory_structure(self):
        """创建YOLO格式的目录结构"""
        directories = [
            self.output_dir,
            self.yolo_images_dir / "train",
            self.yolo_images_dir / "val",
            self.yolo_images_dir / "test",
            self.yolo_labels_dir / "train",
            self.yolo_labels_dir / "val",
            self.yolo_labels_dir / "test",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {directory}")
    
    def convert_voc_to_yolo(self, xml_path, output_txt_path):
        """
        转换单个VOC XML文件到YOLO格式
        
        Args:
            xml_path: VOC XML文件路径
            output_txt_path: 输出YOLO格式文件路径
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像尺寸
            size = root.find('size')
            if size is None:
                print(f"警告: {xml_path} 中没有size信息，使用默认200×200")
                width, height = 200, 200
            else:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            
            annotations = []
            
            # 处理每个缺陷对象
            for obj in root.findall('object'):
                # 获取类别
                cls_name = obj.find('name').text
                if cls_name not in self.class_map:
                    print(f"警告: 未知类别 '{cls_name}'，跳过")
                    continue
                
                cls_id = self.class_map[cls_name]
                
                # 获取边界框
                bbox = obj.find('bndbox')
                if bbox is None:
                    print(f"警告: {xml_path} 中对象没有边界框，跳过")
                    continue
                
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # 转换为YOLO格式 (cx, cy, w, h)
                cx = (xmin + xmax) / 2 / width
                cy = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                # 确保坐标在[0,1]范围内
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            # 写入YOLO格式文件
            if annotations:
                with open(output_txt_path, 'w') as f:
                    f.write('\n'.join(annotations))
                return True
            else:
                print(f"警告: {xml_path} 中没有有效的标注")
                return False
                
        except Exception as e:
            print(f"错误: 处理 {xml_path} 时出错: {e}")
            return False
    
    def split_train_val(self, train_files, val_ratio=0.2):
        """
        分割训练集和验证集
        
        Args:
            train_files: 训练集文件列表
            val_ratio: 验证集比例
        
        Returns:
            train_split, val_split: 分割后的文件列表
        """
        random.seed(42)  # 固定随机种子确保可重复性
        random.shuffle(train_files)
        
        val_size = int(len(train_files) * val_ratio)
        val_split = train_files[:val_size]
        train_split = train_files[val_size:]
        
        return train_split, val_split
    
    def process_dataset(self, val_ratio=0.2):
        """
        处理整个数据集
        
        Args:
            val_ratio: 验证集比例
        """
        print("=" * 60)
        print("开始转换钢铁缺陷检测数据集到YOLO格式")
        print("=" * 60)
        
        # 1. 创建目录结构
        self.create_directory_structure()
        
        # 2. 处理训练集
        print("\n处理训练集...")
        train_xml_files = list(self.train_annotations_dir.glob("*.xml"))
        print(f"找到 {len(train_xml_files)} 个训练标注文件")
        
        # 分割训练集和验证集
        train_files, val_files = self.split_train_val(train_xml_files, val_ratio)
        print(f"训练集: {len(train_files)} 个文件")
        print(f"验证集: {len(val_files)} 个文件")
        
        # 处理训练集
        train_success = 0
        for xml_file in tqdm(train_files, desc="转换训练集"):
            # 对应的图像文件
            img_name = xml_file.stem + ".jpg"
            img_path = self.train_images_dir / img_name
            
            if not img_path.exists():
                print(f"警告: 图像文件不存在 {img_path}")
                continue
            
            # 输出路径
            yolo_txt_path = self.yolo_labels_dir / "train" / f"{xml_file.stem}.txt"
            yolo_img_path = self.yolo_images_dir / "train" / img_name
            
            # 转换标注
            if self.convert_voc_to_yolo(xml_file, yolo_txt_path):
                # 复制图像文件
                shutil.copy2(img_path, yolo_img_path)
                train_success += 1
        
        # 处理验证集
        val_success = 0
        for xml_file in tqdm(val_files, desc="转换验证集"):
            img_name = xml_file.stem + ".jpg"
            img_path = self.train_images_dir / img_name
            
            if not img_path.exists():
                continue
            
            yolo_txt_path = self.yolo_labels_dir / "val" / f"{xml_file.stem}.txt"
            yolo_img_path = self.yolo_images_dir / "val" / img_name
            
            if self.convert_voc_to_yolo(xml_file, yolo_txt_path):
                shutil.copy2(img_path, yolo_img_path)
                val_success += 1
        
        # 3. 处理测试集
        print("\n处理测试集...")
        test_img_files = list(self.test_images_dir.glob("*.jpg"))
        print(f"找到 {len(test_img_files)} 个测试图像文件")
        
        test_success = 0
        for img_path in tqdm(test_img_files, desc="处理测试集"):
            yolo_img_path = self.yolo_images_dir / "test" / img_path.name
            shutil.copy2(img_path, yolo_img_path)
            test_success += 1
        
        # 4. 创建data.yaml配置文件
        self.create_yaml_config()
        
        # 5. 统计信息
        print("\n" + "=" * 60)
        print("转换完成！统计信息:")
        print("=" * 60)
        print(f"训练集: {train_success} 个样本")
        print(f"验证集: {val_success} 个样本")
        print(f"测试集: {test_success} 个样本")
        print(f"总样本数: {train_success + val_success + test_success}")
        print(f"\n类别映射:")
        for name, idx in self.class_map.items():
            print(f"  {idx}: {name}")
        print(f"\nYOLO格式数据保存在: {self.output_dir}")
        print(f"配置文件: {self.output_dir / 'data.yaml'}")
    
    def create_yaml_config(self):
        """创建YOLO数据配置文件"""
        import datetime
        yaml_content = f"""# 钢铁缺陷检测数据集配置
# 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# 数据集路径
path: {self.output_dir.absolute()}  # 数据集根目录
train: images/train  # 训练集图像路径
val: images/val      # 验证集图像路径
test: images/test    # 测试集图像路径

# 类别信息
nc: {len(self.class_map)}  # 类别数量
names: {list(self.class_map.keys())}  # 类别名称

# 图像参数
imgsz: 200  # 图像尺寸（保持原尺寸）
batch: 16   # 批次大小（根据显存调整）

# 数据增强配置（可选）
# augmentation:
#   hsv_h: 0.015  # 色调增强
#   hsv_s: 0.7    # 饱和度增强
#   hsv_v: 0.4    # 明度增强
#   degrees: 10.0 # 旋转角度
#   translate: 0.1 # 平移
#   scale: 0.5    # 缩放
#   fliplr: 0.5   # 水平翻转概率

# 数据集信息
# 来源: NEU-DET钢铁缺陷检测数据集
# 图像尺寸: 200×200 灰度图像
# 缺陷类型: 6种常见钢铁表面缺陷
"""
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"创建配置文件: {yaml_path}")
    
    def verify_conversion(self, sample_size=5):
        """
        验证转换结果
        
        Args:
            sample_size: 抽样验证的样本数量
        """
        print("\n" + "=" * 60)
        print("验证转换结果")
        print("=" * 60)
        
        # 检查目录结构
        required_dirs = [
            self.yolo_images_dir / "train",
            self.yolo_images_dir / "val",
            self.yolo_images_dir / "test",
            self.yolo_labels_dir / "train",
            self.yolo_labels_dir / "val",
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print(f"✓ {dir_path.name}: {file_count} 个文件")
            else:
                print(f"✗ {dir_path.name}: 目录不存在")
        
        # 抽样检查标注文件
        train_label_files = list((self.yolo_labels_dir / "train").glob("*.txt"))
        if train_label_files:
            print(f"\n抽样检查训练集标注（{sample_size}个样本）:")
            for i, label_file in enumerate(train_label_files[:sample_size]):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    img_name = label_file.stem + ".jpg"
                    print(f"  {img_name}: {len(lines)} 个标注")
                    
                    # 显示第一个标注的详细信息
                    if lines:
                        parts = lines[0].strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            cls_name = self.id_to_class.get(cls_id, "未知")
                            print(f"    类别: {cls_name} (ID: {cls_id})")
                            print(f"    中心: ({cx:.3f}, {cy:.3f})")
                            print(f"    尺寸: {w:.3f}×{h:.3f}")
        
        # 检查配置文件
        yaml_path = self.output_dir / "data.yaml"
        if yaml_path.exists():
            print(f"\n✓ 配置文件: {yaml_path}")
            with open(yaml_path, 'r') as f:
                content = f.read()
                if "nc: 6" in content and "names:" in content:
                    print("  配置验证通过")
                else:
                    print("  配置验证失败")
        else:
            print(f"\n✗ 配置文件不存在: {yaml_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="转换VOC格式到YOLO格式")
    parser.add_argument("--data-root", default="data", help="数据根目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--verify", action="store_true", help="验证转换结果")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = SteelDefectConverter(args.data_root)
    
    # 处理数据集
    converter.process_dataset(val_ratio=args.val_ratio)
    
    # 验证结果
    if args.verify:
        converter.verify_conversion()


if __name__ == "__main__":
    main()
