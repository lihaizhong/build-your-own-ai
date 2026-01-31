# CASE-钢铁缺陷检测-P1 项目指南 (YOLOv11版本)

## 项目概述

这是一个专注于钢铁缺陷检测的计算机视觉项目，基于**YOLOv11**目标检测技术对钢铁表面缺陷进行自动识别和定位。项目采用最新的YOLOv11架构，结合针对工业图像优化的数据增强策略，实现对钢铁生产过程中6种常见缺陷的高精度检测。

### 项目特点
- **工业应用场景**: 钢铁生产质量在线检测
- **技术栈**: Ultralytics YOLOv11 / OpenCV / PyTorch
- **缺陷类型**: 轧入氧化皮、斑块、划痕、夹杂物、裂纹、点蚀表面
- **数据集**: NEU-DET钢铁表面缺陷数据集（1800张200×200灰度图像）
- **评估指标**: mAP、精度、召回率、F1分数、IoU

## 项目结构

```
CASE-钢铁缺陷检测-P1/
├── 钢铁缺陷检测.ipynb          # 主要分析笔记本（Jupyter Notebook）
├── README.md                       # 项目说明文档
├── IFLOW.md                        # 本文件，项目交互指南
├── pyproject.toml                  # 项目依赖配置
├── uv.lock                         # UV依赖锁定文件
├── run_training.sh                 # 一键训练启动脚本
├── .venv/                          # Python虚拟环境目录
├── code/                           # 模型脚本目录 ⭐⭐⭐
│   ├── convert_voc_to_yolo.py     # VOC转YOLO格式转换器（已实现）
│   ├── train_yolov11.py           # YOLOv11训练与评估系统（已实现）
│   └── README.md                   # 代码说明文档
├── config/                         # 配置文件目录 ⭐⭐
│   └── training_config.yaml        # YOLOv11训练配置文件（已优化）
├── data/                           # 原始数据文件目录
│   ├── train/                     # 训练集（1400张图像 + XML标注）
│   │   ├── IMAGES/               # 训练图像
│   │   └── ANNOTATIONS/          # PASCAL VOC XML标注文件
│   ├── test/                      # 测试集（400张图像）
│   │   └── IMAGES/               # 测试图像
│   ├── train.zip                  # 训练集压缩包
│   ├── test.zip                   # 测试集压缩包
│   └── yolo_format/               # 转换后的YOLO格式数据集 ⭐⭐⭐
│       ├── data.yaml             # 数据集配置文件
│       ├── images/               # 图像目录（train/val子目录）
│       └── labels/               # YOLO格式标注目录
├── docs/                           # 项目文档目录 ⭐⭐⭐
│   ├── 数据集分析报告.md           # NEU-DET数据集详细分析
│   ├── YOLO模型训练建议.md         # YOLO模型选择与优化建议
│   └── 项目总结.md                 # 项目完整总结文档
└── runs/                           # 训练结果目录 ⭐⭐⭐
    └── detect/                    # YOLOv11训练输出
        ├── steel_defect_yolo11n_20251221_020551/  # 训练运行1
        ├── steel_defect_yolo11n_20251221_021410/  # 训练运行2
        └── (更多训练运行...)
```

## 技术栈

### 深度学习框架
- **Ultralytics YOLOv11**: 最新的YOLO目标检测框架
- **PyTorch**: 底层深度学习框架
- **Torchvision**: 计算机视觉工具库

### 数据处理与图像处理
- **OpenCV**: 图像处理和增强
- **Pillow**: Python图像处理库
- **NumPy**: 数值计算和数组操作

### 数据分析与可视化
- **Pandas**: 数据分析和处理
- **Matplotlib**: 图表生成和数据可视化
- **Seaborn**: 统计可视化
- **scikit-learn**: 机器学习工具和评估指标

### 环境与工具
- **UV**: 现代Python包管理器和虚拟环境工具
- **PyYAML**: YAML配置文件解析

## 快速开始

### 环境设置
```bash
# 使用uv同步依赖（推荐）
uv sync

# 或手动安装核心依赖
uv pip install ultralytics opencv-python pillow matplotlib seaborn tqdm pandas numpy scikit-learn pyyaml

# 激活虚拟环境
source .venv/bin/activate
```

### 一键训练（推荐）
```bash
# 使用启动脚本（自动检查环境、转换数据、开始训练）
./run_training.sh

# 或手动运行训练
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml
```

### 数据集准备
```bash
# 手动转换数据格式（VOC → YOLO）
.venv/bin/python code/convert_voc_to_yolo.py --data-root data --val-ratio 0.2

# 验证数据集
.venv/bin/python code/convert_voc_to_yolo.py --verify
```

### 模型评估与使用
```bash
# 评估训练好的模型
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml --evaluate-only

# 导出模型为ONNX格式
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml --export-only --export-format onnx

# 仅分析数据集（不训练）
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml --analyze-only
```

## 项目目标

### 性能目标
- **mAP@0.5**: > 0.85
- **精度**: > 90%
- **召回率**: > 85%
- **F1分数**: > 0.87
- **推理速度**: < 50ms/图像（在RTX 3060上）

### 技术目标
1. 实现6类钢铁缺陷的准确检测和定位
2. 优化小目标检测性能（200×200图像中的缺陷）
3. 针对灰度工业图像优化数据增强策略
4. 建立完整的训练-评估-导出流水线
5. 提供可复现的实验配置

## 数据集说明

### 数据来源
- **NEU-DET数据集**: 东北大学发布的钢铁表面缺陷数据集
- **图像数量**: 1800张（训练1400张，测试400张）
- **图像规格**: 200×200像素，灰度图像（1通道）
- **标注格式**: PASCAL VOC XML（原始），已转换为YOLO格式

### 缺陷类别（6种）
| 类别ID | 英文名称 | 中文描述 | 实例数量 |
|--------|----------|----------|----------|
| 0 | rolled-in_scale | 轧入氧化皮 | 待统计 |
| 1 | patches | 斑块 | 待统计 |
| 2 | scratches | 划痕 | 待统计 |
| 3 | inclusion | 夹杂物 | 待统计 |
| 4 | crazing | 裂纹 | 待统计 |
| 5 | pitted_surface | 点蚀表面 | 待统计 |

### 数据分布
- **训练集**: 1400张图像（~78%）
- **验证集**: 280张图像（~16%，从训练集划分）
- **测试集**: 400张图像（~22%）
- **总实例数**: 3256个缺陷实例（基于初步分析）

## 模型架构

### YOLOv11模型
项目基于**YOLOv11**最新架构，提供以下优势：
- **高精度**: 改进的骨干网络和检测头
- **高效率**: 优化的推理速度和内存使用
- **易用性**: Ultralytics框架的简单API
- **可扩展性**: 支持多种模型尺寸（nano, small, medium, large）

### 当前配置
- **基础模型**: `yolo11n.pt`（nano版本，平衡速度与精度）
- **输入尺寸**: 200×200（保持原始尺寸）
- **预训练权重**: 使用COCO预训练权重迁移学习

### 优化策略
1. **数据增强**: 针对工业图像的色调、饱和度、明度调整
2. **学习率调度**: 余弦退火配合学习率预热
3. **损失函数优化**: 调整边界框、分类、DFL损失权重
4. **早停策略**: 基于验证集性能的智能早停
5. **小目标优化**: 针对200×200图像的特殊增强策略

## 评估指标

### 目标检测指标
- **mAP@0.5**: 平均精度（IoU阈值0.5）
- **mAP@0.5:0.95**: 平均精度（IoU阈值0.5-0.95，步长0.05）
- **精度**: 正确检测的比例
- **召回率**: 真实缺陷被检测出的比例
- **F1分数**: 精度和召回率的调和平均
- **IoU**: 检测框与真实框的交并比

### 性能指标
- **推理速度**: 每秒帧数（FPS）
- **模型大小**: 参数量和文件大小
- **训练时间**: 达到收敛所需的epoch数

## 使用指南

### 数据预处理
```python
# 使用提供的转换脚本
from code.convert_voc_to_yolo import SteelDefectConverter

converter = SteelDefectConverter(data_root="data", val_ratio=0.2)
converter.convert_all()
```

### 模型训练
```python
# 使用训练脚本
import subprocess

# 启动训练
subprocess.run([
    ".venv/bin/python", "code/train_yolov11.py",
    "--config", "config/training_config.yaml"
])

# 或直接使用YOLO API
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data/yolo_format/data.yaml", epochs=200, imgsz=200)
```

### 模型评估
```python
# 使用提供的评估功能
from code.train_yolov11 import SteelDefectTrainer

trainer = SteelDefectTrainer(config_path="config/training_config.yaml")
metrics = trainer.evaluate_model()
print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
```

### 推理预测
```python
# 使用训练好的模型进行预测
from ultralytics import YOLO

model = YOLO("runs/detect/steel_defect_yolo11n_20251221_020551/weights/best.pt")
results = model.predict("data/test/IMAGES/1400.jpg", conf=0.25)

# 可视化结果
results[0].show()
```

## 配置文件详解

### `config/training_config.yaml`
这是针对钢铁缺陷检测优化的完整配置：
- **数据增强**: 针对灰度图像的HSV调整、旋转、平移
- **学习率**: AdamW优化器，余弦退火调度
- **损失权重**: 针对小目标优化的边界框和分类损失
- **早停策略**: 50个epoch无改善则停止训练
- **验证配置**: 每epoch验证，保存最佳模型

### `pyproject.toml`
项目依赖管理文件：
- **核心依赖**: ultralytics>=8.2.0（支持YOLOv11）
- **开发工具**: jupyter, black, ruff, pytest
- **代码规范**: Black代码格式化，Ruff代码检查

## 开发规范

### 代码规范
- 遵循PEP 8代码风格
- 使用Python 3.11+类型注解
- 添加详细的文档字符串
- 保持代码模块化和可重用

### 文件命名规范
- 模型脚本: `[功能名称]_[技术].py` (如 `train_yolov11.py`)
- 工具脚本: `[功能名称].py` (如 `convert_voc_to_yolo.py`)
- 配置文件: `[功能名称]_config.yaml`
- 模型文件: 保存在 `runs/detect/[实验名称]/weights/`

### 实验管理
- 每次训练生成独立的实验目录
- 保存完整的训练配置和日志
- 记录关键性能指标和可视化结果
- 使用随机种子确保可复现性

## 故障排除

### 常见问题

1. **内存不足错误**
   ```bash
   # 减小批次大小
   # 修改 config/training_config.yaml 中的 batch_size
   batch_size: 8  # 从16减小到8
   ```

2. **训练不收敛**
   ```bash
   # 调整学习率
   lr0: 0.0005  # 减小学习率
   # 或增加训练轮数
   epochs: 300
   ```

3. **数据集转换失败**
   ```bash
   # 手动检查XML文件格式
   .venv/bin/python code/convert_voc_to_yolo.py --verify
   # 重新转换数据集
   .venv/bin/python code/convert_voc_to_yolo.py --force-convert
   ```

4. **依赖安装问题**
   ```bash
   # 使用uv解决依赖冲突
   uv pip install --upgrade ultralytics
   # 或重新创建虚拟环境
   rm -rf .venv && uv sync
   ```

5. **GPU显存不足**
   ```bash
   # 使用更小的模型
   model: "yolo11n.pt"  # 确保使用nano版本
   # 或使用CPU训练（慢）
   device: "cpu"
   ```

### 调试技巧
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据集统计
from code.convert_voc_to_yolo import SteelDefectConverter
converter = SteelDefectConverter()
stats = converter.analyze_dataset()
print(f"图像数量: {stats['total_images']}")
print(f"缺陷实例: {stats['total_instances']}")

# 验证模型加载
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
print(f"模型参数: {sum(p.numel() for p in model.parameters())}")
```

## 扩展计划

### 短期优化（1-2周）
1. **模型升级**: 尝试YOLOv11 small/medium/large版本
2. **数据增强**: 添加针对钢铁缺陷的特殊增强
3. **超参数调优**: 使用贝叶斯优化搜索最佳参数
4. **集成学习**: 多模型集成提升稳定性
5. **实时推理**: 优化推理速度和内存使用

### 中期改进（1-2月）
1. **模型压缩**: 知识蒸馏或量化压缩
2. **多尺度训练**: 适应不同分辨率的输入
3. **不确定性估计**: 添加检测置信度校准
4. **主动学习**: 基于不确定性的数据标注策略
5. **部署优化**: ONNX/TensorRT加速部署

### 长期规划（3-6月）
1. **生产部署**: Docker容器化部署
2. **API服务**: RESTful API提供检测服务
3. **Web界面**: 可视化上传和结果展示
4. **移动端适配**: 边缘设备部署
5. **持续学习**: 在线更新模型适应新缺陷

## 性能基准

### 预期性能（YOLOv11n）
| 指标 | 目标值 | 当前值 | 状态 |
|------|--------|--------|------|
| mAP@0.5 | >0.85 | 待训练 | ⏳ |
| 精度 | >90% | 待训练 | ⏳ |
| 召回率 | >85% | 待训练 | ⏳ |
| FPS | >20 | 待测试 | ⏳ |
| 模型大小 | <10MB | 待导出 | ⏳ |

### 训练时间估计
- **硬件**: NVIDIA RTX 3060 (12GB VRAM)
- **批次大小**: 16
- **训练轮数**: 200
- **预计时间**: 2-4小时

## 联系支持

- **项目仓库**: https://github.com/lihaizhong/build-your-own-ai
- **问题反馈**: 通过GitHub Issues提交
- **技术讨论**: 项目相关技术社区
- **数据集来源**: NEU-DET钢铁表面缺陷数据集

---

*最后更新: 2025年12月21日*
*项目状态: 基于YOLOv11的完整实现*
*版本: v2.0 - YOLOv11目标检测版本*

### 更新日志
- **2025-12-21**: 全面更新为YOLOv11版本，反映实际项目状态
- **2025-12-20**: 初始创建，基于CNN/ResNet分类架构
- **关键变更**:
  - 从图像分类转向目标检测
  - 采用YOLOv11替代传统CNN架构
  - 实现VOC到YOLO格式的完整转换
  - 提供一键训练脚本和优化配置
  - 建立完整的训练-评估-导出流水线

### 注意事项
1. 本项目使用**虚拟环境管理**，确保使用`.venv/bin/python`而非系统Python
2. 所有脚本已适配**YOLOv11最新API**，需要ultralytics>=8.2.0
3. 数据集已成功转换为**YOLO格式**，可直接用于训练
4. 训练配置针对**200×200灰度图像**优化，无需调整尺寸
5. 项目提供完整的**中文文档**和**故障排除指南**