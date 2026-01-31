# CASE-钢铁缺陷检测-P1

## 项目概述

这是一个专注于钢铁缺陷检测的计算机视觉项目，基于**YOLOv11**目标检测技术对钢铁表面缺陷进行自动识别和定位。项目采用最新的YOLOv11架构，结合针对工业图像优化的数据增强策略，实现对钢铁生产过程中6种常见缺陷的高精度检测。

### 项目特点
- **工业应用场景**: 钢铁生产质量在线检测
- **技术栈**: Ultralytics YOLOv11 / OpenCV / PyTorch
- **缺陷类型**: 轧入氧化皮、斑块、划痕、夹杂物、裂纹、点蚀表面
- **数据集**: NEU-DET钢铁表面缺陷数据集（1800张200×200灰度图像）
- **评估指标**: mAP、精度、召回率、F1分数、IoU
- **完整流程**: 提供数据转换、模型训练、评估、导出一站式解决方案

## 项目结构

```
CASE-钢铁缺陷检测-P1/
├── 钢铁缺陷检测.ipynb          # 主要分析笔记本（Jupyter Notebook）
├── README.md                       # 项目说明文档（本文件）
├── IFLOW.md                        # 项目交互指南（详细配置说明）
├── pyproject.toml                  # 项目依赖配置
├── uv.lock                         # UV依赖锁定文件
├── run_training.sh                 # 一键训练启动脚本
├── .venv/                          # Python虚拟环境目录
├── code/                           # 模型脚本目录 ⭐
│   ├── convert_voc_to_yolo.py     # VOC转YOLO格式转换器（已实现）
│   ├── train_yolov11.py           # YOLOv11训练与评估系统（已实现）
│   └── README.md                   # 代码说明文档
├── config/                         # 配置文件目录 ⭐
│   └── training_config.yaml        # YOLOv11训练配置文件（已优化）
├── data/                           # 原始数据文件目录
│   ├── train/                     # 训练集（1400张图像 + XML标注）
│   │   ├── IMAGES/               # 训练图像
│   │   └── ANNOTATIONS/          # PASCAL VOC XML标注文件
│   ├── test/                      # 测试集（400张图像）
│   │   └── IMAGES/               # 测试图像
│   ├── train.zip                  # 训练集压缩包
│   ├── test.zip                   # 测试集压缩包
│   └── yolo_format/               # 转换后的YOLO格式数据集 ⭐
│       ├── data.yaml             # 数据集配置文件
│       ├── images/               # 图像目录（train/val子目录）
│       └── labels/               # YOLO格式标注目录
├── docs/                           # 项目文档目录 ⭐
│   ├── 数据集分析报告.md           # NEU-DET数据集详细分析
│   ├── YOLO模型训练建议.md         # YOLO模型选择与优化建议
│   └── 项目总结.md                 # 项目完整总结文档
└── runs/                           # 训练结果目录 ⭐
    └── detect/                    # YOLOv11训练输出
        ├── steel_defect_yolo11n_20251221_020551/  # 训练运行1
        ├── steel_defect_yolo11n_20251221_021410/  # 训练运行2
        └── (更多训练运行...)
```

## 技术栈

### 深度学习框架
- **Ultralytics YOLOv11**: 最新的YOLO目标检测框架（核心）
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
- **tqdm**: 进度条显示

## 快速开始

### 环境设置
```bash
# 方法1：使用uv同步依赖（推荐）
uv sync

# 方法2：手动安装核心依赖
uv pip install ultralytics opencv-python pillow matplotlib seaborn tqdm pandas numpy scikit-learn pyyaml

# 激活虚拟环境
source .venv/bin/activate
```

### 数据集准备（自动）
```bash
# 运行一键训练脚本（自动检查环境、转换数据、开始训练）
./run_training.sh
```
*脚本会自动：*
1. 检查虚拟环境和依赖
2. 将VOC格式数据转换为YOLO格式
3. 按照8:2比例划分训练集和验证集
4. 启动YOLOv11训练

### 手动操作（可选）
```bash
# 1. 手动转换数据格式
.venv/bin/python code/convert_voc_to_yolo.py --data-root data --val-ratio 0.2

# 2. 验证数据集完整性
.venv/bin/python code/convert_voc_to_yolo.py --verify

# 3. 启动训练
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml

# 4. 仅评估模型（不训练）
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml --evaluate-only

# 5. 导出模型为ONNX格式
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml --export-only --export-format onnx
```

## 项目目标

### 性能目标
- **mAP@0.5**: > 0.85
- **精度**: > 90%
- **召回率**: > 85%
- **F1分数**: > 0.87
- **推理速度**: < 50ms/图像（在RTX 3060上）
- **模型大小**: < 10MB（YOLOv11n）

### 技术目标
1. 实现6类钢铁缺陷的准确检测和定位
2. 针对小目标（200×200图像中的缺陷）优化检测性能
3. 建立完整的工业缺陷检测流水线
4. 提供可复现的实验配置和训练脚本
5. 支持模型导出和部署

## 数据集说明

### NEU-DET数据集
- **来源**: 东北大学发布的钢铁表面缺陷数据集
- **图像数量**: 1800张（训练1400张，测试400张）
- **图像规格**: 200×200像素，灰度图像（1通道）
- **标注格式**: PASCAL VOC XML（原始），已提供YOLO格式转换

### 缺陷类别（6种）
| 类别ID | 英文名称 | 中文描述 | 典型特征 |
|--------|----------|----------|----------|
| 0 | rolled-in_scale | 轧入氧化皮 | 表面氧化皮压入 |
| 1 | patches | 斑块 | 不规则色斑区域 |
| 2 | scratches | 划痕 | 线性表面损伤 |
| 3 | inclusion | 夹杂物 | 材料内部杂质 |
| 4 | crazing | 裂纹 | 网状微小裂纹 |
| 5 | pitted_surface | 点蚀表面 | 点状腐蚀坑 |

### 数据分布
- **训练集**: 1400张图像（~78%）
- **验证集**: 280张图像（~16%，从训练集划分）
- **测试集**: 400张图像（~22%）
- **总缺陷实例**: 3256个（基于初步统计）

## 模型架构

### YOLOv11优势
本项目采用**YOLOv11**最新架构，具有以下优势：
- 🚀 **高性能**: 改进的骨干网络和检测头设计
- ⚡ **高效率**: 优化的推理速度和内存使用
- 🛠️ **易用性**: Ultralytics框架的简单统一API
- 📦 **可扩展性**: 支持nano/small/medium/large多种尺寸

### 当前配置
- **基础模型**: `yolo11n.pt`（nano版本，平衡速度与精度）
- **输入尺寸**: 200×200（保持原始图像尺寸，避免resize失真）
- **预训练**: 使用COCO数据集预训练权重进行迁移学习
- **优化器**: AdamW（适合小数据集）
- **学习率调度**: 余弦退火配合3个epoch的预热

### 针对钢铁缺陷的优化
1. **数据增强**: 针对灰度图像的HSV调整、旋转、平移
2. **损失函数**: 调整边界框、分类、DFL损失权重，优化小目标检测
3. **训练策略**: 早停策略（50个epoch无改善则停止）
4. **验证配置**: 每epoch验证，保存最佳模型

## 评估指标

### 目标检测核心指标
- **mAP@0.5**: IoU阈值为0.5时的平均精度（主要评估指标）
- **mAP@0.5:0.95**: IoU阈值从0.5到0.95的平均精度（严格评估）
- **精度**: 正确检测的比例（Precision）
- **召回率**: 真实缺陷被检测出的比例（Recall）
- **F1分数**: 精度和召回率的调和平均
- **IoU**: 检测框与真实框的交并比

### 性能指标
- **推理速度**: 每秒帧数（FPS），衡量实时性
- **模型大小**: 参数量（Params）和文件大小（MB）
- **训练时间**: 达到收敛所需的epoch数和总时间

## 配置文件详解

### `config/training_config.yaml`
这是针对钢铁缺陷检测优化的完整配置：
```yaml
# 核心配置
model: "yolo11n.pt"      # 使用YOLOv11 nano模型
imgsz: 200               # 保持200×200原始尺寸
epochs: 200              # 训练轮数
batch_size: 16           # 批次大小（根据显存调整）

# 优化器配置
optimizer: "AdamW"       # 适合小数据集的优化器
lr0: 0.001              # 初始学习率
cos_lr: true            # 余弦退火学习率调度

# 数据增强（针对工业图像）
hsv_h: 0.015            # 色调增强
hsv_s: 0.7              # 饱和度增强
hsv_v: 0.4              # 明度增强（模拟不同光照）
degrees: 10.0           # 旋转角度（±10度）

# 损失函数权重（小目标优化）
box: 7.5                # 边界框损失权重
cls: 0.5                # 分类损失权重
dfl: 1.5                # Distribution Focal Loss权重
```

### `pyproject.toml`依赖管理
```toml
dependencies = [
    "ultralytics>=8.2.0",  # YOLOv11支持
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    # ... 其他核心依赖
]
```

## 使用示例

### 训练模型
```python
# 使用提供的训练脚本
from code.train_yolov11 import SteelDefectTrainer

trainer = SteelDefectTrainer(config_path="config/training_config.yaml")
trainer.train()

# 或直接使用YOLO API
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="data/yolo_format/data.yaml", epochs=200, imgsz=200)
```

### 使用训练好的模型预测
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/detect/steel_defect_yolo11n_20251221_020551/weights/best.pt")

# 单张图像预测
results = model.predict("data/test/IMAGES/1400.jpg", conf=0.25)

# 可视化结果
results[0].show()

# 获取检测信息
boxes = results[0].boxes
print(f"检测到 {len(boxes)} 个缺陷")
for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"- 类别: {cls_id}, 置信度: {conf:.3f}")
```

### 批量预测
```bash
# 批量预测测试集
.venv/bin/python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/steel_defect_yolo11n_20251221_020551/weights/best.pt')
model.predict('data/test/IMAGES/', save=True, conf=0.25)
"
```

## 故障排除

### 常见问题

1. **内存不足错误**
   ```bash
   # 修改 config/training_config.yaml
   batch_size: 8  # 减小批次大小
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
   # 检查XML文件格式
   .venv/bin/python code/convert_voc_to_yolo.py --verify
   # 重新转换
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
   # 或使用CPU训练（速度较慢）
   device: "cpu"
   ```

## 扩展与定制

### 模型升级
```bash
# 尝试不同尺寸的YOLOv11模型
model: "yolo11s.pt"  # small版本，精度更高
model: "yolo11m.pt"  # medium版本
model: "yolo11l.pt"  # large版本，最高精度
```

### 自定义训练
1. 修改 `config/training_config.yaml` 中的参数
2. 调整数据增强策略以适应特定缺陷类型
3. 修改损失函数权重以优化特定指标
4. 添加自定义回调函数进行训练监控

### 部署选项
1. **ONNX格式**: 用于跨平台部署
2. **TensorRT**: 用于NVIDIA GPU加速
3. **OpenVINO**: 用于Intel硬件加速
4. **CoreML**: 用于Apple设备部署

## 贡献指南

### 开发规范
- 遵循PEP 8代码风格和类型注解（Python 3.11+）
- 使用Black格式化代码，Ruff进行代码检查
- 添加详细的文档字符串和类型提示
- 保持代码模块化、可重用、可测试

### 提交流程
1. Fork项目并创建功能分支
2. 编写清晰的提交信息和相关测试
3. 更新文档和示例代码
4. 提交Pull Request并描述变更内容

## 联系与支持

- **项目仓库**: https://github.com/lihaizhong/build-your-own-ai
- **问题反馈**: 通过GitHub Issues提交
- **数据集来源**: NEU-DET钢铁表面缺陷数据集
- **技术参考**: Ultralytics YOLOv11文档

## 许可证

本项目基于MIT许可证开源，可用于学术研究和商业应用。

---

*最后更新: 2025年12月21日*
*项目状态: 基于YOLOv11的完整实现*
*版本: v2.0 - YOLOv11目标检测版本*

### 更新日志
- **2025-12-21**: 全面更新为YOLOv11版本，反映实际项目状态
- **2025-12-20**: 初始创建
- **关键特性**:
  - 🎯 基于YOLOv11的6类钢铁缺陷检测
  - 🔄 完整的VOC到YOLO格式转换
  - ⚙️ 针对200×200灰度图像的优化配置
  - 🚀 一键训练脚本和详细使用指南
  - 📊 完整的评估和导出功能

### 注意事项
1. 使用项目虚拟环境（`.venv/bin/python`），避免系统Python冲突
2. 需要NVIDIA GPU以获得最佳训练速度（支持CPU训练）
3. 数据集已转换为YOLO格式，可直接用于训练
4. 所有配置针对钢铁缺陷检测优化，可根据需求调整