# 钢铁缺陷检测 - YOLOv11实现

## 项目概述

使用YOLOv11实现钢铁表面缺陷检测，基于NEU-DET数据集。支持6种常见钢铁缺陷的检测：
1. rolled-in_scale (轧入氧化皮)
2. patches (斑块)
3. scratches (划痕)
4. inclusion (夹杂物)
5. crazing (裂纹)
6. pitted_surface (点蚀表面)

## 文件结构

```
code/
├── convert_voc_to_yolo.py     # VOC转YOLO格式转换器
├── train_yolov11.py          # YOLOv11训练脚本
└── README.md                 # 本文件

config/
└── training_config.yaml      # 训练配置文件

data/
├── train/                    # 原始训练数据
├── test/                     # 原始测试数据
└── yolo_format/              # YOLO格式数据（自动生成）
    ├── images/
    │   ├── train/           # 训练图像
    │   ├── val/             # 验证图像
    │   └── test/            # 测试图像
    ├── labels/
    │   ├── train/           # 训练标注
    │   └── val/             # 验证标注
    └── data.yaml            # YOLO数据配置文件

docs/
├── 数据集分析报告.md         # 数据集详细分析
└── YOLO模型训练建议.md      # 训练建议和最佳实践
```

## 快速开始

### 1. 环境检查

项目已配置完整的Python虚拟环境，依赖已安装。检查环境：

```bash
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai/practice/CASE-钢铁缺陷检测-P1

# 检查虚拟环境
ls .venv/bin/python

# 检查依赖
.venv/bin/python -c "import ultralytics; print(f'Ultralytics版本: {ultralytics.__version__}')"
```

如果依赖未安装，可在项目目录下运行：
```bash
uv pip install --python .venv/bin/python ultralytics opencv-python pillow matplotlib seaborn tqdm pandas numpy scikit-learn pyyaml
```

### 2. 准备数据集

数据集已自动转换为YOLO格式。如果需要重新转换：

```bash
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai/practice/CASE-钢铁缺陷检测-P1
.venv/bin/python code/convert_voc_to_yolo.py --data-root data --val-ratio 0.2 --verify
```

### 3. 训练模型

使用一键训练脚本（推荐）：

```bash
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai/practice/CASE-钢铁缺陷检测-P1
./run_training.sh
```

或手动运行：

```bash
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai/practice/CASE-钢铁缺陷检测-P1
.venv/bin/python code/train_yolov11.py --config config/training_config.yaml
```

### 4. 监控训练

训练过程中会生成：
- 训练日志和损失曲线
- 验证结果和评估指标
- 最佳模型检查点
- 类别分布分析

## 训练配置说明

### 模型选择

默认使用`yolo11n.pt`（nano模型），可根据需求调整：

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| yolo11n.pt | ~3M | 最快 | 较低 | 快速原型/边缘设备 |
| yolo11s.pt | ~11M | 快 | 中等 | 平衡性能 |
| yolo11m.pt | ~26M | 中等 | 高 | 生产环境 |
| yolo11l.pt | ~44M | 较慢 | 最高 | 研究/高精度需求 |

修改`config/training_config.yaml`中的`model`参数即可切换模型。

### 训练参数优化

针对钢铁缺陷检测的特殊优化：

1. **图像尺寸**: 保持200×200原尺寸，避免resize失真
2. **数据增强**: 针对工业场景设计（光照变化、旋转、翻转）
3. **学习率调度**: 余弦退火适合小数据集
4. **损失函数权重**: 针对小目标检测优化
5. **早停策略**: 50个epoch无改善则停止

### 针对小目标的特殊处理

钢铁缺陷通常是小目标，配置文件已包含以下优化：

```yaml
# 损失函数权重调整
box: 7.5    # 提高边界框损失权重
cls: 0.5    # 适当降低分类损失权重
dfl: 1.5    # 使用Distribution Focal Loss

# 数据增强
scale: 0.5  # 缩放增强，帮助模型学习不同尺度
```

## 高级用法

### 仅分析数据集

```bash
python code/train_yolov11.py --config config/training_config.yaml --analyze-only
```

### 仅评估模型

```bash
python code/train_yolov11.py --config config/training_config.yaml --evaluate-only --model-path runs/detect/steel_defect_yolo11n_*/weights/best.pt
```

### 导出模型

支持多种格式导出：

```bash
# 导出为ONNX
python code/train_yolov11.py --config config/training_config.yaml --export-only --export-format onnx

# 导出为TensorRT
python code/train_yolov11.py --config config/training_config.yaml --export-only --export-format engine

# 导出为TorchScript
python code/train_yolov11.py --config config/training_config.yaml --export-only --export-format torchscript
```

### 自定义训练

创建自定义配置文件：

```yaml
# my_config.yaml
model: "yolo11s.pt"  # 使用small模型
epochs: 300          # 更多epoch
batch_size: 32       # 更大batch size
lr0: 0.0005          # 更小的学习率
```

然后运行：

```bash
python code/train_yolov11.py --config my_config.yaml
```

## 性能指标

训练完成后会生成以下评估指标：

1. **mAP50-95**: 主要评估指标（IoU从0.5到0.95的平均精度）
2. **mAP50**: IoU=0.5时的平均精度
3. **精确率**: 正确检测的比例
4. **召回率**: 检测出所有缺陷的比例
5. **F1分数**: 精确率和召回率的调和平均

## 故障排除

### 常见问题

1. **内存不足**
   - 减小`batch_size`（默认16）
   - 使用更小的模型（yolo11n）
   - 启用梯度累积（在配置中添加`accumulate: 2`）

2. **训练不收敛**
   - 检查学习率（默认0.001）
   - 增加数据增强
   - 增加训练epoch（默认200）

3. **过拟合**
   - 增加数据增强
   - 使用早停（默认patience=50）
   - 增加权重衰减（weight_decay）

4. **小目标检测效果差**
   - 确保使用原尺寸训练（imgsz=200）
   - 调整损失函数权重
   - 增加针对小目标的数据增强

### 日志和调试

训练日志保存在：
- `runs/detect/实验名称/training_log.csv`
- `runs/detect/实验名称/events.out.tfevents.*`（TensorBoard）

使用TensorBoard监控训练：

```bash
tensorboard --logdir runs/detect
```

## 部署建议

### 生产环境

1. **模型选择**: 使用yolo11s或yolo11m平衡速度和精度
2. **导出格式**: ONNX或TensorRT以获得最佳推理性能
3. **后处理**: 设置合适的置信度阈值（建议0.25-0.5）
4. **监控**: 定期评估模型性能，监控误报和漏报

### 边缘设备

1. **模型量化**: 使用INT8量化减少模型大小
2. **TensorRT**: 在NVIDIA设备上使用TensorRT加速
3. **OpenVINO**: 在Intel设备上使用OpenVINO
4. **TFLite**: 在移动设备上使用TensorFlow Lite

## 后续改进

1. **模型集成**: 尝试YOLOv10、YOLO-World等最新模型
2. **主动学习**: 针对难样本进行主动学习
3. **领域自适应**: 适应不同钢铁厂的生产环境
4. **实时检测**: 优化推理速度，实现实时检测

## 参考资料

1. [Ultralytics YOLOv11文档](https://docs.ultralytics.com/)
2. [NEU-DET数据集论文](https://arxiv.org/abs/1901.00611)
3. [YOLO小目标检测优化](https://arxiv.org/abs/2107.08430)
4. [工业视觉检测最佳实践](https://ieeexplore.ieee.org/document/9093283)

## 许可证

本项目基于MIT许可证开源。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。