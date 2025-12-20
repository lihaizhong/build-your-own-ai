# CASE-钢铁缺陷检测-P1

## 项目概述

这是一个专注于钢铁缺陷检测的计算机视觉项目，基于深度学习技术对钢铁表面缺陷进行自动识别和分类。项目采用先进的卷积神经网络（CNN）架构，结合数据增强和迁移学习技术，实现对钢铁生产过程中常见缺陷的高精度检测。

### 项目特点
- **工业应用场景**: 钢铁生产质量检测
- **技术栈**: PyTorch/OpenCV/Scikit-learn
- **缺陷类型**: 裂纹、划痕、孔洞、氧化等常见缺陷
- **数据规模**: 大规模钢铁表面图像数据集
- **评估指标**: 准确率、召回率、F1分数、mAP

## 项目结构

```
CASE-钢铁缺陷检测-P1/
├── 钢铁缺陷检测.ipynb          # 主要分析笔记本（Jupyter Notebook）
├── README.md                       # 项目说明文档
├── IFLOW.md                        # 项目交互指南
├── code/                           # 模型脚本目录
│   ├── cnn_v1_model.py            # CNN基础模型 v1.0
│   ├── resnet_v1_model.py         # ResNet迁移学习模型 v1.0
│   ├── efficientnet_v1_model.py   # EfficientNet高效模型 v1.0
│   ├── data_augmentation.py       # 数据增强脚本
│   └── model_evaluation.py        # 模型评估脚本
├── data/                           # 原始数据文件目录
│   ├── train/                     # 训练集图像
│   ├── test/                      # 测试集图像
│   ├── validation/                # 验证集图像
│   └── annotations/               # 标注文件
├── docs/                           # 项目文档目录
│   ├── 数据预处理报告.md           # 数据预处理分析
│   ├── 模型对比分析报告.md         # 模型性能对比
│   └── 缺陷检测技术方案.md         # 技术方案设计
├── feature/                        # 特征工程目录
│   ├── image_preprocessing.py     # 图像预处理工具
│   ├── feature_extraction.py      # 特征提取工具
│   └── visualization.py           # 可视化工具
├── model/                          # 训练好的模型文件目录
│   ├── cnn_v1_model.pth           # CNN模型权重
│   ├── resnet_v1_model.pth        # ResNet模型权重
│   └── efficientnet_v1_model.pth  # EfficientNet模型权重
├── prediction_result/              # 预测结果目录
│   ├── test_predictions.csv       # 测试集预测结果
│   ├── confusion_matrix.png       # 混淆矩阵可视化
│   └── performance_metrics.csv    # 性能指标文件
└── user_data/                      # 数据分析和可视化结果目录
    ├── training_curves.png        # 训练曲线图
    ├── defect_samples.png         # 缺陷样本可视化
    └── feature_maps.png           # 特征图可视化
```

## 技术栈

### 深度学习框架
- **PyTorch**: 主要深度学习框架
- **Torchvision**: 计算机视觉工具库
- **OpenCV**: 图像处理库
- **PIL**: Python图像处理库

### 数据处理
- **Pandas**: 数据分析和处理
- **NumPy**: 数值计算
- **Scikit-learn**: 机器学习工具

### 可视化
- **Matplotlib**: 图表生成
- **Seaborn**: 统计可视化
- **TensorBoard**: 训练过程可视化

## 快速开始

### 环境要求
```bash
# 使用 uv 管理环境
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 运行基础模型
```bash
# 训练CNN基础模型
uv run python code/cnn_v1_model.py

# 运行数据增强
uv run python code/data_augmentation.py

# 评估模型性能
uv run python code/model_evaluation.py
```

### 查看预测结果
```bash
# 查看测试集预测结果
cat prediction_result/test_predictions.csv

# 查看性能指标
cat prediction_result/performance_metrics.csv
```

## 项目目标

### 性能目标
- **准确率**: > 95%
- **召回率**: > 90%
- **F1分数**: > 92%
- **推理速度**: < 100ms/图像

### 技术目标
1. 实现多类别缺陷分类
2. 优化模型推理速度
3. 提高小样本缺陷检测能力
4. 开发实时检测系统原型

## 数据集说明

### 数据来源
- 钢铁生产现场采集图像
- 公开钢铁缺陷数据集
- 合成数据增强

### 数据特点
- **图像尺寸**: 统一为 224×224 或 256×256
- **缺陷类型**: 6-10种常见缺陷类别
- **数据分布**: 平衡或适当的不平衡分布
- **标注格式**: COCO格式或YOLO格式

## 模型架构

### 基础模型
1. **CNN基础架构**: 自定义卷积神经网络
2. **ResNet迁移学习**: 基于预训练的ResNet模型
3. **EfficientNet**: 高效网络架构
4. **YOLO目标检测**: 实时缺陷检测

### 优化策略
- 数据增强（旋转、翻转、色彩调整）
- 迁移学习和微调
- 学习率调度
- 早停策略

## 评估指标

### 分类指标
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)

### 检测指标
- 平均精度 (mAP)
- 交并比 (IoU)
- 检测速度 (FPS)

## 贡献指南

### 开发规范
- 遵循PEP 8代码风格
- 使用类型注解（Python 3.11+）
- 添加详细的文档字符串
- 保持代码模块化和可重用

### 提交要求
1. Fork项目并创建功能分支
2. 编写测试用例
3. 更新相关文档
4. 提交Pull Request

## 联系支持

- 项目仓库: https://github.com/lihaizhong/build-your-own-ai
- 问题反馈: 通过GitHub Issues提交
- 技术讨论: 项目相关技术社区

---

*最后更新: 2025年12月19日*
*项目状态: 初始创建*
*版本: v1.0*