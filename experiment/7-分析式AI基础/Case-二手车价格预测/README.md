# 二手车价格预测项目

> 专注于探索机器学习在二手车价格预测中的能力和优化策略

## 项目概述

本项目通过机器学习方法预测二手车的交易价格，主要探索**机器学习**在这一任务中的表现和优化策略。项目包含了完整的数据预处理、特征工程、模型训练和性能评估流程。

## 数据来源

- [【AI入门系列】车市先知：二手车价格预测学习赛](https://tianchi.aliyun.com/competition/entrance/231784/information)

## 考试排名

1. modeling_v17.py
    - 分数：531.0229
1. modeling_v17_fast.py
    - 分数：535.0420
2. modeling_v16.py
    - 分数：539.8390
3. modeling_v12.py
    - 分数：605.7830
4. modeling_v14.py
    - 分数：649.1365

## 项目结构

```
Case-二手车价格预测/
├── README.md                 # 项目说明
├── code/                     # 核心功能模块
│   ├── main.py
│   ├── requirements.txt
│   └── utilities/                # 工具集合
│       ├── data_analysis_tools.py # 数据分析工具
│       ├── model_validation_tools.py # 模型验证工具
│       └── feature_tools.py      # 特征工程工具
├── data/                     # 数据文件
│   ├── used_car_train_20200313.csv # 训练数据
│   ├── used_car_testB_20200421.csv # 测试数据
│   └── used_car_sample_submit.csv  # 提交样例
├── docs/                     # 文档报告
│   ├── 项目方案总结报告.md
│   ├── 诊断分析报告.md
│   └── ...
├── prediction-result/        # 预测结果
│   └── rf_result_*.csv
├── user_data/                # 用户产生的临时数据
│   └── ...
```

## 快速开始

### 1. 环境准备

```bash
# 确保已安装 uv
# 安装依赖
cd Case-二手车价格预测
uv add scikit-learn pandas numpy matplotlib seaborn
```

### 2. 数据预处理

```bash
# 运行数据预处理
uv run python core/data_preprocessing.py
```

### 3. 探索性数据分析

```bash
# 运行EDA分析
uv run python core/eda_analysis.py
```

### 4. 数据建模

## 核心功能模块

### 📋 core/data_preprocessing.py
- **功能**: 完整的数据预处理流程
- **主要特性**:
  - 缺失值处理和异常值检测
  - 分类特征编码 (标签编码、One-Hot编码等)
  - 数值特征标准化和分箱
  - 时间特征提取

### 📈 core/eda_analysis.py
- **功能**: 深入的探索性数据分析
- **主要特性**:
  - 数据质量评估和统计描述
  - 特征相关性分析和可视化
  - 价格分布分析和异常值检测
  - 生成详细的EDA报告

### 🌲 core/rf_modeling.py
- **功能**: 专注于随机森林的优化建模
- **主要特性**:
  - 保守且稳健的RF参数配置
  - 多模型集成策略 (RF + ExtraTrees)
  - 交叉验证和性能评估
  - 稳健的特征工程策略

## 工具集合

### 🔧 utilities/data_analysis_tools.py
- 分类特征分析
- 数据一致性检查
- 数据质量验证

### 📏 utilities/model_validation_tools.py
- MAE深入分析
- 模型性能诊断
- 预测问题分析
- 综合模型评估

### ⚙️ utilities/feature_tools.py
- 特征相关性分析
- RF友好的特征创建
- 分类特征编码
- 特征重要性选择

## 参考实现

### reference/simple_rf_modeling.py
- 基础随机森林实现参考
- 适合初学者理解RF基本流程

### reference/diagnostic_analysis.py
- 模型性能诊断工具
- 训练验证与考试结果差异分析

## 主要成果

### 技术成果
- ✅ 完整的数据预处理流程
- ✅ 深入的EDA分析和可视化
- ✅ 专业的随机森林优化策略
- ✅ 完善的模型诊断和验证框架
- ✅ 模块化和可重用的代码架构

### 实验经验
- 📈 识别并解决了过拟合问题
- 📈 建立了稳健的模型验证策略
- 📈 探索了RF在价格预测中的能力边界
- 📈 累积了丰富的特征工程经验

## 性能指标

- **目标指标**: MAE (Mean Absolute Error)
- **目标分数**: MAE \< 500

## 使用说明

### 新手入门
1. 从 `code/eda_analysis.py` 开始，了解数据特性
2. 运行 `code/data_preprocessing.py` 预处理数据
3. 使用 `code/simple_rf_modeling.py` 学习基础流程
4. 进阶到 `model/rf_modeling.py` 进行优化

### 高级用户
1. 直接使用 `model/rf_modeling.py` 进行建模
2. 利用 `code/utilities/` 中的工具进行定制化分析
3. 参考 `code/diagnostic_analysis.py` 进行模型诊断

本项目使用的数据来自阿里云天池竞赛：
- **竞赛名称**: 车市先知：二手车价格预测学习赛
- **数据规模**: 训练集150,000条，测试集50,000条
- **特征数量**: 31个特征 (包含15个匿名特征)

## 技术栈

- **算法**: 随机森林 (RandomForest) + ExtraTrees
- **框架**: scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **项目管理**: uv

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 项目仓库: [GitHub Repository]
- 技术讨论: [相关论坛或群组]