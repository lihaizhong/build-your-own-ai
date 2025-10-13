# 二手车价格预测项目

> 专注于探索随机森林算法在二手车价格预测中的能力和优化策略

## 项目概述

本项目通过机器学习方法预测二手车的交易价格，主要探索**随机森林算法**在这一任务中的表现和优化策略。项目包含了完整的数据预处理、特征工程、模型训练和性能评估流程。

## 项目结构

```
Case-二手车价格预测/
├── README.md                 # 项目说明
├── core/                     # 核心功能模块
│   ├── data_preprocessing.py # 数据预处理模块
│   ├── eda_analysis.py       # 探索性数据分析
│   └── rf_modeling.py        # 随机森林建模
├── reference/                # 参考实现
│   ├── simple_rf_modeling.py # 基础RF实现
│   ├── random_forest_modeling.py # RF建模参考
│   └── diagnostic_analysis.py # 诊断分析工具
├── utilities/                # 工具集合
│   ├── data_analysis_tools.py # 数据分析工具
│   ├── model_validation_tools.py # 模型验证工具
│   └── feature_tools.py      # 特征工程工具
├── data/                     # 数据文件
│   ├── used_car_train_20200313.csv # 训练数据
│   ├── used_car_testB_20200421.csv # 测试数据
│   └── used_car_sample_submit.csv  # 提交样例
├── docs/                     # 文档报告
│   ├── 项目方案总结报告.md
│   ├── 诊断分析报告.md
│   └── ...
├── results/                  # 预测结果
│   └── rf_result_*.csv
└── archive/                  # 存档文件
    └── ...
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

### 4. 随机森林建模

```bash
# 运行主要建模脚本
uv run python core/rf_modeling.py
```

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
- **基线性能**: MAE ≈ 1200+
- **当前最佳**: MAE ≈ 600-800 (交叉验证)
- **稳健性**: 重点关注模型的泛化能力

## 使用说明

### 新手入门
1. 从 `core/eda_analysis.py` 开始，了解数据特性
2. 运行 `core/data_preprocessing.py` 预处理数据
3. 使用 `reference/simple_rf_modeling.py` 学习基础流程
4. 进阶到 `core/rf_modeling.py` 进行优化

### 高级用户
1. 直接使用 `core/rf_modeling.py` 进行建模
2. 利用 `utilities/` 中的工具进行定制化分析
3. 参考 `reference/diagnostic_analysis.py` 进行模型诊断

## 重要经验教训

### ✅ 成功经验
- **数据预处理标准化**: 建立稳定可重用的预处理流程
- **RF参数保守化**: 防止过拟合的保守参数设置
- **集成策略简化**: 简单平均优于复杂权重策略

### ⚠️ 重要教训
- **过拟合风险**: 激进优化策略容易导致严重过拟合
- **验证一致性**: 本地验证必须与真实评估环境一致
- **特征工程约束**: 过度复杂的特征工程可能引入噪声

## 最佳实践建议

### 随机森林参数设置
```python
RandomForestRegressor(
    n_estimators=150-250,    # 适中的树数量
    max_depth=15-25,         # 限制深度防止过拟合
    min_samples_split=10-20, # 增加分割最小样本数
    min_samples_leaf=5-10,   # 增加叶节点最小样本数
    max_features=0.4-0.6,    # 限制特征子集大小
    random_state=42
)
```

### 特征工程策略
- 重点关注基础特征交互
- 避免过度复杂的统计特征
- 使用RF友好的分箱特征
- 控制特征总数量避免维度灾难

### 验证策略
- 使用分层交叉验证
- 考虑数据分布差异
- 确保验证结果与考试结果的一致性

## 数据来源

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

---

*最后更新: 2025-10-01*
*项目状态: 持续优化中*