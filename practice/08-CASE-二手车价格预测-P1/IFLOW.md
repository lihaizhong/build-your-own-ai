# 二手车价格预测项目 - IFLOW使用指南

## 项目概述

这是一个专注于二手车价格预测的机器学习项目，通过29个版本的迭代优化，从初始MAE 1220+优化到487.7112。项目展现了完整的机器学习开发流程，包括数据预处理、特征工程、模型训练、性能评估和持续优化。

### 核心特征
- **数据规模**: 训练集15万条，测试集5万条，31个特征（包含15个匿名特征）
- **目标指标**: MAE (Mean Absolute Error)
- **最佳成绩**: V28版本MAE = 487.7112
- **迭代深度**: 从V1到V29共29个版本优化
- **技术栈**: scikit-learn、pandas、numpy、matplotlib等

## 项目文件结构

### 核心目录
```
Case-二手车价格预测-P1/
├── README.md                    # 项目概述和性能记录
├── IFLOW.md                     # 本使用指南
├── DIALOGUE.md                  # 项目历史交互记录
├── data/                        # 原始数据文件目录
│   ├── used_car_train_20200313.csv      # 训练数据
│   ├── used_car_testB_20200421.csv      # 测试数据
│   └── used_car_sample_submit.csv       # 提交模板
├── code/                        # 预测模型脚本目录 ⭐
│   ├── modeling_v1.py ~ modeling_v29.py # 29个版本模型脚本
│   └── rf_modeling.py                    # 随机森林基准模型脚本
├── feature/                     # 分析工具和特征工程目录 ⭐
│   ├── modeling_v*_analysis.py          # 各版本特征分析
│   ├── plot_learning_curve.py           # 学习曲线可视化
│   └── plot_model_comparison.py         # 模型对比分析
├── model/                       # 训练好的模型文件目录（预留）
├── prediction_result/           # 预测结果目录
│   └── modeling_v*_result_*.csv         # 各版本预测结果
├── user_data/                   # 用户生成内容目录
│   ├── 预处理数据/
│   ├── 分析图表/
│   └── 模型可视化/
└── docs/                        # 项目文档目录
    ├── 探索性数据分析（EDA）报告.md
    ├── 数据集说明.md
    ├── 数据预处理建议.md
    ├── 项目方案总结报告.md
    ├── model_comparison.md
    └── model_detailed_analysis.md
```

### 关键文件说明

#### 数据文件
- **训练数据**: 15万条二手车交易记录，31个特征
- **测试数据**: 5万条记录，30个特征（无price目标变量）
- **预测结果**: 包含SaleID和price预测值的CSV文件

#### 代码文件
- **modeling_v1.py ~ modeling_v29.py**: 完整的模型迭代记录（位于code/目录）
- **modeling_v28.py**: 当前最佳模型（MAE=487.7112）
- **modeling_v29.py**: 最新版本（目标突破475分）

## 技术栈与依赖

### 核心依赖
```python
# 数据处理
pandas>=1.3.0
numpy>=1.21.0

# 机器学习
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.6.0
catboost>=1.0.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0

# 项目管理
uv (Python包管理器)
```

### 主要算法
- **随机森林** (RandomForest): 基准算法和特征分析
- **XGBoost**: 梯度提升决策树
- **LightGBM**: 微软轻量级梯度提升机
- **CatBoost**: Yandex梯度提升库
- **ExtraTrees**: 极随机树
- **Stacking集成**: 多模型元学习集成

## 核心使用方法

### 1. 数据加载与预处理
```python
# 标准数据加载
def load_data(file_path):
    """加载二手车数据"""
    return pd.read_csv(file_path, sep=' ', na_values=['-'])

# 预处理管道（基于项目最佳实践）
def enhanced_preprocessing():
    """增强数据预处理"""
    # 缺失值处理、特征编码、标准化等
    pass
```

### 2. 模型训练
```python
# 运行最新模型
python code/modeling_v29.py

# 运行特定版本
python code/modeling_v28.py

# 运行快速版本（用于快速验证）
python code/modeling_v28_fast.py
```

### 3. 性能验证
```python
# 查看预测结果
ls prediction_result/

# 检查模型性能
python -c "
import pandas as pd
result = pd.read_csv('prediction_result/modeling_v28_20251027_235409.csv')
print(f'预测样本数: {len(result)}')
print(f'价格范围: {result[\"price\"].min():.2f} - {result[\"price\"].max():.2f}')
"
```

### 4. 特征工程
```python
# 运行特征分析
python feature/modeling_v28_analysis.py

# 查看特征重要性
python -c "
import pandas as pd
import matplotlib.pyplot as plt
features = pd.read_csv('user_data/modeling_v28/特征重要性.csv')
plt.barh(features['feature'], features['importance'])
plt.show()
"
```

## 模型版本演进历史

### 最佳成绩版本
1. **modeling_v28.py** - MAE: 487.7112 (当前最佳)
2. **modeling_v24_simplified.py** - MAE: 488.7255
3. **modeling_v23.py** - MAE: 497.6048
4. **modeling_v26.py** - MAE: 497.9590
5. **modeling_v24_fast.py** - MAE: 501.8398

### 版本特点
- **V1-V9**: 基础算法探索和参数调优
- **V10-V15**: 特征工程增强和集成策略
- **V16-V20**: 抗过拟合优化和稳健性改进
- **V21-V25**: 高级集成技术和分层建模
- **V26-V29**: 深度优化和智能校准

### 关键突破
- **分层建模**: 按价格区间分别建模
- **Stacking集成**: 使用元学习器优化集成
- **深度特征交互**: 三阶交互和多项式特征
- **动态权重**: 基于验证集性能调整权重
- **智能校准**: 优化后处理和异常值处理

## 性能分析

### 竞赛表现
- **历史成绩**: 从V1的1220+ MAE优化到V28的487.7112
- **优化幅度**: 总体提升约60%
- **稳定性**: V24-V28版本在500分以内稳定表现
- **目标**: V29版本冲击475分以内

### 诊断分析
```python
# 查看学习曲线
python feature/plot_learning_curve.py

# 模型对比分析
python feature/plot_model_comparison.py
```

## 核心优化策略

### 1. 数据预处理
- **缺失值处理**: 智能填充和缺失指示变量
- **特征编码**: 标签编码、频率编码、One-Hot编码
- **时间特征**: 从日期提取年份、月份、车龄等
- **异常值检测**: 基于统计和业务逻辑的异常值处理

### 2. 特征工程
- **交互特征**: 关键特征组合（如功率×车龄）
- **统计特征**: 基于品牌、地区的统计信息
- **多项式特征**: 二阶和三阶多项式特征
- **特征选择**: 基于重要性和相关性分析

### 3. 模型集成
- **多模型融合**: RF、XGBoost、LightGBM、CatBoost
- **Stacking集成**: 元学习器优化集成效果
- **动态权重**: 基于交叉验证性能调整权重
- **分层建模**: 按价格区间建立专门模型

### 4. 后处理优化
- **校准调整**: 分位数校准和分布调整
- **异常值处理**: 极端值的智能处理
- **集成优化**: 多级集成和权重平衡

## 开发最佳实践

### 代码规范
```python
# 标准模板
def get_project_path(*paths):
    """获取项目路径的统一方法"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    return os.path.join(project_dir, *paths)

# 结果保存
def save_results(predictions, model_name):
    """保存预测结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_result_{timestamp}.csv"
    predictions.to_csv(get_project_path('prediction_result', filename), index=False)
```

### 性能监控
```python
# 交叉验证
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
mae_score = -cv_scores.mean()

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

## 故障排除

### 常见问题
1. **内存不足**: 减少数据样本或使用分块处理
2. **模型过拟合**: 调整正则化参数，减少特征数量
3. **性能下降**: 检查数据预处理和特征工程步骤
4. **验证不一致**: 确保本地验证与真实评估环境一致

### 调试技巧
```python
# 添加调试日志
import logging
logging.basicConfig(level=logging.INFO)

# 性能分析
import cProfile
cProfile.run('main_function()', 'profile_output')

# 内存分析
import tracemalloc
tracemalloc.start()
# 执行代码
snapshot = tracemalloc.take_snapshot()
```

## 下一步发展方向

### 短期优化
- **V30版本**: 目标突破475分
- **模型解释性**: 增强SHAP值分析
- **自动化管道**: 建立端到端自动化流程

### 长期规划
- **在线学习**: 增量学习能力
- **多模态数据**: 整合图片和文本信息
- **实时预测**: 构建在线预测服务
- **商业应用**: 实际业务场景应用

## 关键学习点

### 技术收获
1. **完整的ML项目流程**: 从数据到部署的端到端经验
2. **模型迭代优化**: 29个版本的技术演进历程
3. **性能诊断方法**: 过拟合检测和验证策略
4. **特征工程艺术**: 多层次特征设计理念

### 业务价值
1. **定价策略优化**: 为二手车定价提供数据支持
2. **市场洞察**: 基于数据分析的市场趋势识别
3. **风险评估**: 车辆价值波动的风险控制
4. **用户体验**: 为买家和卖家提供准确的价格参考

## 联系与贡献

### 项目资源
- **项目仓库**: https://github.com/lihaizhong/build-your-own-ai
- **竞赛链接**: 阿里云天池二手车价格预测竞赛
- **技术文档**: docs目录下详细技术文档

### 贡献指南
1. Fork项目并创建功能分支
2. 添加新的模型版本或优化策略
3. 更新相关文档和性能记录
4. 提交Pull Request进行代码审查

---

*本文档随项目发展持续更新，最后更新时间: 2025年12月15日*
*项目状态: 持续优化中，当前最佳成绩: MAE 487.7112 (V28)*
*目录结构: 已标准化为与CASE-资金流入流出预测-P1一致的结构*