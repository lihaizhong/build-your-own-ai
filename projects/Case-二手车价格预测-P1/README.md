# 二手车价格预测项目

> 专注于探索机器学习在二手车价格预测中的能力和优化策略

## 项目概述

本项目通过机器学习方法预测二手车的交易价格，主要探索**机器学习**在这一任务中的表现和优化策略。项目包含了完整的数据预处理、特征工程、模型训练和性能评估流程。

## 数据来源

- [【AI入门系列】车市先知：二手车价格预测学习赛](https://tianchi.aliyun.com/competition/entrance/231784/information)

## 考试排名

1. modeling_v24_simplified.py
    - 分数：**488.7255**
2. modeling_v23.py
    - 分数：**497.6048**
3. modeling_v26.py
    - 分数：**497.9590**
3. modeling_v24_fast.py
    - 分数：**501.8398**
4. modeling_v22.py
    - 分数：**502.1616**

### 其他考试成绩

- modeling_v27.py
    - 分数：783.9360
- modeling_v27_core.py
    - 分数：768.5758
- modeling_v27_fast.py
    - 分数：531.8382
- modeling_v25.py
    - 分数：1298.1723
- modeling_v24.py
    - 分数：5900.2078
- modeling_v19.py
    - 分数：516.7588
- modeling_v17.py
    - 分数：531.0229
- modeling_v16.py
    - 分数：539.8390

## 项目结构

```plaintext
Case-二手车价格预测/
├── README.md                                # 项目说明
├── code/                                    # 代码模块（主要用于数据分析、数据验证等）
│   ├── data_analysis_tools.py
│   └── model_validation_tools.py
├── feature                                 # 特征工程    
│   ├── feature_engineering.py
│   └── feature_selection.py
├── data/                                   # 数据文件
│   ├── used_car_train_20200313.csv
│   ├── used_car_testB_20200421.csv
│   └── used_car_sample_submit.csv
├── docs/                                   # 文档报告
│   ├── 项目方案总结报告.md
│   └── 诊断分析报告.md
├── prediction-result/                      # 预测结果
│   └── rf_result_*.csv
├── user_data/                              # 用户产生的临时数据
│   └── ...
```

## 性能指标

- **目标指标**: MAE (Mean Absolute Error)
- **目标分数**: MAE \< 500

## 使用说明

本项目使用的数据来自阿里云天池竞赛：
- **竞赛名称**: 车市先知：二手车价格预测学习赛
- **数据规模**: 训练集150,000条，测试集50,000条
- **特征数量**: 31个特征 (包含15个匿名特征)

## 技术栈

- **算法**: 随机森林 (RandomForest) 、ExtraTrees、XGBoost、LightGBM、CatBoost 等
- **框架**: scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **项目管理**: uv

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 项目仓库: [GitHub Repository](https://github.com/lihaizhong/build-your-own-ai)
- 技术讨论: [相关论坛](https://tianchi.aliyun.com/competition/entrance/231784)