# 资金流入流出预测项目

> 专注于探索时间序列模型在金融数据预测中的能力和优化策略

## 项目概述

本项目通过时间序列分析方法预测基金的申购和赎回资金流动，主要探索**时间序列预测模型**在这一任务中的表现和优化策略。项目包含了完整的数据预处理、特征工程、模型训练和性能评估流程，使用了Prophet、ARIMA等多种时间序列预测方法。

## 数据来源

- [资金流入流出预测-挑战Baseline](https://tianchi.aliyun.com/competition/entrance/231573)

## 考试排名

1. prophet_v6_prediction.py
    - 分数：**123.9908**

## 项目结构

```plaintext
practice/11-CASE-资金流入流出预测-P1/
├── README.md                                # 项目说明
├── 资金流入流出预测.ipynb                   # Jupyter笔记本（主分析）
├── code/                                    # 代码模块
│   └── prophet_v6_prediction.py            # 最佳模型预测代码
├── feature/                                 # 特征工程与数据分析
│   ├── analyze_weekend_effect.py           # 周末效应分析
│   ├── data_analysis.py                    # 数据分析工具
│   ├── data_loader.py                      # 数据加载工具
│   ├── prophet_model_comparison.py        # Prophet模型对比
│   ├── test_holiday_impact.py              # 节假日影响测试
│   ├── time_series_analysis.py             # 时间序列分析
│   └── visualization.py                    # 可视化工具
├── data/                                   # 数据文件
│   ├── comp_predict_table.csv              # 竞赛预测表
│   ├── mfd_bank_shibor.csv                 # 银行间拆借利率数据
│   ├── mfd_day_share_interest.csv          # 股票利率数据
│   └── user_profile_table.csv              # 用户画像数据
├── model/                                  # 模型文件
│   └── *.pkl                              # 训练好的模型
├── prediction_result/                      # 预测结果
│   └── *.csv                              # 模型预测输出
└── docs/                                   # 文档报告
    └── ...
```

## 性能指标

- **目标指标**: MAE (Mean Absolute Error)
- **目标分数**: MAE \< 150
- **最佳成绩**: 123.9908 (prophet_v6_prediction.py)

## 使用说明

本项目使用的数据来自阿里云天池竞赛：
- **竞赛名称**: 资金流入流出预测-挑战Baseline
- **任务类型**: 时间序列预测
- **预测目标**: 基金申购和赎回金额
- **数据特征**: 包含用户画像、市场利率、历史交易等多维度数据

## 技术栈

- **算法**: Prophet、ARIMA、周期因子分析、混合模型
- **框架**: statsmodels, fbprophet
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **项目管理**: uv

## 模型说明

本项目尝试了多种时间序列预测方法：

1. **Prophet模型**: Facebook开发的时间序列预测模型，能够处理趋势、季节性和节假日效应
2. **ARIMA模型**: 自回归积分滑动平均模型，适用于线性时间序列预测
3. **周期因子分析**: 基于周期性特征的预测方法
4. **混合模型**: 结合多种方法的集成预测

经过多次迭代和调优，prophet_v6版本取得了最佳成绩123.9908。

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 项目仓库: [GitHub Repository](https://github.com/lihaizhong/build-your-own-ai)
- 技术讨论: [天池竞赛页面](https://tianchi.aliyun.com/competition/entrance/231573)