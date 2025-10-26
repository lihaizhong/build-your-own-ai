# 二手车价格预测项目

> 专注于探索机器学习在二手车价格预测中的能力和优化策略

## 项目概述

本项目通过机器学习方法预测二手车的交易价格，主要探索**机器学习**在这一任务中的表现和优化策略。项目包含了完整的数据预处理、特征工程、模型训练和性能评估流程。

该数据来自阿里云天池竞赛平台，总数据量超过40万条，包含31列变量信息，其中15列为匿名变量。

## 数据来源

- [【AI入门系列】车市先知：二手车价格预测学习赛](https://tianchi.aliyun.com/competition/entrance/231784/information)

## 数据集说明

### 数据规模
- **训练集**: 15万条记录
- **测试集A**: 5万条记录  
- **测试集B**: 5万条记录
- **总数据量**: 超过40万条记录

## 性能指标

- **目标指标**: MAE (Mean Absolute Error)
- **目标分数**: MAE < 500

### 数据字段
| 字段名 | 说明 | 类型 |
|--------|------|------|
| SaleID | 销售样本ID | 数值 |
| name | 汽车编码 | 数值 |
| regDate | 汽车注册时间 | 日期 |
| model | 车型编码 | 数值 |
| brand | 品牌 | 类别 |
| bodyType | 车身类型 | 类别 |
| fuelType | 燃油类型 | 类别 |
| gearbox | 变速箱 | 类别 |
| power | 汽车功率 | 数值 |
| kilometer | 汽车行驶里程 | 数值 |
| notRepairedDamage | 汽车有尚未修复的损坏 | 类别 |
| v_0, v_1, ..., v_14 | 匿名特征 | 数值 |

### 数据特点
- 包含缺失值：bodyType、fuelType、gearbox等字段存在缺失
- 存在数据倾斜：seller、offerType等特征严重倾斜
- 包含异常值：power、price等字段存在极端值
- 匿名特征：v_0到v_14为匿名数值特征

## 项目结构

```plaintext
Case-二手车价格预测/
├── README.md                                # 项目说明
├── code/                                    # 代码模块
├── data/                                   # 数据文件目录
│   ├── used_car_train_20200313.csv    # 训练数据
│   ├── used_car_testB_20200421.csv    # 测试数据B
│   └── used_car_sample_submit.csv     # 提交样例
├── docs/                                   # 文档报告
├── prediction_result/              # 预测结果目录
├── user_data/                      # 用户数据和报告
├── Datawhale 零基础入门数据挖掘-*.ipynb  # Jupyter教程文件
│   ├── Task1 赛题理解.ipynb           # 赛题理解和数据概览
│   ├── Task2 数据分析.ipynb           # 探索性数据分析
│   ├── Task3 特征工程.ipynb           # 特征工程处理
│   ├── Task4 建模调参.ipynb           # 模型训练和调优
│   └── Task5 模型融合.ipynb           # 模型融合策略
├── Datawhale 零基础入门数据挖掘-FAQ.md    # 常见问题解答
├── Datawhale 零基础入门数据挖掘-Baseline.ipynb  # 基准模型
├── Datawhale 零基础入门数据挖掘-PyTorch基础代码.ipynb  # PyTorch基础
└── 启动数据分析报告.py             # 数据分析报告服务器
```

## 快速开始

### 环境要求
- Python 3.11+
- 主要依赖包：pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
- 项目管理：uv (推荐用于依赖管理)

### 快速环境设置
```bash
# 使用 uv 管理环境
cd build-your-own-ai
uv sync
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

### 运行步骤

1. **环境准备**
```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

2. **数据探索**
```bash
# 启动Jupyter Notebook
jupyter notebook

# 按顺序学习教程文件：
# 1. Task1 赛题理解.ipynb
# 2. Task2 数据分析.ipynb
# 3. Task3 特征工程.ipynb
# 4. Task4 建模调参.ipynb
# 5. Task5 模型融合.ipynb
```

3. **查看数据分析报告**
```bash
# 启动数据分析报告服务器
python 启动数据分析报告.py
# 浏览器会自动打开 http://localhost:8000
```

## 项目特点

### 完整的学习路径
- **Task1**: 赛题理解和数据概览
- **Task2**: 探索性数据分析(EDA)
- **Task3**: 特征工程和数据处理
- **Task4**: 模型训练和参数调优
- **Task5**: 模型融合和集成学习

### 技术亮点
- **数据预处理**: 缺失值处理、异常值检测、数据倾斜处理
- **特征工程**: 特征编码、特征选择、特征变换
- **模型选择**: 线性模型、树模型、集成学习对比
- **模型优化**: 交叉验证、超参数调优、模型融合

### 实用工具
- **FAQ文档**: 涵盖常见技术问题和解决方案
- **基准模型**: 提供完整的baseline代码
- **数据可视化**: 自动生成数据分析报告
- **PyTorch基础**: 深度学习入门代码

## 技术栈

- **算法**: 随机森林 (RandomForest)、ExtraTrees、XGBoost、LightGBM、CatBoost 等
- **框架**: scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **项目管理**: uv

## 数据分析建议

### 缺失值处理
- bodyType、fuelType、gearbox存在缺失，建议进行填补
- notRepairedDamage字段中的"-"字符需要清洗为NaN或0/1

### 特征工程
- power、kilometer与price有非线性关系，建议尝试log变换
- v0~v14匿名特征需要分析相关性，避免多重共线性
- 对严重倾斜的特征(seller、offerType)考虑直接删除

### 模型选择
- 树模型(RandomForest、XGBoost、LightGBM、CatBoost)适合处理非线性关系
- 决策树模型不推荐对离散特征进行one-hot编码
- 可以尝试模型融合提升预测效果

## 常见问题

**Q: 如何处理特征的严重倾斜？**
A: 可以通过pandas_profiling查看分布图，对于严重倾斜的特征如seller、offerType可以考虑直接删除。

**Q: XGBoost处理分类特征时需要one-hot编码吗？**
A: 不推荐。决策树模型对离散特征进行one-hot编码会产生样本切分不平衡问题和影响决策树学习。

**Q: 目标变量为什么要符合高斯分布？**
A: 因为很多算法(如线性回归)的前提假设是数据符合正态分布，这样可以获得更好的模型效果。

**Q: 如何处理测试集中的异常值？**
A: 假设测试集与训练集同分布，如果异常值比例较小可以忽略，如果比例较大可以考虑使用生成式方法。

## 学习目标

通过本项目，您将学到：
- 完整的机器学习项目流程
- 数据探索和可视化技巧
- 特征工程的常用方法
- 模型选择和调优策略
- 模型融合和集成学习
- 实际业务问题的建模思路

## 扩展学习

完成本项目后，建议继续学习：

### 进阶版本
- **Case-二手车价格预测-P1**: 优化版本，包含23个版本的模型迭代
  - 最新成绩：MAE达到488.7255（远超500分目标）
  - 包含高级技术：分层建模、Stacking集成、智能校准、贝叶斯优化
  - 运行方式：`cd projects/Case-二手车价格预测-P1 && python model/modeling_v23.py`

### 相关领域
- **时间序列预测**: 学习时间序列分析方法
  - 参考：`projects/CASE-资金流入流出预测-P1`
- **深度学习**: 探索神经网络在表格数据中的应用
  - 参考：`courseware/12-神经网络基础与Tensorflow实战`
- **Agent开发**: 构建智能AI助手
  - 参考：`courseware/5-Agent进阶实战与插件开发`

### 性能优化成果
- **原始版本**: 本项目，提供完整的机器学习基础入门
- **优化版本**: P1版本，通过23次迭代优化，MAE从1000+优化到488分
- **技术突破**: 实现了分层建模、动态权重调整、智能校准等高级技术

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库: https://github.com/lihaizhong/build-your-own-ai.git
- 问题反馈: 通过GitHub Issues提交
- 技术讨论: [阿里云天池竞赛](https://tianchi.aliyun.com/competition/entrance/231784)

---

*本项目基于Datawhale零基础入门数据挖掘教程和阿里云天池竞赛，适用于机器学习初学者和竞赛入门。*
*最后更新: 2025年10月26日*