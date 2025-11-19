# CASE-资金流入流出预测-P1 项目指南

## 项目概述

这是一个金融科技领域的机器学习项目，专注于预测用户的资金流入流出行为。项目基于真实的用户行为数据，构建预测模型来预测用户在特定时间点的资金流动情况。该项目是资金流入流出预测的实践案例，采用完整的数据科学工作流程。

## 项目结构

```
CASE-资金流入流出预测-P1/
├── 资金流入流出预测.ipynb          # 主要分析笔记本（Jupyter Notebook）
├── README.md                       # 项目说明文档（当前为空）
├── IFLOW.md                        # 本文件，项目交互指南
├── code/                           # 代码脚本目录
│   ├── display_user_balance.py     # 数据展示脚本
│   ├── read_user_balance_simple.py # 简化的数据读取脚本
│   └── read_user_balance.py        # 高级数据读取脚本（需要pandas）
├── data/                           # 数据文件目录
│   ├── user_profile_table.csv      # 用户画像数据表
│   ├── user_balance_table.csv      # 用户余额和交易记录数据表
│   ├── mfd_day_share_interest.csv  # 货币基金日收益率数据
│   ├── mfd_bank_shibor.csv         # 银行间拆借利率数据
│   └── comp_predict_table.csv      # 预测结果表（推测）
├── docs/                           # 文档目录
├── feature/                        # 特征工程目录
├── model/                          # 模型存储目录
├── prediction_result/              # 预测结果输出目录
└── user_data/                      # 用户数据目录
```

## 核心技术栈

### 数据处理
- **Python 3.11.13** - 主要编程语言
- **Pandas** - 数据分析和处理
- **Jupyter Notebook** - 交互式分析环境

### 机器学习
- **scikit-learn** - 传统机器学习算法
- **时间序列分析** - 针对金融时序数据的专门方法
- **特征工程** - 从原始数据中提取预测特征

### 数据源类型
- **用户画像数据**: 用户基本信息（ID、性别、城市、星座）
- **交易数据**: 余额、购买、赎回、消费、转账等行为数据
- **市场数据**: 货币基金收益率、银行间拆借利率等宏观数据

## 数据文件说明

### 核心数据表

#### 1. user_profile_table.csv - 用户画像表
- **user_id**: 用户唯一标识
- **sex**: 性别（数值编码）
- **city**: 城市代码
- **constellation**: 星座信息

#### 2. user_balance_table.csv - 用户余额交易表
- **user_id**: 用户ID
- **report_date**: 报告日期（YYYYMMDD格式）
- **tBalance**: 总余额
- **yBalance**: 昨日余额
- **total_purchase_amt**: 总购买金额
- **direct_purchase_amt**: 直接购买金额
- **purchase_bal_amt**: 余额购买金额
- **purchase_bank_amt**: 银行购买金额
- **total_redeem_amt**: 总赎回金额
- **consume_amt**: 消费金额
- **transfer_amt**: 转账金额
- **tftobal_amt**: 转账至余额金额
- **tftocard_amt**: 转账至卡金额
- **share_amt**: 份额金额
- **category1-4**: 分类字段

#### 3. mfd_day_share_interest.csv - 货币基金日收益率
- **mfd_date**: 日期
- **mfd_daily_yield**: 日收益率
- **mfd_7daily_yield**: 7日年化收益率

#### 4. mfd_bank_shibor.csv - 银行间拆借利率
- **mfd_date**: 日期
- **Interest_O_N**: 隔夜利率
- **Interest_1_W**: 1周利率
- **Interest_2_W**: 2周利率
- **Interest_1_M**: 1月利率
- **Interest_3_M**: 3月利率
- **Interest_6_M**: 6月利率
- **Interest_9_M**: 9月利率
- **Interest_1_Y**: 1年利率

## 工作流程

### Step 1: 数据加载
使用统一的路径管理函数加载所有数据源：
```python
def get_project_path(*paths):
    """获取项目路径的统一方法"""
    try:
        return os.path.join(os.path.dirname(__file__), *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

# 加载数据
Train_UPT = pd.read_csv(get_project_path('data', 'user_profile_table.csv'))
Train_UBT = pd.read_csv(get_project_path('data', 'user_balance_table.csv'))
Train_MDSI = pd.read_csv(get_project_path('data', 'mfd_day_share_interest.csv'))
Train_MBS = pd.read_csv(get_project_path('data', 'mfd_bank_shibor.csv'))
```

### Step 2: 数据探索（EDA）
- 分析用户画像数据分布
- 检查余额和交易数据的统计特征
- 探索时间序列模式
- 识别缺失值和异常值

### Step 3: 数据预处理
- 数据清洗和缺失值处理
- 特征工程（时间特征、聚合特征等）
- 数据标准化和编码

### Step 4: 模型训练
- 模型选择（待完善）
- 超参数调优（待完善）
- 交叉验证（待完善）

### Step 5: 模型预测
- 预测结果生成（待完善）
- 结果评估（待完善）

## 使用指南

### 环境要求
- Python 3.11+
- pandas库（数据处理）
- Jupyter Notebook环境

### 快速开始

1. **查看数据概览**:
   ```bash
   cd /path/to/CASE-资金流入流出预测-P1
   python3 code/display_user_balance.py
   ```

2. **数据分析**:
   ```bash
   jupyter notebook 资金流入流出预测.ipynb
   ```

3. **数据探索**:
   ```bash
   python3 code/read_user_balance_simple.py
   ```

### 常用操作

#### 查看用户余额数据前5行
```bash
python3 code/display_user_balance.py
```

#### 查看所有数据文件结构
```bash
# 查看数据表头
head -1 data/*.csv

# 查看数据文件大小
ls -lh data/
```

## 项目状态

### 已完成部分
- ✅ 数据文件结构定义
- ✅ 数据加载框架
- ✅ 基础数据探索代码
- ✅ 数据展示工具

### 待完善部分
- ⏳ 完整的数据探索分析（EDA）
- ⏳ 特征工程实现
- ⏳ 模型训练和验证
- ⏳ 预测结果输出
- ⏳ 模型性能评估

### 项目特点
- **金融专业性**: 专注于资金流动预测，具有明确的金融业务背景
- **多数据源融合**: 整合用户行为、市场利率、收益数据等多个维度
- **时序预测**: 涉及时间序列分析，适合金融时序数据建模
- **完整工作流**: 涵盖数据科学项目的完整流程

## 开发规范

### 代码规范
- 使用类型注解（Python 3.11+）
- 遵循PEP 8代码风格
- 函数和模块添加文档字符串
- 统一的路径管理方法

### 文件组织
- **code/**: 存放可执行的Python脚本
- **data/**: 存放原始数据文件
- **model/**: 存放训练好的模型文件
- **prediction_result/**: 存放预测结果
- **feature/**: 存放特征工程相关代码
- **docs/**: 存放项目文档

### 最佳实践
1. 使用相对路径和统一路径管理
2. 数据文件不要提交到版本控制
3. 模型和结果文件分类存放
4. 代码模块化和可复用

## 故障排除

### 常见问题
1. **pandas未安装**: 使用简化版本脚本或安装pandas
2. **文件路径错误**: 使用`get_project_path()`函数统一管理路径
3. **数据文件过大**: 使用分块读取或采样分析

### 调试技巧
```python
# 检查数据文件
import os
print(os.path.exists('data/user_balance_table.csv'))

# 查看数据基本信息
df = pd.read_csv('data/user_balance_table.csv', nrows=5)
print(df.info())
```

## 后续开发建议

### 短期目标
1. 完善数据探索分析（EDA）
2. 实现特征工程流水线
3. 尝试基础机器学习模型

### 长期目标
1. 深度学习模型应用
2. 模型可解释性分析
3. 实时预测系统构建
4. 模型性能监控

---

*本指南将随项目发展持续更新，建议定期查看最新版本。*
*最后更新: 2025年11月19日*