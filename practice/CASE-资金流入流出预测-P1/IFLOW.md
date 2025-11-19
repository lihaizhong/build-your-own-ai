# CASE-资金流入流出预测-P1 项目指南

## 项目概述

这是一个金融科技领域的机器学习项目，专注于预测用户的资金流入流出行为。项目基于真实的用户行为数据，构建预测模型来预测用户在特定时间点的资金流动情况。该项目是资金流入流出预测的实践案例，采用完整的数据科学工作流程。

## 项目结构

```
build-your-own-ai/                 # 项目根目录
├── .venv/                         # uv虚拟环境目录
├── CASE-资金流入流出预测-P1/      # 当前子项目
│   ├── 资金流入流出预测.ipynb     # 主要分析笔记本（Jupyter Notebook）
│   ├── README.md                  # 项目说明文档（当前为空）
│   ├── IFLOW.md                   # 本文件，项目交互指南
│   ├── code/                      # 代码脚本目录
│   │   ├── display_user_balance.py     # 数据展示脚本
│   │   ├── read_user_balance_simple.py # 简化的数据读取脚本
│   │   ├── read_user_balance.py        # 高级数据读取脚本（需要pandas）
│   │   ├── analyze_daily_flow.py       # 每日资金流动分析脚本
│   │   ├── ascii_chart.py              # ASCII艺术图表生成脚本
│   │   ├── html_chart.py               # 交互式HTML图表生成脚本
│   │   └── plot_daily_flow.py          # 完整matplotlib图表脚本
│   ├── data/                      # 数据文件目录
│   │   ├── user_profile_table.csv      # 用户画像数据表
│   │   ├── user_balance_table.csv      # 用户余额和交易记录数据表
│   │   ├── mfd_day_share_interest.csv  # 货币基金日收益率数据
│   │   ├── mfd_bank_shibor.csv         # 银行间拆借利率数据
│   │   └── comp_predict_table.csv      # 预测结果表（推测）
│   ├── docs/                      # 文档目录
│   ├── feature/                   # 特征工程目录
│   ├── model/                     # 模型存储目录
│   ├── prediction_result/         # 考试预测结果目录
│   │   └── tc_comp_predict_table.csv  # 考试提交的预测文件（待生成）
│   ├── user_data/                 # 用户数据处理目录
│   │   ├── daily_summary.csv           # 每日数据汇总统计（现有数据整理）
│   │   ├── daily_flow_trend.png        # 申购赎回趋势图表
│   │   └── chart_data.json            # 图表数据文件
│   ├── user_data/                 # 用户数据处理目录
│   │   ├── daily_flow_trend.png        # 申购赎回趋势图表
│   │   └── chart_data.json            # 图表数据文件
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

#### 5. comp_predict_table.csv - 考试预测格式参考
- **格式说明**: 预测文件的格式参考，包含日期、申购金额、赎回金额
- **日期格式**: YYYYMMDD（无连字符）
- **用途**: 用于了解考试提交的预测文件格式

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
- **Python版本**: Python 3.11+
- **环境管理**: 使用 uv 管理Python环境和依赖
- **虚拟环境位置**: `.venv` 目录位于 `build-your-own-ai` 项目根目录
- **核心依赖**: pandas, matplotlib, scikit-learn 等数据分析库
- **开发工具**: Jupyter Notebook环境

### 环境管理（重要）

本项目使用 **uv** 作为Python包管理器，具有以下优势：
- 🚀 **更快的依赖安装**: 比pip快10-100倍
- 🔒 **更好的依赖锁定**: 自动生成uv.lock文件
- 🎯 **智能缓存**: 避免重复下载相同的包
- 📦 **虚拟环境管理**: 无需手动激活，直接使用uv run

#### uv常用命令
```bash
# 安装新依赖
uv add 包名

# 安装开发依赖
uv add --dev 包名

# 同步所有依赖（从uv.lock文件）
uv sync

# 运行Python脚本（自动使用uv环境）
uv run python script.py

# 创建新的uv项目
uv init
```

#### 激活虚拟环境
```bash
# 进入项目目录
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai/practice/CASE-资金流入流出预测-P1

# 激活uv虚拟环境（从当前项目目录向上两级到build-your-own-ai根目录）
source ../../.venv/bin/activate

# 或者使用绝对路径
source /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai/.venv/bin/activate

# 验证环境
python --version
which python
```

#### 依赖情况
```bash
# 验证已安装的依赖（uv环境已预装这些包）
uv run python -c "import pandas as pd; import matplotlib.pyplot as plt; print('pandas和matplotlib已可用')"

# 如需安装额外依赖
uv add scikit-learn jupyter

# 或者使用requirements.txt（如果存在）
uv sync
```

### 快速开始

#### 步骤1: 激活虚拟环境
```bash
# 进入项目目录
cd /path/to/CASE-资金流入流出预测-P1

# 激活uv虚拟环境（.venv在build-your-own-ai目录下）
source ../.venv/bin/activate

# 验证环境已激活
python --version  # 应该显示Python 3.11+
which python     # 应该显示.venv中的python路径
```

#### 步骤2: 查看数据概览
```bash
# 运行基础数据展示脚本
python code/display_user_balance.py
```

#### 步骤3: 数据分析
```bash
# 启动Jupyter Notebook
jupyter notebook 资金流入流出预测.ipynb
```

#### 步骤4: 数据探索
```bash
# 运行数据探索脚本
python code/read_user_balance_simple.py
```

#### 步骤5: 数据分析和图表生成
```bash
# 基础数据分析（在Python中直接执行）
uv run python -c "
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据并按日期汇总
df = pd.read_csv('data/user_balance_table.csv')
daily_summary = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
daily_summary['net_flow'] = daily_summary['total_purchase_amt'] - daily_summary['total_redeem_amt']

print(f'数据期间: {daily_summary[\"report_date\"].min()} 至 {daily_summary[\"report_date\"].max()}')
print(f'总申购额: {daily_summary[\"total_purchase_amt\"].sum()/1e8:.2f} 亿元')
print(f'总赎回额: {daily_summary[\"total_redeem_amt\"].sum()/1e8:.2f} 亿元')

# 保存数据汇总结果
daily_summary.to_csv('user_data/daily_summary.csv', index=False)
print('数据汇总已保存到 user_data/daily_summary.csv')
"

# 生成可视化图表
uv run python -c "

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 读取汇总数据
daily_summary = pd.read_csv('prediction_result/daily_summary.csv')
daily_summary['date'] = pd.to_datetime(daily_summary['report_date'], format='%Y%m%d')

# 创建图表
plt.figure(figsize=(12, 8))

# 申购赎回趋势
plt.subplot(2, 1, 1)
plt.plot(daily_summary['date'], daily_summary['total_purchase_amt']/1e6, label='Purchase Amount', linewidth=1.5)
plt.plot(daily_summary['date'], daily_summary['total_redeem_amt']/1e6, label='Redeem Amount', linewidth=1.5)
plt.title('Daily Purchase and Redeem Trends')
plt.ylabel('Amount (Million Yuan)')
plt.legend()
plt.grid(True, alpha=0.3)

# 净流入趋势
plt.subplot(2, 1, 2)
colors = ['green' if x >= 0 else 'red' for x in daily_summary['net_flow']]
plt.bar(daily_summary['date'], daily_summary['net_flow']/1e6, color=colors, alpha=0.7, width=0.8)
plt.title('Daily Net Flow')
plt.ylabel('Net Flow (Million Yuan)')
plt.xlabel('Date')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('user_data/daily_flow_trend.png', dpi=300, bbox_inches='tight')
print('图表已保存到 user_data/daily_flow_trend.png')
"
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
- ✅ 完整数据分析（284万条记录，427天数据）
- ✅ 每日申购赎回趋势分析
- ✅ 可视化图表生成（PNG格式）
- ✅ 数据汇总输出（CSV格式）

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
- **实际数据分析**: 基于284万条用户交易记录（2013年7月-2014年8月）
- **可视化结果**: 总申购额925.91亿元，总赎回额727.18亿元，净流入198.73亿元

## 开发规范

### 代码规范
- 使用类型注解（Python 3.11+）
- 遵循PEP 8代码风格
- 函数和模块添加文档字符串
- 统一的路径管理方法

### 文件组织
- **code/**: 存放可执行的Python脚本
- **data/**: 存放原始数据文件和考试格式参考
  - `comp_predict_table.csv`: 考试预测文件格式参考
- **model/**: 存放训练好的模型文件
- **prediction_result/**: 存放考试提交的预测结果
  - `tc_comp_predict_table.csv`: 考试最终预测文件（待模型生成）
- **user_data/**: 存放数据处理结果、中间文件和可视化图表
- **feature/**: 存放特征工程相关代码
- **docs/**: 存放项目文档

### 最佳实践
1. 使用相对路径和统一路径管理
2. 数据文件不要提交到版本控制
3. 模型和结果文件分类存放
4. 代码模块化和可复用

## 故障排除

### 常见问题

#### 环境相关
1. **虚拟环境未激活**: 确保已激活uv虚拟环境 `source ../.venv/bin/activate`
2. **Python版本不匹配**: 确认使用的是Python 3.11+
3. **依赖未安装**: uv环境已预装pandas、matplotlib等核心依赖，如需添加额外依赖使用 `uv add 包名`

#### 数据处理
4. **pandas未安装**: 使用简化版本脚本或安装pandas
5. **文件路径错误**: 使用`get_project_path()`函数统一管理路径
6. **数据文件过大**: 使用分块读取或采样分析

#### 权限和路径
7. **权限错误**: 检查文件读写权限，确保在项目目录内操作
8. **相对路径问题**: 注意.venv目录在build-your-own-ai下，从当前项目需要使用`../`

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

## 图表分析脚本补充说明

本项目新增了多个资金流动分析脚本，用于生成不同形式的可视化图表：

### 脚本功能
- **analyze_daily_flow.py**: 基础数据分析和文本概览
- **ascii_chart.py**: 生成ASCII艺术风格的图表（无需外部依赖）
- **html_chart.py**: 生成交互式HTML图表（推荐使用）
- **plot_daily_flow.py**: 生成高质量matplotlib图表（需要安装依赖）

### 使用方法
```bash
# 激活环境后运行
source ../.venv/bin/activate
python code/ascii_chart.py     # 文本版图表
python code/html_chart.py      # 交互式图表
python code/plot_daily_flow.py # 完整图表（需安装matplotlib）
```

### 输出文件
- `prediction_result/daily_flow_chart.html` - 交互式图表
- `prediction_result/daily_flow_summary.csv` - 汇总数据

---

## 实际数据分析结果

基于完整的284万条用户交易记录分析，已生成以下文件：

### 输出文件

#### 考试提交文件
- **最终预测**: `prediction_result/tc_comp_predict_table.csv` - 考试提交的预测文件（待模型训练生成）
- **格式**: YYYYMMDD,申购金额,赎回金额（无表头）

#### 数据处理文件
- **数据汇总**: `user_data/daily_summary.csv` - 每日数据汇总统计（现有数据整理）
- **申购赎回图表**: `user_data/daily_flow_trend.png` - 申购赎回趋势图表
- **图表数据**: `user_data/chart_data.json` - 图表数据文件

### 核心发现
- **数据期间**: 2013年7月1日 至 2014年8月31日（427天）
- **总申购额**: 925.91亿元
- **总赎回额**: 727.18亿元
- **净流入**: 198.73亿元
- **日均申购**: 2.17亿元
- **日均赎回**: 1.70亿元

### 使用建议
1. **快速开始**: 无需激活环境，直接使用 `uv run python` 命令
2. **数据探索**: 查看 `user_data/daily_summary.csv` 每日数据汇总统计
3. **查看图表**: 打开 `user_data/daily_flow_trend.png` 查看申购赎回趋势
4. **模型训练**: 基于现有数据进行机器学习模型训练
5. **考试提交**: 将最终预测结果保存为 `prediction_result/tc_comp_predict_table.csv`
6. **格式参考**: 参考 `data/comp_predict_table.csv` 了解考试文件格式要求

### 快速运行示例

#### 数据分析脚本
```bash
# 基础数据分析
uv run python -c "import pandas as pd; print('uv环境可用')"
uv run python -c "import pandas as pd; df=pd.read_csv('data/user_balance_table.csv'); print(f'数据量: {len(df)}条')"

# 生成每日汇总数据
python3 code/daily_flow_analysis.py

# 生成可视化报告
uv run python3 code/visualize_trends.py
```

#### 生成考试预测文件
```bash
# 生成考试提交预测文件（基于格式参考）
python3 code/generate_prediction.py

# 查看预测结果
cat prediction_result/tc_comp_predict_table.csv
```

---

*本指南将随项目发展持续更新，建议定期查看最新版本。*
*最后更新: 2025年11月20日*  
*完成完整数据分析和验证，更新uv环境配置和实际分析结果*