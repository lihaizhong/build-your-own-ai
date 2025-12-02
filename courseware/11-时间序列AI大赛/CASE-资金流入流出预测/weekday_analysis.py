import pandas as pd
import matplotlib.pyplot as plt

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据
df = pd.read_csv('user_balance_table.csv', encoding='utf-8')

# 转换 report_date 为日期格式
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 首先，按天聚合得到每日的总申购和赎回额
daily_summary = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum()

# 然后，从日期索引中提取星期几 (Monday=0, Sunday=6)
daily_summary['weekday'] = daily_summary.index.dayofweek

# 按星期几分组，计算申购和赎回金额的均值
weekday_mean = daily_summary.groupby('weekday')[['total_purchase_amt', 'total_redeem_amt']].mean()

# 定义星期名称用于图表显示
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

# 按照星期顺序（0-6）重新索引，以确保绘图顺序正确
weekday_mean = weekday_mean.reindex(range(7))
weekday_mean['weekday_name'] = weekday_names

# 打印计算结果
print("按周一到周日计算的申购和赎回均值:")
print(weekday_mean)

# 使用折线图呈现结果
plt.figure(figsize=(12, 7))

plt.plot(weekday_mean['weekday_name'], weekday_mean['total_purchase_amt'], marker='o', linestyle='-', label='平均总申购金额')
plt.plot(weekday_mean['weekday_name'], weekday_mean['total_redeem_amt'], marker='o', linestyle='-', label='平均总赎回金额')

plt.title('周一至周日平均总申购与总赎回金额分析')
plt.xlabel('星期')
plt.ylabel('平均金额')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# 每日分析：按1-31日计算均值
# 从每日汇总数据中提取月份中的日期
daily_summary['day_of_month'] = daily_summary.index.day

# 按月份中的日期分组，计算申购和赎回金额的均值
day_of_month_mean = daily_summary.groupby('day_of_month')[['total_purchase_amt', 'total_redeem_amt']].mean()

# 确保索引是从1到31
day_of_month_mean = day_of_month_mean.reindex(range(1, 32))

# 打印计算结果
print("\n按1-31日计算的申购和赎回均值:")
print(day_of_month_mean)

# 使用折线图呈现结果
plt.figure(figsize=(15, 8))

plt.plot(day_of_month_mean.index, day_of_month_mean['total_purchase_amt'], marker='o', linestyle='-', label='平均总申购金额')
plt.plot(day_of_month_mean.index, day_of_month_mean['total_redeem_amt'], marker='o', linestyle='-', label='平均总赎回金额')

plt.title('每月1日至31日平均总申购与总赎回金额分析')
plt.xlabel('日期（日）')
plt.ylabel('平均金额')
plt.xticks(range(1, 32)) # 设置X轴刻度为1到31
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show() 