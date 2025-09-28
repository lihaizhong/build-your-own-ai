import pandas as pd

# 读取前5行数据
df = pd.read_csv('train.csv', nrows=5, encoding='utf-8')
print(df)