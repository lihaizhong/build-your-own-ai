import pandas as pd
import numpy as np

def main():
    # 读取数据
    df = pd.read_csv('user_balance_table.csv', encoding='utf-8')
    df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

    # 只保留2014-03-01到2014-08-31的数据
    mask = (df['report_date'] >= '2014-03-01') & (df['report_date'] <= '2014-08-31')
    df = df.loc[mask]

    # 按天聚合
    daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
    daily['weekday'] = daily['report_date'].dt.dayofweek
    daily['day'] = daily['report_date'].dt.day

    # 全局均值
    purchase_mean = daily['total_purchase_amt'].mean()
    redeem_mean = daily['total_redeem_amt'].mean()

    # 计算 weekday 因子
    weekday_purchase = daily.groupby('weekday')['total_purchase_amt'].mean() / purchase_mean
    weekday_redeem = daily.groupby('weekday')['total_redeem_amt'].mean() / redeem_mean

    # 计算 day 因子
    day_purchase = daily.groupby('day')['total_purchase_amt'].mean() / purchase_mean
    day_redeem = daily.groupby('day')['total_redeem_amt'].mean() / redeem_mean

    # 预测未来30天
    future_dates = pd.date_range('2014-09-01', periods=30, freq='D')
    future = pd.DataFrame({'report_date': future_dates})
    future['weekday'] = future['report_date'].dt.dayofweek
    future['day'] = future['report_date'].dt.day

    # 预测值 = 全局均值 × weekday因子 × day因子
    future['purchase'] = purchase_mean * future['weekday'].map(weekday_purchase) * future['day'].map(day_purchase)
    future['redeem'] = redeem_mean * future['weekday'].map(weekday_redeem) * future['day'].map(day_redeem)

    # clip负值
    future['purchase'] = future['purchase'].clip(0)
    future['redeem'] = future['redeem'].clip(0)

    # 输出
    output = future[['report_date', 'purchase', 'redeem']].copy()
    output['report_date'] = output['report_date'].dt.strftime('%Y%m%d')
    output.to_csv('periodic_factor_forecast_201409.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print('预测结果已保存到 periodic_factor_forecast_201409.csv')
    print(output)

if __name__ == '__main__':
    main() 