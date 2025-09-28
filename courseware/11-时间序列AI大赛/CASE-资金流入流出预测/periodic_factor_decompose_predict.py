import pandas as pd
import numpy as np

def fit_multiplicative_factors(data, value_col, weekday_col, day_col, n_iter=20, eps=1e-8):
    """
    协同分解法：
    purchase ≈ mean × weekday_factor × day_factor
    通过交替归一化迭代拟合 weekday 和 day 的乘法因子。
    """
    # 初始化
    mean = data[value_col].mean()
    weekday_ids = sorted(data[weekday_col].unique())
    day_ids = sorted(data[day_col].unique())
    weekday_factor = pd.Series(1.0, index=weekday_ids)
    day_factor = pd.Series(1.0, index=day_ids)

    for _ in range(n_iter):
        # 更新 weekday 因子
        for w in weekday_ids:
            mask = data[weekday_col] == w
            if mask.sum() > 0:
                weekday_factor[w] = (data.loc[mask, value_col] / (mean * day_factor[data.loc[mask, day_col]].values + eps)).mean()
        # 归一化
        weekday_factor /= weekday_factor.mean()
        # 更新 day 因子
        for d in day_ids:
            mask = data[day_col] == d
            if mask.sum() > 0:
                day_factor[d] = (data.loc[mask, value_col] / (mean * weekday_factor[data.loc[mask, weekday_col]].values + eps)).mean()
        # 归一化
        day_factor /= day_factor.mean()
    return mean, weekday_factor, day_factor

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

    # 拟合申购因子
    purchase_mean, purchase_weekday_factor, purchase_day_factor = fit_multiplicative_factors(
        daily, 'total_purchase_amt', 'weekday', 'day')
    # 拟合赎回因子
    redeem_mean, redeem_weekday_factor, redeem_day_factor = fit_multiplicative_factors(
        daily, 'total_redeem_amt', 'weekday', 'day')

    # 预测未来30天
    future_dates = pd.date_range('2014-09-01', periods=30, freq='D')
    future = pd.DataFrame({'report_date': future_dates})
    future['weekday'] = future['report_date'].dt.dayofweek
    future['day'] = future['report_date'].dt.day

    # 预测值 = mean × weekday因子 × day因子
    future['purchase'] = purchase_mean * future['weekday'].map(purchase_weekday_factor) * future['day'].map(purchase_day_factor)
    future['redeem'] = redeem_mean * future['weekday'].map(redeem_weekday_factor) * future['day'].map(redeem_day_factor)

    # clip负值
    future['purchase'] = future['purchase'].clip(0)
    future['redeem'] = future['redeem'].clip(0)

    # 输出
    output = future[['report_date', 'purchase', 'redeem']].copy()
    output['report_date'] = output['report_date'].dt.strftime('%Y%m%d')
    output.to_csv('periodic_factor_decompose_forecast_201409.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print('预测结果已保存到 periodic_factor_decompose_forecast_201409.csv')
    print(output)

if __name__ == '__main__':
    main()
