import pandas as pd
import statsmodels.formula.api as smf

def generate_periodic_forecast():
    """
    使用周期性因子（星期和月中日期）进行预测。
    """
    # 1. 加载和准备数据
    try:
        df = pd.read_csv('user_balance_table.csv', encoding='utf-8')
    except FileNotFoundError:
        print("错误：user_balance_table.csv 文件未找到。请确保文件在当前目录中。")
        return

    df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

    # 只保留2014-03-01到2014-08-31的数据
    mask = (df['report_date'] >= '2014-03-01') & (df['report_date'] <= '2014-08-31')
    df = df.loc[mask]

    # 按天聚合申购和赎回总额
    daily_summary = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

    # 2. 特征工程：提取周期因子
    daily_summary['weekday'] = daily_summary['report_date'].dt.dayofweek
    daily_summary['day_of_month'] = daily_summary['report_date'].dt.day

    # 3. 使用线性回归模型（OLS）来学习周期因子的叠加影响
    # C() 会将整数列视为分类变量，为每个类别（如每个星期）创建独立的效应
    
    print("正在构建申购金额预测模型...")
    # 申购金额模型
    purchase_model = smf.ols('total_purchase_amt ~ C(weekday) + C(day_of_month)', data=daily_summary).fit()

    print("正在构建赎回金额预测模型...")
    # 赎回金额模型
    redeem_model = smf.ols('total_redeem_amt ~ C(weekday) + C(day_of_month)', data=daily_summary).fit()
    print("模型构建完成。")

    # 4. 创建未来30天的数据框用于预测
    future_dates = pd.date_range(start='2014-09-01', periods=30, freq='D')
    future_df = pd.DataFrame({'report_date': future_dates})
    future_df['weekday'] = future_df['report_date'].dt.dayofweek
    future_df['day_of_month'] = future_df['report_date'].dt.day

    # 5. 进行预测
    predicted_purchase = purchase_model.predict(future_df)
    predicted_redeem = redeem_model.predict(future_df)

    # 6. 格式化并保存输出结果
    output_df = pd.DataFrame({
        'report_date': future_df['report_date'].dt.strftime('%Y%m%d'),
        'purchase': predicted_purchase,
        'redeem': predicted_redeem
    })
    
    # 保证预测结果不为负数
    output_df['purchase'] = output_df['purchase'].clip(0)
    output_df['redeem'] = output_df['redeem'].clip(0)


    # 保存到 CSV 文件
    output_filename = 'periodic_forecast_201409.csv'
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig', float_format='%.2f')

    print(f"\n预测结果已保存到 {output_filename}")
    print("预测结果预览:")
    print(output_df)

if __name__ == '__main__':
    generate_periodic_forecast() 