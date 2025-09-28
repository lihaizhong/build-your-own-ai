
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. 读取数据并进行预处理
def load_and_preprocess_data(filepath):
    """
    读取并预处理数据
    """
    # 读入数据集
    df = pd.read_csv(filepath)
    # 将Datetime列转换为datetime对象，并处理格式
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
    # 将Datetime列设置为索引
    df.set_index('Datetime', inplace=True)
    # 按天聚合，求每天的总数
    df_daily = df['Count'].resample('D').sum().reset_index()
    # Prophet要求列名为ds和y
    df_daily.rename(columns={'Datetime': 'ds', 'Count': 'y'}, inplace=True)
    return df_daily

# 2. 训练Prophet模型
def train_prophet_model(df):
    """
    训练Prophet模型
    """
    # 初始化并拟合模型
    model = Prophet()
    model.fit(df)
    return model

# 3. 预测未来数据
def make_future_prediction(model, periods):
    """
    预测未来数据
    """
    # 构建待预测日期数据框
    future = model.make_future_dataframe(periods=periods)
    # 预测数据集
    forecast = model.predict(future)
    return forecast

# 4. 可视化结果
def plot_forecast(model, forecast):
    """
    可视化结果
    """
    # 绘制预测结果
    fig1 = model.plot(forecast)
    plt.title('未来7个月每日乘客数量预测')
    plt.xlabel('日期')
    plt.ylabel('乘客数量')
    plt.savefig('daily_passenger_forecast.png')
    plt.show()

    # 绘制成分分析
    fig2 = model.plot_components(forecast)
    plt.savefig('daily_passenger_forecast_components.png')
    plt.show()

if __name__ == '__main__':
    # 数据文件路径
    filepath = 'train.csv'
    # 加载和预处理数据
    daily_data = load_and_preprocess_data(filepath)
    print("数据聚合完成，显示最后5条聚合数据:")
    print(daily_data.tail())

    # 训练模型
    prophet_model = train_prophet_model(daily_data)
    print("Prophet模型训练完成.")

    # 预测未来7个月（约210天）
    future_forecast = make_future_prediction(prophet_model, periods=210)
    print("未来7个月数据预测完成，显示预测结果的最后5条:")
    print(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # 可视化结果
    plot_forecast(prophet_model, future_forecast)
    print("预测结果图已保存为 daily_passenger_forecast.png 和 daily_passenger_forecast_components.png") 