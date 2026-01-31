"""
进一步分析脚本 - 深入分析测试集B特征与优化方向

目标:
1. 分析测试集B与训练集的详细分布差异
2. 识别可能导致MAE偏高的关键特征
3. 提供更具体的优化建议
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

def get_user_data_path(*paths):
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

def load_data():
    """加载训练集和测试集数据"""
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df

def analyze_price_distribution(train_df, test_df):
    """分析价格分布差异"""
    print("\n=== 价格分布分析 ===")
    
    train_price = train_df['price']
    # 由于测试集没有price列，我们使用预测值进行分析
    # 这里我们先分析训练集的价格分布特征
    
    print(f"训练集价格统计:")
    print(f"  均值: {train_price.mean():.2f}")
    print(f"  中位数: {train_price.median():.2f}")
    print(f"  标准差: {train_price.std():.2f}")
    print(f"  最小值: {train_price.min():.2f}")
    print(f"  最大值: {train_price.max():.2f}")
    print(f"  25%分位数: {train_price.quantile(0.25):.2f}")
    print(f"  75%分位数: {train_price.quantile(0.75):.2f}")
    
    # 分析高价车比例
    high_price_train = (train_price > 10000).sum() / len(train_price)
    print(f"  高价车(>10000)比例: {high_price_train:.2%}")
    
    return train_price

def analyze_feature_distributions(train_df, test_df):
    """分析特征分布差异"""
    print("\n=== 特征分布分析 ===")
    
    # 合并数据进行统一处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 处理power异常值
    if 'power' in all_df.columns:
        all_df['power'] = np.clip(all_df['power'], 0, 600)
    
    # 分类特征缺失值处理
    for col in ['fuelType', 'gearbox', 'bodyType']:
        if col in all_df.columns:
            mode_value = all_df[col].mode()
            if len(mode_value) > 0:
                all_df[col] = all_df[col].fillna(mode_value.iloc[0])
    
    model_mode = all_df['model'].mode()
    if len(model_mode) > 0:
        all_df['model'] = all_df['model'].fillna(model_mode.iloc[0])
    
    # 特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    
    # 重新分离
    processed_train = all_df.iloc[:len(train_df)].copy()
    processed_test = all_df.iloc[len(train_df):].copy()
    
    # 分析关键特征
    key_features = ['power', 'car_age', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'brand']
    
    print("关键特征在训练集和测试集上的分布对比:")
    for feature in key_features:
        if feature in processed_train.columns:
            train_vals = processed_train[feature]
            test_vals = processed_test[feature]
            
            print(f"\n{feature}:")
            print(f"  训练集 - 均值: {train_vals.mean():.2f}, 标准差: {train_vals.std():.2f}")
            print(f"  测试集 - 均值: {test_vals.mean():.2f}, 标准差: {test_vals.std():.2f}")
            print(f"  差异 - 均值差: {abs(train_vals.mean() - test_vals.mean()):.2f}, "
                  f"标准差差: {abs(train_vals.std() - test_vals.std()):.2f}")

def analyze_prediction_bias():
    """分析预测偏差"""
    print("\n=== 预测偏差分析 ===")
    
    # 加载优化模型的预测结果
    pred_files = []
    pred_dir = get_project_path('prediction_result')
    if os.path.exists(pred_dir):
        for file in os.listdir(pred_dir):
            if 'modeling_v2' in file and file.endswith('.csv'):
                pred_files.append(os.path.join(pred_dir, file))
    
    if pred_files:
        # 选择最新的预测文件
        latest_pred_file = sorted(pred_files)[-1]
        pred_df = pd.read_csv(latest_pred_file)
        print(f"加载预测文件: {latest_pred_file}")
        print(f"预测结果统计:")
        print(f"  均值: {pred_df['price'].mean():.2f}")
        print(f"  中位数: {pred_df['price'].median():.2f}")
        print(f"  标准差: {pred_df['price'].std():.2f}")
        print(f"  最小值: {pred_df['price'].min():.2f}")
        print(f"  最大值: {pred_df['price'].max():.2f}")
    else:
        print("未找到预测文件")

def generate_detailed_report(train_df, test_df):
    """生成详细分析报告"""
    print("\n" + "="*60)
    print("详细分析报告")
    print("="*60)
    
    # 价格分布分析
    train_price = analyze_price_distribution(train_df, test_df)
    
    # 特征分布分析
    analyze_feature_distributions(train_df, test_df)
    
    # 预测偏差分析
    analyze_prediction_bias()
    
    print("\n" + "="*60)
    print("可能的优化方向")
    print("="*60)
    print("1. 分层建模策略:")
    print("   - 针对高价车(>10000)和低价车分别训练模型")
    print("   - 使用不同的特征工程策略")
    print("")
    print("2. 增强正则化:")
    print("   - 进一步降低模型复杂度")
    print("   - 增加L1/L2正则化强度")
    print("   - 使用Dropout等技术(对于神经网络)")
    print("")
    print("3. 特征工程优化:")
    print("   - 添加更多交叉特征")
    print("   - 使用分箱特征")
    print("   - 考虑时间序列特征")
    print("")
    print("4. 集成方法改进:")
    print("   - 调整模型权重")
    print("   - 添加更多基模型")
    print("   - 使用Blending而非简单加权平均")
    print("")
    print("5. 预测校准增强:")
    print("   - 使用更复杂的校准方法")
    print("   - 考虑分段校准")
    print("   - 使用 isotonic regression")
    print("")
    print("6. 数据增强:")
    print("   - 通过SMOTE等方法平衡数据分布")
    print("   - 生成合成样本增强训练集")
    print("="*60)

def main():
    """主函数"""
    print("开始进一步分析...")
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 生成详细报告
    generate_detailed_report(train_df, test_df)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()