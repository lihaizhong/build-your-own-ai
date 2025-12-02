"""
V3版本模型分析脚本 - 深入分析模型表现和优化方向

目标:
1. 分析V3模型在不同价格区间的预测表现
2. 识别模型在特定特征上的弱点
3. 提供更具体的优化建议
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
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

def load_v3_predictions():
    """加载V3模型的预测结果"""
    # 查找最新的V3预测文件
    pred_dir = get_project_path('prediction_result')
    v3_files = []
    
    if os.path.exists(pred_dir):
        for file in os.listdir(pred_dir):
            if 'lgmb_modeling_v3' in file and file.endswith('.csv'):
                v3_files.append(os.path.join(pred_dir, file))
    
    if v3_files:
        # 选择最新的V3预测文件
        latest_v3_file = sorted(v3_files)[-1]
        pred_df = pd.read_csv(latest_v3_file)
        print(f"加载V3预测文件: {latest_v3_file}")
        return pred_df, latest_v3_file
    else:
        print("未找到V3预测文件")
        return None, None

def analyze_price_segments(train_df, pred_df):
    """分析不同价格区间的预测表现"""
    print("\n=== 价格区间分析 ===")
    
    if pred_df is None:
        print("无法分析价格区间，缺少预测数据")
        return
    
    # 将训练集价格分段
    train_df['price_segment'] = pd.cut(train_df['price'], 
                                      bins=[0, 5000, 10000, 15000, 20000, 30000, 50000, 100000],
                                      labels=['<5K', '5K-10K', '10K-15K', '15K-20K', '20K-30K', '30K-50K', '>50K'])
    
    # 计算每个价格区间的样本数和统计信息
    segment_stats = train_df['price_segment'].value_counts().sort_index()
    print("训练集价格区间分布:")
    for segment, count in segment_stats.items():
        print(f"  {segment}: {count} 样本 ({count/len(train_df)*100:.1f}%)")
    
    # 分析预测值的分布
    pred_df['price_segment'] = pd.cut(pred_df['price'], 
                                     bins=[0, 5000, 10000, 15000, 20000, 30000, 50000, 100000],
                                     labels=['<5K', '5K-10K', '10K-15K', '15K-20K', '20K-30K', '30K-50K', '>50K'])
    
    pred_segment_stats = pred_df['price_segment'].value_counts().sort_index()
    print("\n预测值价格区间分布:")
    for segment, count in pred_segment_stats.items():
        print(f"  {segment}: {count} 样本 ({count/len(pred_df)*100:.1f}%)")

def analyze_feature_importance():
    """分析特征重要性"""
    print("\n=== 特征重要性分析 ===")
    
    # 查找V3模型文件
    model_dir = get_project_path('model')
    v3_model_file = None
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file == 'modeling_v3.py':
                v3_model_file = os.path.join(model_dir, file)
                break
    
    if v3_model_file:
        print(f"找到V3模型文件: {v3_model_file}")
        # 这里可以进一步分析模型中的特征重要性
        # 由于我们没有保存模型，只能通过代码分析
        print("特征重要性需要通过重新训练模型来获取")
    else:
        print("未找到V3模型文件")

def analyze_prediction_errors(train_df, pred_df):
    """分析预测误差"""
    print("\n=== 预测误差分析 ===")
    
    if pred_df is None:
        print("无法分析预测误差，缺少预测数据")
        return
    
    print(f"预测结果统计:")
    print(f"  均值: {pred_df['price'].mean():.2f}")
    print(f"  中位数: {pred_df['price'].median():.2f}")
    print(f"  标准差: {pred_df['price'].std():.2f}")
    print(f"  最小值: {pred_df['price'].min():.2f}")
    print(f"  最大值: {pred_df['price'].max():.2f}")

def compare_v2_v3_performance():
    """比较V2和V3模型性能"""
    print("\n=== V2与V3模型性能比较 ===")
    
    # 查找V2和V3的预测文件
    pred_dir = get_project_path('prediction_result')
    v2_files = []
    v3_files = []
    
    if os.path.exists(pred_dir):
        for file in os.listdir(pred_dir):
            if 'modeling_v2' in file and file.endswith('.csv'):
                v2_files.append(os.path.join(pred_dir, file))
            elif 'modeling_v3' in file and file.endswith('.csv'):
                v3_files.append(os.path.join(pred_dir, file))
    
    print(f"找到 {len(v2_files)} 个V2预测文件")
    print(f"找到 {len(v3_files)} 个V3预测文件")
    
    if v2_files and v3_files:
        # 选择最新的文件进行比较
        latest_v2 = sorted(v2_files)[-1]
        latest_v3 = sorted(v3_files)[-1]
        
        v2_pred = pd.read_csv(latest_v2)
        v3_pred = pd.read_csv(latest_v3)
        
        print(f"V2预测文件: {os.path.basename(latest_v2)}")
        print(f"V3预测文件: {os.path.basename(latest_v3)}")
        
        print(f"V2预测均值: {v2_pred['price'].mean():.2f}")
        print(f"V3预测均值: {v3_pred['price'].mean():.2f}")
        
        print(f"V2预测标准差: {v2_pred['price'].std():.2f}")
        print(f"V3预测标准差: {v3_pred['price'].std():.2f}")

def generate_detailed_report():
    """生成详细分析报告"""
    print("\n" + "="*60)
    print("V3模型详细分析报告")
    print("="*60)
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 加载V3预测结果
    pred_df, pred_file = load_v3_predictions()
    
    # 分析价格区间
    analyze_price_segments(train_df, pred_df)
    
    # 分析特征重要性
    analyze_feature_importance()
    
    # 分析预测误差
    analyze_prediction_errors(train_df, pred_df)
    
    # 比较V2和V3性能
    compare_v2_v3_performance()
    
    print("\n" + "="*60)
    print("可能的优化方向")
    print("="*60)
    print("1. 更精细的价格分段:")
    print("   - 当前使用10000作为分界点可能不够精确")
    print("   - 可以尝试多个分段点(5K, 10K, 20K, 30K等)")
    print("")
    print("2. 增强特征工程:")
    print("   - 添加更多时间序列特征")
    print("   - 使用更复杂的交叉特征")
    print("   - 考虑非线性特征变换")
    print("")
    print("3. 模型集成优化:")
    print("   - 调整不同模型的权重")
    print("   - 添加CatBoost等其他模型")
    print("   - 使用Stacking而非简单加权平均")
    print("")
    print("4. 高级校准技术:")
    print("   - 使用分位数回归")
    print("   - 实现局部校准")
    print("   - 考虑贝叶斯校准方法")
    print("")
    print("5. 引入深度学习:")
    print("   - 使用神经网络捕获复杂模式")
    print("   - 实现混合模型(传统ML+深度学习)")
    print("="*60)

def main():
    """主函数"""
    print("开始V3模型分析...")
    
    # 生成详细报告
    generate_detailed_report()
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()