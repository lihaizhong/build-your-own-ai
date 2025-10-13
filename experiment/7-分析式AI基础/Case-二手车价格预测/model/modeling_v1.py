# 导入必要的数据处理和机器学习库
import os
import pandas as pd          # 用于数据操作和分析
import numpy as np           # 用于数值计算
from sklearn.model_selection import train_test_split  # 用于数据分割和交叉验证
from sklearn.ensemble import RandomForestRegressor          # 随机森林回归器
from sklearn.linear_model import Ridge                                       # 岭回归
from sklearn.preprocessing import LabelEncoder               # 标签编码器,标准化器
from sklearn.metrics import mean_absolute_error                              # 平均绝对误差
import lightgbm as lgb                                                       # LightGBM 机器学习库
import warnings                                                              # 用于警告管理
warnings.filterwarnings('ignore')                                            # 忽略警告信息

def get_project_path(*paths):
    """
    获取项目路径的统一方法
    类似TypeScript中的path.join(__dirname, ...paths)
    """
    try:
        # __file__ 类似于 TypeScript 中的 __filename
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录(向上一级)
        project_dir = os.path.dirname(current_dir)
        
        return os.path.join(project_dir, *paths)
    except NameError:
        # 在某些环境下(如Jupyter)__file__不可用,使用当前工作目录
        return os.path.join(os.getcwd(), *paths)

def get_user_data_path(*paths):
    """
    获取用户数据路径
    相当于TypeScript中的 path.join(projectRoot, 'user_data', ...paths)
    """
    return get_project_path('user_data', *paths)

def load_and_preprocess_data():
    """
    加载并预处理数据
    包括:加载数据、合并训练测试集、处理异常值、缺失值填充、特征工程、重新分离
    """
    # 定义训练和测试数据路径(使用统一的路径管理方法)
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    # 读取CSV文件到DataFrame(指定空格分隔符以处理格式问题,并将'-'作为缺失值)
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    # 打印原始数据集的形状和列名
    print(f"原始训练集: {train_df.shape}")
    print(f"训练集列名: {train_df.columns.tolist()[:10]}...")  # 只显示前10个列名
    print(f"原始测试集: {test_df.shape}")
    
    # 合并训练和测试数据，用于统一预处理（如编码、填充缺失值）
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 检查power列是否存在,如果不存在则跳过处理
    if 'power' in all_df.columns:
        # 修复power列中的异常值(<0 或 >600的值):将值限制在[0, 600]范围内
        all_df['power'] = np.clip(all_df['power'], 0, 600)
        print(f"处理了 {len(all_df[(all_df['power'] < 0) | (all_df['power'] > 600)])} 个power异常值")
    else:
        print(f"警告: 数据中未找到'power'列,可用列名: {all_df.columns.tolist()[:10]}...")
    
    # 对分类特征进行缺失值处理:用众数填充,并创建缺失指示变量
    for col in ['fuelType', 'gearbox', 'bodyType']:
        mode_value = all_df[col].mode()
        if len(mode_value) > 0:
            all_df[col] = all_df[col].fillna(mode_value.iloc[0])  # 用众数填充
        all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)  # 创建缺失指示变量

    # 对model列进行缺失值处理(仅填充众数)
    model_mode = all_df['model'].mode()
    if len(model_mode) > 0:
        all_df['model'] = all_df['model'].fillna(model_mode.iloc[0])

    # 特征工程开始
    # 解析regDate时间特征
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020  # 假设当前年份为2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year  # 计算车龄
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)   # 填充缺失年份为0，并转为整数
    
    # 删除原始的regDate列，因为已提取为car_age
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 创建动力与车龄的交互特征(仅在power列存在时)
    if 'power' in all_df.columns:
        all_df['power_age_ratio'] = all_df['power'] / (all_df['car_age'] + 1)
    else:
        all_df['power_age_ratio'] = 0  # 如果没有power列,设置默认值0
    
    # 品牌统计特征（Target Encoding with smoothing）
    if 'price' in all_df.columns:  # 确保price列存在
        # 计算每个品牌的平均价格和样本数
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        # 使用平滑公式计算平滑均值:(mean*count + global_mean*10) / (count+10)
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + all_df['price'].mean() * 10) / (brand_stats['count'] + 10)
        # 将平滑均值映射回原数据框
        brand_map: dict = brand_stats.set_index('brand')['smooth_mean'].to_dict()  # type: ignore
        all_df['brand_avg_price'] = all_df['brand'].map(brand_map)  # type: ignore
    
    # 对分类特征进行标签编码
    categorical_cols = ['model', 'brand', 'fuelType', 'gearbox', 'bodyType']
    for col in categorical_cols:
        if col in all_df.columns:  # 确保列存在
            le = LabelEncoder()  # 创建标签编码器实例
            all_df[col] = le.fit_transform(all_df[col].astype(str))  # 拟合并转换为数值
    
    # 填充所有剩余的数值型列的缺失值(使用中位数)
    numeric_cols = all_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 使用bool()显式转换以避免类型检查器警告
        has_null = bool(all_df[col].isnull().any())  # type: ignore[arg-type]
        if has_null:
            median_value = all_df[col].median()
            all_df[col] = all_df[col].fillna(median_value)
            print(f"用中位数{median_value:.2f}填充了列'{col}'的缺失值")
    
    # 重新分离训练集和测试集
    train_df = all_df.iloc[:len(train_df)].copy()  # 前N行作为训练集
    test_df = all_df.iloc[len(train_df):].copy()   # 后M行作为测试集
    
    # 删除价格异常值（使用IQR方法识别）
    Q1 = train_df['price'].quantile(0.25)  # 第一四分位数
    Q3 = train_df['price'].quantile(0.75)  # 第三四分位数
    IQR = Q3 - Q1  # 四分位距
    lower_bound = Q1 - 1.5 * IQR  # 下界
    upper_bound = Q3 + 1.5 * IQR  # 上界
    # 保留价格在合理范围内的样本
    train_df = train_df[(train_df['price'] >= lower_bound) & (train_df['price'] <= upper_bound)]
    
    # 打印清洗后的数据集形状
    print(f"清理后训练集: {train_df.shape}")
    print(f"清理后测试集: {test_df.shape}")
    
    return train_df, test_df

def create_features(df):
    """
    创建更多高级特征
    :param df: 输入的DataFrame
    :return: 添加了新特征的DataFrame
    """
    df = df.copy()  # 复制DataFrame以避免修改原对象
    
    # 车龄分段特征：将连续的车龄转换为离散类别
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], labels=['new', 'medium', 'old', 'vintage'])
    df['age_segment'] = df['age_segment'].cat.codes  # 将类别转换为数值编码
    
    # 动力分段特征：根据power值分段(仅在power列存在时)
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 50, 100, 150, 200, float('inf')], labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['power_segment'] = df['power_segment'].cat.codes  # 将类别转换为数值编码
    else:
        df['power_segment'] = 0  # 如果没有power列,设置默认值0
    
    # 里程与车龄比值（如果存在kilometer字段）
    # if 'kilometer' in df.columns:
    #     df['mileage_per_year'] = df['kilometer'] / (df['car_age'] + 1)
    
    # 功率与价格比（如果price在当前数据中可用）
    # if 'price' in df.columns:
    #     df['power_price_ratio'] = df['power'] / (df['price'] + 1)
    
    return df

def optimize_model():
    """
    主优化流程函数
    执行完整的数据预处理、模型训练、验证和预测流程
    """
    print("开始优化...")
    # 加载并预处理数据
    train_df, test_df = load_and_preprocess_data()
    
    # 应用额外的特征工程
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # 准备特征和目标变量
    y_col = 'price'  # 目标变量列名
    # 获取所有特征列名（排除目标变量和SaleID）
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    # 分离特征和目标
    X_train = train_df[feature_cols].copy()  # 训练特征
    y_train = train_df[y_col].copy()         # 训练目标
    X_test = test_df[feature_cols].copy()    # 测试特征
    
    # 对目标变量进行对数变换（log1p = log(1+x)），以处理偏态分布
    y_train_log = np.log1p(y_train)
    
    # 定义多个模型
    # 1. 优化的随机森林模型
    # 根据sklearn官方文档: max_features可以是 {"sqrt", "log2", None}, int或float
    # 参考: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    rf_model = RandomForestRegressor(
        n_estimators=200,         # 树的数量
        max_depth=6,              # 最大深度,控制过拟合
        min_samples_split=50,     # 分割内部节点所需的最小样本数
        min_samples_leaf=30,      # 叶节点最小样本数
        max_features='sqrt',      # 每次分割考虑的最大特征数(使用sqrt策略,完全合法) # type: ignore[arg-type]
        random_state=42,          # 随机种子
        n_jobs=-1                 # 并行使用所有CPU核心
    )
    
    # 2. LightGBM模型参数
    lgb_params = {
        'objective': 'mae',       # 目标函数：平均绝对误差
        'metric': 'mae',          # 评估指标：MAE
        'boosting_type': 'gbdt',  # 提升类型：梯度提升决策树
        'num_leaves': 64,         # 叶节点数量
        'max_depth': 6,           # 最大深度
        'min_data_in_leaf': 50,   # 叶节点最小样本数
        'feature_fraction': 0.8,  # 每次构建树时随机选择的特征比例
        'bagging_fraction': 0.8,  # 每次迭代时随机选择的数据比例
        'bagging_freq': 5,        # 每5次迭代执行一次bagging
        'lambda_l1': 0.1,         # L1正则化项
        'lambda_l2': 0.1,         # L2正则化项
        'verbose': -1,            # 不输出训练信息
        'random_state': 42        # 随机种子
    }
    
    # 创建LightGBM回归器实例
    lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=300)
    
    # 3. Ridge回归模型（用于Stacking的元学习器之一）
    ridge_model = Ridge(alpha=1.0)  # alpha为正则化强度
    
    # 数据分割用于验证（80%训练，20%验证）
    X_tr, X_val, y_tr_log, y_val_log = train_test_split(
        X_train, y_train_log, test_size=0.2, random_state=42
    )
    
    # 训练各个模型
    print("训练模型...")
    
    # 训练随机森林模型（在log变换后的目标上）
    rf_model.fit(X_tr, y_tr_log)
    rf_pred_val_log = rf_model.predict(X_val)  # 预测log变换后的目标
    rf_pred_val = np.expm1(rf_pred_val_log)    # 反变换回原始尺度
    rf_mae_val = mean_absolute_error(np.expm1(y_val_log), rf_pred_val)  # 计算MAE
    print(f"RF 验证 MAE (反变换): {rf_mae_val:.2f}")
    
    # 训练LightGBM模型(在log变换后的目标上)
    lgb_model.fit(X_tr, y_tr_log)
    lgb_pred_val_log = np.array(lgb_model.predict(X_val))  # 预测log变换后的目标
    lgb_pred_val = np.expm1(lgb_pred_val_log)    # 反变换回原始尺度
    lgb_mae_val = mean_absolute_error(np.expm1(y_val_log), lgb_pred_val)  # 计算MAE
    print(f"LGBM 验证 MAE (反变换): {lgb_mae_val:.2f}")
    
    # 训练Ridge模型（在log变换后的目标上）
    ridge_model.fit(X_tr, y_tr_log)
    ridge_pred_val_log = ridge_model.predict(X_val)  # 预测log变换后的目标
    ridge_pred_val = np.expm1(ridge_pred_val_log)    # 反变换回原始尺度
    ridge_mae_val = mean_absolute_error(np.expm1(y_val_log), ridge_pred_val)  # 计算MAE
    print(f"Ridge 验证 MAE (反变换): {ridge_mae_val:.2f}")
    
    # Stacking 预测器:将三个模型的预测结果作为新特征
    stack_train = np.column_stack([rf_pred_val, lgb_pred_val, ridge_pred_val])  # 验证集上的预测
    stack_test = np.column_stack([
        np.expm1(rf_model.predict(X_test)),   # 测试集上RF的预测
        np.expm1(np.array(lgb_model.predict(X_test))),  # 测试集上LGB的预测
        np.expm1(ridge_model.predict(X_test)) # 测试集上Ridge的预测
    ])
    
    # 元学习器（Ridge）：在stack_train上训练，目标是原始价格
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_train, np.expm1(y_val_log))  # 拟合元模型
    final_pred_val = meta_model.predict(stack_train)  # 预测验证集
    final_mae_val = mean_absolute_error(np.expm1(y_val_log), final_pred_val)  # 计算最终验证MAE
    print(f"Stacking 验证 MAE: {final_mae_val:.2f}")
    
    # 在全量训练集上重新训练所有基模型
    print("在全量训练集上重新训练...")
    rf_model.fit(X_train, y_train_log)   # 重新训练RF
    lgb_model.fit(X_train, y_train_log)  # 重新训练LGB
    ridge_model.fit(X_train, y_train_log) # 重新训练Ridge
    
    # 使用重新训练的模型进行最终预测
    rf_pred_test = np.expm1(rf_model.predict(X_test))   # RF预测
    lgb_pred_test = np.expm1(np.array(lgb_model.predict(X_test))) # LGB预测
    ridge_pred_test = np.expm1(ridge_model.predict(X_test)) # Ridge预测
    
    # 将测试集上的预测结果进行Stacking
    final_stack_test = np.column_stack([rf_pred_test, lgb_pred_test, ridge_pred_test])
    final_pred_test = meta_model.predict(final_stack_test)  # 元模型预测最终结果
    
    # 打印最终预测的统计信息
    print(f"最终集成预测均值: {final_pred_test.mean():.2f}")
    print(f"最终集成预测范围: {final_pred_test.min():.2f} - {final_pred_test.max():.2f}")
    
    # 创建提交格式的DataFrame
    submission_df = pd.DataFrame({
        'SaleID': test_df.index,  # 假设索引对应SaleID（可能需要根据实际情况调整）
        'price': final_pred_test  # 最终预测价格
    })
    
    # 保存预测结果到CSV文件(使用统一的路径管理方法)
    result_dir = get_project_path('prediction_result')
    os.makedirs(result_dir, exist_ok=True)  # 确保目录存在
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    result_file = os.path.join(result_dir, f"modeling_v1_result_{timestamp}.csv")  # 构建文件路径
    submission_df.to_csv(result_file, index=False)  # 保存文件,不包含索引
    print(f"结果已保存到: {result_file}")
    
    return final_mae_val, final_pred_test  # 返回验证MAE和测试预测结果

if __name__ == "__main__":
    # 当脚本直接运行时执行主函数
    val_mae, test_pred = optimize_model()  # 获取验证MAE和测试预测
    print(f"\n优化后验证 MAE: {val_mae:.2f}")  # 打印验证MAE
    print(f"目标: < 500, 当前验证 MAE: {val_mae:.2f}")  # 打印目标与实际对比
    if val_mae < 500:  # 判断是否达到目标
        print("✅ 优化成功！验证 MAE 已低于 500")  # 成功提示
    else:
        print(f"⚠️  验证 MAE 仍高于 500，还需进一步优化")  # 失败提示



