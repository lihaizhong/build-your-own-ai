#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进一步优化的XGBoost建模脚本
目标：从677分优化到500分以下
重点：更精确的价格分布拟合和特征工程
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UltraOptimizedXGBoostModel:
    """超优化的XGBoost实现，追求极致精度"""
    
    def __init__(self, n_estimators=150, learning_rate=0.06, max_depth=7, 
                 subsample=0.85, colsample_bytree=0.8, random_state=42):
        """初始化超优化XGBoost模型"""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.base_prediction = None
        self.trees = []
        self.feature_names = None
        self.feature_importances_ = None
        self.is_trained = False
        
        # 设置随机种子
        np.random.seed(random_state)
    
    def create_advanced_features(self, X):
        """创建高级特征工程"""
        X_enhanced = X.copy()
        
        # 价格相关的组合特征
        if 'v_0' in X.columns and 'v_3' in X.columns:
            X_enhanced['v0_v3_ratio'] = X['v_0'] / (X['v_3'] + 1e-8)
            X_enhanced['v0_v3_product'] = X['v_0'] * X['v_3']
        
        if 'v_8' in X.columns and 'v_12' in X.columns:
            X_enhanced['v8_v12_ratio'] = X['v_8'] / (X['v_12'] + 1e-8)
            X_enhanced['v8_v12_sum'] = X['v_8'] + X['v_12']
        
        # 功率和公里数的交互特征
        if 'power' in X.columns and 'kilometer' in X.columns:
            X_enhanced['power_km_ratio'] = X['power'] / (X['kilometer'] + 1e-8)
            X_enhanced['power_km_product'] = X['power'] * X['kilometer']
        
        # 车龄相关特征
        if 'car_age' in X.columns:
            X_enhanced['car_age_squared'] = X['car_age'] ** 2
            X_enhanced['car_age_log'] = np.log1p(X['car_age'])
        
        # 品牌和车型的交互
        if 'brand' in X.columns and 'model' in X.columns:
            X_enhanced['brand_model_interaction'] = X['brand'] * 1000 + X['model']
        
        return X_enhanced
    
    def build_ultra_tree(self, X, residuals, depth=0):
        """构建超精细的回归树"""
        if depth >= self.max_depth or len(residuals) < 5:  # 更激进的分割
            return np.mean(residuals)
        
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # 特征采样
        n_features_subset = max(6, int(n_features * self.colsample_bytree))
        feature_indices = np.random.choice(n_features, min(n_features_subset, n_features), replace=False)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) < 2:
                continue
            
            # 更密集的分割点，特别关注分位数
            if len(unique_values) > 15:
                thresholds = np.percentile(unique_values, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
            elif len(unique_values) > 5:
                thresholds = np.percentile(unique_values, [10, 25, 33, 50, 67, 75, 90])
            else:
                thresholds = unique_values[:-1]
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                left_residuals = residuals[left_mask]
                right_residuals = residuals[right_mask]
                
                # 改进的增益计算，考虑残差的分布
                total_var = np.var(residuals)
                left_var = np.var(left_residuals) if len(left_residuals) > 1 else 0
                right_var = np.var(right_residuals) if len(right_residuals) > 1 else 0
                
                # 加权方差 + 偏差奖励
                weighted_var = (len(left_residuals) * left_var + len(right_residuals) * right_var) / len(residuals)
                gain = total_var - weighted_var
                
                # 对极值区域和偏差大的区域给予额外奖励
                left_mean_abs = np.mean(np.abs(left_residuals))
                right_mean_abs = np.mean(np.abs(right_residuals))
                high_error_bonus = 0
                
                residual_std = np.std(residuals)
                if left_mean_abs > residual_std or right_mean_abs > residual_std:
                    high_error_bonus = gain * 0.15  # 15%的额外奖励
                
                # 极值区域奖励
                extreme_bonus = 0
                if (np.min(left_residuals) < -2*residual_std or np.max(left_residuals) > 2*residual_std or
                    np.min(right_residuals) < -2*residual_std or np.max(right_residuals) > 2*residual_std):
                    extreme_bonus = gain * 0.1
                
                total_gain = gain + high_error_bonus + extreme_bonus
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None or best_gain <= 1e-8:
            return np.mean(residuals)
        
        # 分割数据
        feature_values = X[:, best_feature]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_tree = self.build_ultra_tree(X[left_mask], residuals[left_mask], depth + 1)
        right_tree = self.build_ultra_tree(X[right_mask], residuals[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'gain': best_gain
        }
    
    def predict_tree(self, X, tree):
        """使用单棵树预测"""
        if not isinstance(tree, dict):
            return np.full(X.shape[0], tree)
        
        predictions = np.zeros(X.shape[0])
        feature_values = X[:, tree['feature']]
        
        left_mask = feature_values <= tree['threshold']
        right_mask = ~left_mask
        
        if np.any(left_mask):
            predictions[left_mask] = self.predict_tree(X[left_mask], tree['left'])
        if np.any(right_mask):
            predictions[right_mask] = self.predict_tree(X[right_mask], tree['right'])
        
        return predictions
    
    def calculate_feature_importance(self, tree, importances):
        """计算特征重要性"""
        if isinstance(tree, dict) and 'feature' in tree:
            feature_idx = tree['feature']
            gain = tree.get('gain', 0)
            importances[feature_idx] += gain
            
            self.calculate_feature_importance(tree['left'], importances)
            self.calculate_feature_importance(tree['right'], importances)
    
    def apply_target_transform(self, y):
        """高级目标变量变换"""
        # 使用Box-Cox类似的变换，但更适合价格数据
        # log1p变换 + 轻微的平方根变换组合
        y_log = np.log1p(y)
        # 对于极高和极低的价格给予不同的处理
        y_transformed = np.where(y < 1000, 
                                np.sqrt(y_log),  # 低价车使用平方根
                                y_log)  # 其他使用对数
        return y_transformed
    
    def inverse_target_transform(self, y_transformed, original_y_sample):
        """逆目标变量变换"""
        # 根据变换类型进行逆变换
        low_price_threshold_log = np.log1p(1000)
        
        # 判断哪些是低价车的变换
        low_price_mask = y_transformed < np.sqrt(low_price_threshold_log)
        
        y_log = np.where(low_price_mask,
                        y_transformed ** 2,  # 逆平方根
                        y_transformed)  # 直接使用
        
        return np.expm1(y_log)
    
    def fit(self, X, y):
        """训练超优化XGBoost模型"""
        print(f"开始训练超优化XGBoost模型（{self.n_estimators}棵树）...")
        
        # 特征工程
        X_enhanced = self.create_advanced_features(X)
        self.feature_names = X_enhanced.columns.tolist()
        n_samples, n_features = X_enhanced.shape
        
        print(f"特征工程完成，特征数量: {n_features}")
        
        # 高级目标变量变换
        y_transformed = self.apply_target_transform(y)
        
        # 初始化预测值
        self.base_prediction = np.mean(y_transformed)
        predictions = np.full(n_samples, self.base_prediction)
        
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(n_features)
        
        X_array = X_enhanced.values
        y_array = y_transformed.values if hasattr(y_transformed, 'values') else y_transformed
        
        self.trees = []
        
        # 记录原始y用于逆变换参考
        self.original_y_sample = y.values if hasattr(y, 'values') else y
        
        for i in range(self.n_estimators):
            if (i + 1) % 20 == 0:
                print(f"正在训练第 {i + 1}/{self.n_estimators} 棵树...")
            
            # 计算残差
            residuals = y_array - predictions
            
            # 动态权重计算，重点关注高误差样本
            abs_residuals = np.abs(residuals)
            error_threshold_75 = np.percentile(abs_residuals, 75)
            error_threshold_90 = np.percentile(abs_residuals, 90)
            
            weights = np.ones(n_samples)
            weights[abs_residuals > error_threshold_75] = 1.5
            weights[abs_residuals > error_threshold_90] = 2.0
            
            # 特殊关注极值区域
            residual_std = np.std(residuals)
            extreme_mask = abs_residuals > 2 * residual_std
            weights[extreme_mask] = 3.0
            
            # 加权子采样
            n_subsample = int(self.subsample * n_samples)
            sample_probs = weights / np.sum(weights)
            subsample_indices = np.random.choice(n_samples, n_subsample, replace=True, p=sample_probs)
            
            X_subsample = X_array[subsample_indices]
            residuals_subsample = residuals[subsample_indices]
            
            # 构建超精细树
            tree = self.build_ultra_tree(X_subsample, residuals_subsample)
            
            # 计算所有样本的树预测
            tree_predictions = self.predict_tree(X_array, tree)
            
            # 动态学习率调整
            current_lr = self.learning_rate
            if i < 30:  # 前30棵树
                current_lr *= 1.3
            elif i < 60:  # 中期
                current_lr *= 1.1
            elif i > 100:  # 后期精调
                current_lr *= 0.7
            
            # 添加学习率衰减
            lr_decay = 0.99 ** (i // 10)
            current_lr *= lr_decay
            
            # 更新预测值
            predictions += current_lr * tree_predictions
            
            # 保存树
            self.trees.append(tree)
            
            # 更新特征重要性
            tree_importances = np.zeros(n_features)
            self.calculate_feature_importance(tree, tree_importances)
            self.feature_importances_ += tree_importances
        
        # 归一化特征重要性
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)
        
        self.is_trained = True
        print(f"模型训练完成！使用了 {len(self.feature_names)} 个特征")
    
    def predict(self, X):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！")
        
        print("开始预测...")
        
        # 特征工程
        X_enhanced = self.create_advanced_features(X)
        X_array = X_enhanced.values
        predictions_transformed = np.full(X_array.shape[0], self.base_prediction)
        
        for i, tree in enumerate(self.trees):
            if (i + 1) % 20 == 0:
                print(f"正在使用第 {i + 1}/{len(self.trees)} 棵树预测...")
            
            tree_predictions = self.predict_tree(X_array, tree)
            
            # 使用相同的动态学习率
            current_lr = self.learning_rate
            if i < 30:
                current_lr *= 1.3
            elif i < 60:
                current_lr *= 1.1
            elif i > 100:
                current_lr *= 0.7
                
            lr_decay = 0.99 ** (i // 10)
            current_lr *= lr_decay
                
            predictions_transformed += current_lr * tree_predictions
        
        # 逆变换
        predictions = self.inverse_target_transform(predictions_transformed, self.original_y_sample)
        
        print("预测完成！")
        return predictions
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def load_processed_data():
    """加载预处理后的数据"""
    print("加载预处理后的数据...")
    
    train_df = pd.read_csv('processed_train_data.csv')
    print(f"训练集加载完成，形状: {train_df.shape}")
    
    test_df = pd.read_csv('processed_test_data.csv')
    print(f"测试集加载完成，形状: {test_df.shape}")
    
    original_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    sale_ids = original_test['SaleID'].values
    print(f"获取到 {len(sale_ids)} 个SaleID")
    
    return train_df, test_df, sale_ids

def prepare_features(train_df, test_df):
    """准备特征数据"""
    print("准备特征数据...")
    
    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    X_test = test_df
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"训练目标形状: {y_train.shape}")
    print(f"测试特征形状: {X_test.shape}")
    
    common_features = list(set(X_train.columns) & set(X_test.columns))
    common_features.sort()
    
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"使用 {len(common_features)} 个共同特征")
    
    return X_train, y_train, X_test

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

def ultra_post_process(test_pred, original_train_price):
    """超精细后处理"""
    print("进行超精细后处理...")
    
    # 确保正价格
    test_pred = np.maximum(test_pred, 50)
    
    # 基于原始训练集分布进行精确校准
    original_percentiles = {}
    target_percentiles = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]
    
    for p in target_percentiles:
        original_percentiles[p] = np.percentile(original_train_price, p)
    
    # 获取当前预测的分位数
    current_percentiles = {}
    for p in target_percentiles:
        current_percentiles[p] = np.percentile(test_pred, p)
    
    # 分段校准
    test_pred_calibrated = test_pred.copy()
    
    # 对每个分位数区间进行校准
    for i, p in enumerate(target_percentiles[:-1]):
        p_next = target_percentiles[i + 1]
        
        # 当前区间的掩码
        mask = (test_pred >= current_percentiles[p]) & (test_pred < current_percentiles[p_next])
        
        if np.sum(mask) > 0:
            # 线性映射到目标区间
            current_min = current_percentiles[p]
            current_max = current_percentiles[p_next]
            target_min = original_percentiles[p]
            target_max = original_percentiles[p_next]
            
            # 避免除零
            if current_max > current_min:
                ratio = (test_pred[mask] - current_min) / (current_max - current_min)
                test_pred_calibrated[mask] = target_min + ratio * (target_max - target_min)
    
    # 处理最高分位数
    highest_mask = test_pred >= current_percentiles[99]
    if np.sum(highest_mask) > 0:
        # 最高1%的价格进行特殊处理
        max_reasonable = original_percentiles[99]
        test_pred_calibrated[highest_mask] = np.clip(
            test_pred_calibrated[highest_mask], 
            original_percentiles[95], 
            max_reasonable * 1.2  # 允许略微超出
        )
    
    print("超精细后处理完成")
    return test_pred_calibrated

def train_and_predict():
    """训练模型并预测"""
    print("="*60)
    print("超优化XGBoost建模开始 - 目标500分")
    print("="*60)
    
    train_df, test_df, sale_ids = load_processed_data()
    X_train, y_train, X_test = prepare_features(train_df, test_df)
    
    # 创建超优化XGBoost模型
    model = UltraOptimizedXGBoostModel(
        n_estimators=150,
        learning_rate=0.06,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 训练集预测和评估
    print("\n评估模型性能...")
    train_pred = model.predict(X_train)
    
    metrics = evaluate_model(y_train, train_pred)
    print("\n模型性能评估:")
    for metric, value in metrics.items():
        print(f"训练集 {metric}: {value:.4f}")
    
    # 特征重要性
    feature_importance = model.get_feature_importance()
    print("\n前15个重要特征:")
    print(feature_importance.head(15))
    
    # 测试集预测
    print("\n开始预测测试集...")
    test_pred = model.predict(X_test)
    
    return test_pred, sale_ids, model, feature_importance, y_train

def save_ultra_predictions(test_pred, sale_ids, original_train_price):
    """保存超优化预测结果"""
    print("\n保存超优化预测结果...")
    
    # 超精细后处理
    test_pred = ultra_post_process(test_pred, original_train_price)
    
    result_df = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_pred
    })
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ultra_optimized_result_{current_time}.csv"
    
    result_df.to_csv(filename, index=False)
    
    print(f"预测结果已保存至: {filename}")
    print(f"预测了 {len(result_df)} 条记录")
    
    # 详细统计
    print("\n超优化预测结果统计:")
    print(f"预测价格均值: {test_pred.mean():.2f}")
    print(f"预测价格中位数: {np.median(test_pred):.2f}")
    print(f"预测价格标准差: {test_pred.std():.2f}")
    print(f"预测价格范围: {test_pred.min():.2f} - {test_pred.max():.2f}")
    
    # 关键分位数
    key_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n关键分位数:")
    for p in key_percentiles:
        value = np.percentile(test_pred, p)
        print(f"{p:2d}%: {value:.2f}")
    
    return filename, result_df

def main():
    """主函数"""
    try:
        test_pred, sale_ids, model, feature_importance, y_train = train_and_predict()
        filename, result_df = save_ultra_predictions(test_pred, sale_ids, y_train)
        
        print("="*60)
        print("超优化XGBoost建模完成！目标：冲击500分！")
        print("="*60)
        
        return filename, result_df
        
    except Exception as e:
        print(f"建模过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()