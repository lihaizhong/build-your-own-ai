#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的XGBoost建模脚本 - 针对价格分布优化
重点改进低价和高价车的预测准确性
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedXGBoostModel:
    """优化的XGBoost实现，重点改进价格分布拟合"""
    
    def __init__(self, n_estimators=120, learning_rate=0.08, max_depth=6, subsample=0.9, random_state=42):
        """初始化优化的XGBoost模型"""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.base_prediction = None
        self.trees = []
        self.feature_names = None
        self.feature_importances_ = None
        self.is_trained = False
        
        # 设置随机种子
        np.random.seed(random_state)
    
    def build_tree(self, X, residuals, depth=0):
        """构建优化的回归树"""
        if depth >= self.max_depth or len(residuals) < 8:  # 减少最小样本数以获得更细粒度的分割
            return np.mean(residuals)
        
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # 增加特征采样比例
        n_features_subset = max(5, int(np.sqrt(n_features) * 1.5))
        feature_indices = np.random.choice(n_features, min(n_features_subset, n_features), replace=False)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) < 2:
                continue
            
            # 增加分割点密度，特别关注极值区域
            if len(unique_values) > 10:
                thresholds = np.percentile(unique_values, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            else:
                thresholds = unique_values[:-1]
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 3 or np.sum(right_mask) < 3:  # 更激进的分割
                    continue
                
                # 使用加权方差作为分割标准，对极值给予更高权重
                left_residuals = residuals[left_mask]
                right_residuals = residuals[right_mask]
                
                # 计算增益时考虑样本权重
                total_var = np.var(residuals)
                left_var = np.var(left_residuals) if len(left_residuals) > 1 else 0
                right_var = np.var(right_residuals) if len(right_residuals) > 1 else 0
                
                # 加权方差计算
                weighted_var = (len(left_residuals) * left_var + len(right_residuals) * right_var) / len(residuals)
                gain = total_var - weighted_var
                
                # 对极值区域给予额外奖励
                extreme_bonus = 0
                if np.mean(np.abs(left_residuals)) > np.std(residuals) or np.mean(np.abs(right_residuals)) > np.std(residuals):
                    extreme_bonus = gain * 0.1  # 10%的额外奖励
                
                total_gain = gain + extreme_bonus
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None or best_gain <= 0:
            return np.mean(residuals)
        
        # 分割数据
        feature_values = X[:, best_feature]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_tree = self.build_tree(X[left_mask], residuals[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], residuals[right_mask], depth + 1)
        
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
            
            # 递归计算子树的重要性
            self.calculate_feature_importance(tree['left'], importances)
            self.calculate_feature_importance(tree['right'], importances)
    
    def apply_log_transform(self, y):
        """应用对数变换"""
        # 对价格进行log1p变换以处理0值
        return np.log1p(y)
    
    def inverse_log_transform(self, y_log):
        """逆对数变换"""
        return np.expm1(y_log)
    
    def fit(self, X, y):
        """训练优化的XGBoost模型"""
        print(f"开始训练优化XGBoost模型（{self.n_estimators}棵树）...")
        
        self.feature_names = X.columns.tolist()
        n_samples, n_features = X.shape
        
        # 对目标变量进行对数变换
        y_log = self.apply_log_transform(y)
        
        # 初始化预测值为对数变换后的均值
        self.base_prediction = np.mean(y_log)
        predictions = np.full(n_samples, self.base_prediction)
        
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(n_features)
        
        X_array = X.values
        y_log_array = y_log.values if hasattr(y_log, 'values') else y_log
        
        self.trees = []
        
        for i in range(self.n_estimators):
            if (i + 1) % 15 == 0:
                print(f"正在训练第 {i + 1}/{self.n_estimators} 棵树...")
            
            # 计算残差
            residuals = y_log_array - predictions
            
            # 对于极值样本给予更高的采样权重
            abs_residuals = np.abs(residuals)
            high_error_threshold = np.percentile(abs_residuals, 75)
            weights = np.where(abs_residuals > high_error_threshold, 1.5, 1.0)
            
            # 加权子采样
            n_subsample = int(self.subsample * n_samples)
            sample_probs = weights / np.sum(weights)
            subsample_indices = np.random.choice(n_samples, n_subsample, replace=True, p=sample_probs)
            
            X_subsample = X_array[subsample_indices]
            residuals_subsample = residuals[subsample_indices]
            
            # 构建树
            tree = self.build_tree(X_subsample, residuals_subsample)
            
            # 计算所有样本的树预测
            tree_predictions = self.predict_tree(X_array, tree)
            
            # 使用自适应学习率
            current_lr = self.learning_rate
            if i < 20:  # 前20棵树使用较大学习率
                current_lr *= 1.2
            elif i > 80:  # 后期使用较小学习率精调
                current_lr *= 0.8
            
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
        
        X_array = X.values
        predictions_log = np.full(X_array.shape[0], self.base_prediction)
        
        for i, tree in enumerate(self.trees):
            if (i + 1) % 15 == 0:
                print(f"正在使用第 {i + 1}/{len(self.trees)} 棵树预测...")
            
            tree_predictions = self.predict_tree(X_array, tree)
            
            # 使用相同的自适应学习率
            current_lr = self.learning_rate
            if i < 20:
                current_lr *= 1.2
            elif i > 80:
                current_lr *= 0.8
                
            predictions_log += current_lr * tree_predictions
        
        # 逆对数变换
        predictions = self.inverse_log_transform(predictions_log)
        
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

def train_and_predict():
    """训练模型并预测"""
    print("="*50)
    print("优化XGBoost建模开始")
    print("="*50)
    
    train_df, test_df, sale_ids = load_processed_data()
    X_train, y_train, X_test = prepare_features(train_df, test_df)
    
    # 创建优化的XGBoost模型
    model = OptimizedXGBoostModel(
        n_estimators=120,  # 增加树的数量
        learning_rate=0.08,  # 适中的学习率
        max_depth=6,  # 增加树的深度
        subsample=0.9,  # 提高采样率
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
    print("\n前10个重要特征:")
    print(feature_importance.head(10))
    
    # 测试集预测
    print("\n开始预测测试集...")
    test_pred = model.predict(X_test)
    
    return test_pred, sale_ids, model, feature_importance

def post_process_predictions(test_pred):
    """后处理预测结果以更好匹配原始分布"""
    print("后处理预测结果...")
    
    # 确保价格为正数
    test_pred = np.maximum(test_pred, 50)
    
    # 根据原始分布调整极值
    # 原始数据：1%: 150, 5%: 400, 95%: 19970, 99%: 34950
    
    # 调整过低的预测
    low_threshold = np.percentile(test_pred, 1)
    if low_threshold < 150:
        low_mask = test_pred <= low_threshold
        test_pred[low_mask] = np.random.uniform(50, 400, np.sum(low_mask))
    
    # 调整过高的预测（但保持一些极值）
    high_threshold = np.percentile(test_pred, 99)
    if high_threshold > 50000:
        high_mask = test_pred >= high_threshold
        # 只调整最极端的值，保持合理的高价车
        extreme_mask = test_pred >= np.percentile(test_pred, 99.5)
        test_pred[extreme_mask] = np.random.uniform(20000, 40000, np.sum(extreme_mask))
    
    print("后处理完成")
    return test_pred

def save_predictions(test_pred, sale_ids):
    """保存预测结果"""
    print("\n保存预测结果...")
    
    # 后处理预测结果
    test_pred = post_process_predictions(test_pred)
    
    result_df = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_pred
    })
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimized_xgb_result_{current_time}.csv"
    
    result_df.to_csv(filename, index=False)
    
    print(f"预测结果已保存至: {filename}")
    print(f"预测了 {len(result_df)} 条记录")
    
    # 显示预测结果统计
    print("\n预测结果统计:")
    print(f"预测价格均值: {test_pred.mean():.2f}")
    print(f"预测价格中位数: {np.median(test_pred):.2f}")
    print(f"预测价格标准差: {test_pred.std():.2f}")
    print(f"预测价格范围: {test_pred.min():.2f} - {test_pred.max():.2f}")
    
    # 显示分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n预测价格分位数:")
    for p in percentiles:
        value = np.percentile(test_pred, p)
        print(f"{p:2d}%: {value:.2f}")
    
    return filename, result_df

def main():
    """主函数"""
    try:
        test_pred, sale_ids, model, feature_importance = train_and_predict()
        filename, result_df = save_predictions(test_pred, sale_ids)
        
        print("="*50)
        print("优化XGBoost建模完成！")
        print("="*50)
        
        return filename, result_df
        
    except Exception as e:
        print(f"建模过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()