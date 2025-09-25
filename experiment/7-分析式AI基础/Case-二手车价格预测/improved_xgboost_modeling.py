#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的XGBoost建模脚本
使用更稳定的梯度提升实现
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedXGBoostModel:
    """改进的XGBoost实现"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42):
        """初始化XGBoost模型"""
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
        """构建回归树"""
        if depth >= self.max_depth or len(residuals) < 10:
            return np.mean(residuals)
        
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # 随机选择特征子集
        n_features_subset = max(3, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) < 2:
                continue
            
            # 选择分割点
            thresholds = np.percentile(unique_values, [20, 40, 60, 80])
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                
                # 计算增益
                left_residuals = residuals[left_mask]
                right_residuals = residuals[right_mask]
                
                total_var = np.var(residuals)
                left_var = np.var(left_residuals) if len(left_residuals) > 1 else 0
                right_var = np.var(right_residuals) if len(right_residuals) > 1 else 0
                
                weighted_var = (len(left_residuals) * left_var + len(right_residuals) * right_var) / len(residuals)
                gain = total_var - weighted_var
                
                if gain > best_gain:
                    best_gain = gain
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
    
    def fit(self, X, y):
        """训练XGBoost模型"""
        print(f"开始训练改进XGBoost模型（{self.n_estimators}棵树）...")
        
        self.feature_names = X.columns.tolist()
        n_samples, n_features = X.shape
        
        # 初始化预测值为目标变量的均值
        self.base_prediction = np.mean(y)
        predictions = np.full(n_samples, self.base_prediction)
        
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(n_features)
        
        X_array = X.values
        y_array = y.values
        
        self.trees = []
        
        for i in range(self.n_estimators):
            if (i + 1) % 10 == 0:
                print(f"正在训练第 {i + 1}/{self.n_estimators} 棵树...")
            
            # 计算残差
            residuals = y_array - predictions
            
            # 子采样
            n_subsample = int(self.subsample * n_samples)
            subsample_indices = np.random.choice(n_samples, n_subsample, replace=False)
            X_subsample = X_array[subsample_indices]
            residuals_subsample = residuals[subsample_indices]
            
            # 构建树
            tree = self.build_tree(X_subsample, residuals_subsample)
            
            # 计算所有样本的树预测
            tree_predictions = self.predict_tree(X_array, tree)
            
            # 更新预测值（使用学习率）
            predictions += self.learning_rate * tree_predictions
            
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
        predictions = np.full(X_array.shape[0], self.base_prediction)
        
        for i, tree in enumerate(self.trees):
            if (i + 1) % 10 == 0:
                print(f"正在使用第 {i + 1}/{len(self.trees)} 棵树预测...")
            
            tree_predictions = self.predict_tree(X_array, tree)
            predictions += self.learning_rate * tree_predictions
        
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
    
    # 加载训练集
    train_df = pd.read_csv('processed_train_data.csv')
    print(f"训练集加载完成，形状: {train_df.shape}")
    
    # 加载测试集
    test_df = pd.read_csv('processed_test_data.csv')
    print(f"测试集加载完成，形状: {test_df.shape}")
    
    # 加载原始测试集以获取SaleID
    original_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    sale_ids = original_test['SaleID'].values
    print(f"获取到 {len(sale_ids)} 个SaleID")
    
    return train_df, test_df, sale_ids

def prepare_features(train_df, test_df):
    """准备特征数据"""
    print("准备特征数据...")
    
    # 分离特征和目标变量
    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    X_test = test_df
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"训练目标形状: {y_train.shape}")
    print(f"测试特征形状: {X_test.shape}")
    
    # 确保训练集和测试集有相同的特征
    common_features = list(set(X_train.columns) & set(X_test.columns))
    common_features.sort()  # 保持顺序一致
    
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"使用 {len(common_features)} 个共同特征")
    
    return X_train, y_train, X_test

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    # 计算RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 计算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 计算R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def train_and_predict():
    """训练模型并预测"""
    print("="*50)
    print("改进XGBoost建模开始")
    print("="*50)
    
    # 加载数据
    train_df, test_df, sale_ids = load_processed_data()
    
    # 准备特征
    X_train, y_train, X_test = prepare_features(train_df, test_df)
    
    # 创建改进的XGBoost模型
    model = ImprovedXGBoostModel(
        n_estimators=80,
        learning_rate=0.05,  # 降低学习率以提高稳定性
        max_depth=4,         # 降低树深度防止过拟合
        subsample=0.8,
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

def fix_predictions(test_pred):
    """修正预测结果，确保价格合理"""
    print("修正预测结果...")
    
    # 检查负价格
    negative_count = (test_pred < 0).sum()
    if negative_count > 0:
        print(f"发现 {negative_count} 个负价格，正在修正...")
        test_pred = np.maximum(test_pred, 100)  # 最小价格100元
    
    # 限制价格范围
    test_pred = np.clip(test_pred, 100, 100000)  # 价格范围100-100000元
    
    print("预测结果修正完成")
    return test_pred

def save_predictions(test_pred, sale_ids):
    """保存预测结果"""
    print("\n保存预测结果...")
    
    # 修正预测结果
    test_pred = fix_predictions(test_pred)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_pred
    })
    
    # 生成文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xgb_result_{current_time}.csv"
    
    # 保存结果
    result_df.to_csv(filename, index=False)
    
    print(f"预测结果已保存至: {filename}")
    print(f"预测了 {len(result_df)} 条记录")
    
    # 显示预测结果统计
    print("\n预测结果统计:")
    print(f"预测价格均值: {test_pred.mean():.2f}")
    print(f"预测价格中位数: {np.median(test_pred):.2f}")
    print(f"预测价格标准差: {test_pred.std():.2f}")
    print(f"预测价格范围: {test_pred.min():.2f} - {test_pred.max():.2f}")
    
    return filename, result_df

def save_model_and_results(model, feature_importance):
    """保存模型和结果"""
    print("\n保存模型和特征重要性...")
    
    # 保存模型
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgb_model_{current_time}.pkl"
    
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存至: {model_filename}")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    # 保存特征重要性
    if feature_importance is not None:
        importance_filename = f"xgb_feature_importance_{current_time}.csv"
        feature_importance.to_csv(importance_filename, index=False)
        print(f"特征重要性已保存至: {importance_filename}")

def main():
    """主函数"""
    try:
        # 训练模型并预测
        test_pred, sale_ids, model, feature_importance = train_and_predict()
        
        # 保存预测结果
        filename, result_df = save_predictions(test_pred, sale_ids)
        
        # 保存模型和其他结果
        save_model_and_results(model, feature_importance)
        
        print("="*50)
        print("改进XGBoost建模完成！")
        print("="*50)
        
        return filename, result_df
        
    except Exception as e:
        print(f"建模过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()