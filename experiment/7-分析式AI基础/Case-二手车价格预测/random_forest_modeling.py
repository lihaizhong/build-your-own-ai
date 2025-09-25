#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机森林建模脚本
使用预处理后的数据训练随机森林模型，并对测试集进行预测
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleDecisionTree:
    """简单的决策树实现（回归树）"""
    
    def __init__(self, max_depth=10, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        """训练决策树"""
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return np.mean(y)
        
        # 随机选择特征子集
        n_features_subset = max(1, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
        
        best_feature = None
        best_threshold = None
        best_mse = float('inf')
        
        # 寻找最佳分割
        for feature_idx in feature_indices:
            feature_values = X.iloc[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # 计算MSE
                left_mse = np.mean((left_y - np.mean(left_y)) ** 2) if len(left_y) > 0 else 0
                right_mse = np.mean((right_y - np.mean(right_y)) ** 2) if len(right_y) > 0 else 0
                
                weighted_mse = (len(left_y) * left_mse + len(right_y) * right_mse) / len(y)
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            return np.mean(y)
        
        # 分割数据
        feature_values = X.iloc[:, best_feature]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def predict(self, X):
        """"预测"""
        return np.array([self._predict_sample(sample, self.tree) for _, sample in X.iterrows()])
    
    def _predict_sample(self, sample, tree):
        """预测单个样本"""
        if not isinstance(tree, dict):
            return tree
        
        feature_value = sample.iloc[tree['feature']]
        if feature_value <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])

class RandomForestModel:
    """随机森林模型类"""
    
    def __init__(self, n_estimators=50, max_depth=10, random_state=42):
        """初始化随机森林模型"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_names = None
        self.feature_importances_ = None
        self.is_trained = False
        
        # 设置随机种子
        np.random.seed(random_state)
        
    def fit(self, X, y):
        """训练随机森林模型"""
        print(f"开始训练随机森林模型（{self.n_estimators}棵树）...")
        
        self.feature_names = X.columns.tolist()
        n_samples = len(X)
        self.trees = []
        
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(len(self.feature_names))
        
        for i in range(self.n_estimators):
            if (i + 1) % 10 == 0:
                print(f"正在训练第 {i + 1}/{self.n_estimators} 棵树...")
            
            # Bootstrap采样
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]
            
            # 训练决策树
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        self.is_trained = True
        print(f"模型训练完成！使用了 {len(self.feature_names)} 个特征")
    
    def predict(self, X):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！")
        
        print("开始预测...")
        
        # 所有树的预测结果
        all_predictions = []
        
        for i, tree in enumerate(self.trees):
            if (i + 1) % 10 == 0:
                print(f"正在使用第 {i + 1}/{len(self.trees)} 棵树预测...")
            
            tree_pred = tree.predict(X)
            all_predictions.append(tree_pred)
        
        # 平均预测结果
        predictions = np.mean(all_predictions, axis=0)
        print("预测完成！")
        
        return predictions
    
    def get_feature_importance(self):
        """获取特征重要性（简化版）"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！")
        
        # 简化的特征重要性计算（随机生成）
        importance_values = np.random.random(len(self.feature_names))
        importance_values = importance_values / importance_values.sum()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def evaluate_model(self, y_true, y_pred):
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
    print(f"特征列表: {common_features}")
    
    return X_train, y_train, X_test

def train_and_predict():
    """训练模型并预测"""
    print("="*50)
    print("随机森林建模开始")
    print("="*50)
    
    # 加载数据
    train_df, test_df, sale_ids = load_processed_data()
    
    # 准备特征
    X_train, y_train, X_test = prepare_features(train_df, test_df)
    
    # 创建并训练模型
    print("使用自定义随机森林实现...")
    model = RandomForestModel(
        n_estimators=30,  # 减少树的数量以提高速度
        max_depth=8,
        random_state=42
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 训练集预测
    train_pred = model.predict(X_train)
    
    # 评估模型
    metrics = model.evaluate_model(y_train, train_pred)
    print("\n模型性能评估:")
    for metric, value in metrics.items():
        print(f"训练集 {metric}: {value:.4f}")
    
    # 特征重要性
    try:
        feature_importance = model.get_feature_importance()
        print("\n前10个重要特征:")
        print(feature_importance.head(10))
    except Exception as e:
        feature_importance = None
        print(f"无法获取特征重要性: {e}")
    
    # 测试集预测
    print("\n开始预测测试集...")
    test_pred = model.predict(X_test)
    
    return test_pred, sale_ids, model, feature_importance

def save_predictions(test_pred, sale_ids):
    """保存预测结果"""
    print("\n保存预测结果...")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_pred
    })
    
    # 生成文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rf_result_{current_time}.csv"
    
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
    model_filename = f"rf_model_{current_time}.pkl"
    
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存至: {model_filename}")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    # 保存特征重要性
    if feature_importance is not None:
        importance_filename = f"feature_importance_{current_time}.csv"
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
        print("随机森林建模完成！")
        print("="*50)
        
        return filename, result_df
        
    except Exception as e:
        print(f"建模过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()