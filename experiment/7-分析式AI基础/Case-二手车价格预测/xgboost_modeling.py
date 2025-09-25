#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost建模脚本
使用预处理后的数据训练XGBoost模型，并对测试集进行预测
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleXGBoostModel:
    """简化的XGBoost实现（基于梯度提升）"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
        """初始化XGBoost模型"""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = []
        self.feature_names = None
        self.feature_importances_ = None
        self.is_trained = False
        
        # 设置随机种子
        np.random.seed(random_state)
    
    def simple_tree_regressor(self, X, y, max_depth=6):
        """简单的回归树实现"""
        n_samples, n_features = X.shape
        
        if n_samples < 10 or max_depth <= 0:
            return np.mean(y)
        
        best_feature = None
        best_threshold = None
        best_mse = float('inf')
        
        # 随机选择部分特征进行分割
        n_features_subset = max(1, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.percentile(feature_values, [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # 计算MSE
                left_mse = np.var(left_y) if len(left_y) > 0 else 0
                right_mse = np.var(right_y) if len(right_y) > 0 else 0
                
                weighted_mse = (len(left_y) * left_mse + len(right_y) * right_mse) / len(y)
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            return np.mean(y)
        
        # 构建树节点
        feature_values = X[:, best_feature]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self.simple_tree_regressor(X[left_mask], y[left_mask], max_depth - 1)
        right_subtree = self.simple_tree_regressor(X[right_mask], y[right_mask], max_depth - 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def predict_tree(self, X, tree):
        """使用树进行预测"""
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
    
    def fit(self, X, y):
        """训练XGBoost模型"""
        print(f"开始训练XGBoost模型（{self.n_estimators}棵树）...")
        
        self.feature_names = X.columns.tolist()
        n_samples = len(X)
        
        # 初始化预测值（使用均值）
        predictions = np.full(n_samples, np.mean(y))
        self.models = []
        
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(len(self.feature_names))
        
        X_array = X.values
        y_array = y.values
        
        for i in range(self.n_estimators):
            if (i + 1) % 20 == 0:
                print(f"正在训练第 {i + 1}/{self.n_estimators} 棵树...")
            
            # 计算残差（梯度）
            residuals = y_array - predictions
            
            # 添加一些随机性（类似于随机梯度提升）
            sample_indices = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
            X_sample = X_array[sample_indices]
            residuals_sample = residuals[sample_indices]
            
            # 训练树来拟合残差
            tree = self.simple_tree_regressor(X_sample, residuals_sample, self.max_depth)
            
            # 使用当前树预测全部样本
            tree_predictions = self.predict_tree(X_array, tree)
            
            # 更新预测值
            predictions += self.learning_rate * tree_predictions
            
            # 保存树
            self.models.append(tree)
            
            # 简单的特征重要性估计
            if isinstance(tree, dict) and 'feature' in tree:
                self.feature_importances_[tree['feature']] += 1
        
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
        predictions = np.zeros(X_array.shape[0])
        
        for i, tree in enumerate(self.models):
            if (i + 1) % 20 == 0:
                print(f"正在使用第 {i + 1}/{len(self.models)} 棵树预测...")
            
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
    print("XGBoost建模开始")
    print("="*50)
    
    # 加载数据
    train_df, test_df, sale_ids = load_processed_data()
    
    # 准备特征
    X_train, y_train, X_test = prepare_features(train_df, test_df)
    
    # 尝试使用真正的XGBoost
    try:
        import xgboost as xgb
        print("使用真正的XGBoost库...")
        
        # 创建XGBoost模型
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # 训练模型
        print("开始训练XGBoost模型...")
        model.fit(X_train, y_train)
        print("模型训练完成！")
        
        # 训练集预测
        train_pred = model.predict(X_train)
        
        # 评估模型
        metrics = evaluate_model(y_train, train_pred)
        print("\n模型性能评估:")
        for metric, value in metrics.items():
            print(f"训练集 {metric}: {value:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n前10个重要特征:")
        print(feature_importance.head(10))
        
        # 测试集预测
        print("\n开始预测测试集...")
        test_pred = model.predict(X_test)
        print("测试集预测完成！")
        
        return test_pred, sale_ids, model, feature_importance
        
    except ImportError:
        print("XGBoost库不可用，使用简化的XGBoost实现...")
        
        # 使用简化的XGBoost实现
        model = SimpleXGBoostModel(
            n_estimators=50,  # 减少树的数量以提高速度
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 训练集预测
        print("\n评估模型性能...")
        train_pred = model.predict(X_train)
        
        # 评估模型
        metrics = evaluate_model(y_train, train_pred)
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
        print("XGBoost建模完成！")
        print("="*50)
        
        return filename, result_df
        
    except Exception as e:
        print(f"建模过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()