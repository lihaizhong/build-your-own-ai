#!/usr/bin/env python3
"""
预测模块
使用训练好的模型进行房价预测
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class HousePricePredictor:
    """房价预测器类"""
    
    def __init__(self, model_dir='../model/'):
        """
        初始化预测器
        
        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.best_model_name = None
    
    def load_models(self):
        """加载所有训练好的模型"""
        try:
            # 加载各种模型
            model_files = {
                'linear_regression': 'linear_regression.pkl',
                'ridge': 'ridge.pkl',
                'lasso': 'lasso.pkl',
                'random_forest': 'random_forest.pkl',
                'gradient_boosting': 'gradient_boosting.pkl',
                'svr': 'svr.pkl',
                'neural_network': 'neural_network.pkl'
            }
            
            for model_name, filename in model_files.items():
                try:
                    self.models[model_name] = joblib.load(f"{self.model_dir}{filename}")
                    print(f"成功加载模型: {model_name}")
                except FileNotFoundError:
                    print(f"模型文件不存在: {filename}")
            
            # 加载标准化器
            try:
                self.scalers['standard'] = joblib.load(f"{self.model_dir}standard_scaler.pkl")
                print("成功加载标准化器")
            except FileNotFoundError:
                print("标准化器文件不存在")
                
        except Exception as e:
            print(f"加载模型时出错: {e}")
    
    def load_best_model(self):
        """加载最佳模型"""
        try:
            self.best_model_name = joblib.load(f"{self.model_dir}best_model_name.pkl")
            self.models['best'] = joblib.load(f"{self.model_dir}best_model.pkl")
            print(f"成功加载最佳模型: {self.best_model_name}")
        except FileNotFoundError:
            print("最佳模型文件不存在")
    
    def preprocess_input(self, X, use_scaler=True):
        """
        预处理输入数据
        
        Args:
            X: 输入特征数据
            use_scaler: 是否使用标准化
            
        Returns:
            np.array: 预处理后的特征
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if use_scaler and 'standard' in self.scalers:
            X = self.scalers['standard'].transform(X)
        
        return X
    
    def predict_single(self, features, model_name=None, use_best=True):
        """
        单个样本预测
        
        Args:
            features: 特征数组 (1D array)
            model_name: 指定模型名称，如果为None则使用最佳模型
            use_best: 是否使用最佳模型
            
        Returns:
            float: 预测的房价
        """
        # 转换为2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 选择模型
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在")
            model = self.models[model_name]
        elif use_best and 'best' in self.models:
            model = self.models['best']
        else:
            raise ValueError("没有可用的模型进行预测")
        
        # 预处理和预测
        features_processed = self.preprocess_input(features)
        prediction = model.predict(features_processed)[0]
        
        return prediction
    
    def predict_batch(self, X, model_name=None, use_best=True):
        """
        批量预测
        
        Args:
            X: 特征数据 (2D array 或 DataFrame)
            model_name: 指定模型名称
            use_best: 是否使用最佳模型
            
        Returns:
            np.array: 预测结果数组
        """
        # 选择模型
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在")
            model = self.models[model_name]
        elif use_best and 'best' in self.models:
            model = self.models['best']
        else:
            raise ValueError("没有可用的模型进行预测")
        
        # 预处理和预测
        features_processed = self.preprocess_input(X)
        predictions = model.predict(features_processed)
        
        return predictions
    
    def predict_with_confidence(self, features, model_name=None, n_bootstrap=100):
        """
        带置信区间的预测
        
        Args:
            features: 特征数组
            model_name: 指定模型名称
            n_bootstrap: Bootstrap采样次数
            
        Returns:
            tuple: (预测值, 置信区间下界, 置信区间上界)
        """
        # TODO: 实现带置信区间的预测
        pass
    
    def compare_model_predictions(self, features, save_path=None):
        """
        比较不同模型的预测结果
        
        Args:
            features: 特征数组
            save_path: 保存路径
            
        Returns:
            pd.DataFrame: 不同模型的预测结果
        """
        predictions = {}
        
        for model_name in self.models:
            try:
                pred = self.predict_single(features, model_name=model_name)
                predictions[model_name] = pred
            except Exception as e:
                print(f"模型 {model_name} 预测失败: {e}")
        
        df_predictions = pd.DataFrame(list(predictions.items()), 
                                    columns=['Model', 'Prediction'])
        
        if save_path:
            df_predictions.to_csv(save_path, index=False)
        
        return df_predictions
    
    def plot_prediction_uncertainty(self, X, y_true=None, save_path=None):
        """
        绘制预测不确定性
        
        Args:
            X: 特征数据
            y_true: 真实值（可选）
            save_path: 保存路径
        """
        # TODO: 实现预测不确定性可视化
        pass
    
    def save_predictions(self, predictions, save_path='../prediction_result/predictions.csv'):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            save_path: 保存路径
        """
        if isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=['Predicted_Price'])
        
        predictions.to_csv(save_path, index=False)
        print(f"预测结果已保存到: {save_path}")

def create_sample_input():
    """创建示例输入数据"""
    # 波士顿房价数据集的示例特征值
    sample_features = np.array([
        0.00632,  # CRIM
        18.0,     # ZN
        2.31,     # INDUS
        0.0,      # CHAS
        0.538,    # NOX
        6.575,    # RM
        65.2,     # AGE
        4.0900,   # DIS
        1.0,      # RAD
        296.0,    # TAX
        15.3,     # PTRATIO
        396.90,   # B
        4.98      # LSTAT
    ])
    
    return sample_features

def main():
    """
    主函数 - 执行房价预测流程
    """
    predictor = HousePricePredictor()
    
    # 加载模型
    predictor.load_models()
    predictor.load_best_model()
    
    # 创建示例输入
    sample_input = create_sample_input()
    print("示例输入特征:")
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    for name, value in zip(feature_names, sample_input):
        print(f"  {name}: {value}")
    
    # 使用最佳模型预测
    try:
        prediction = predictor.predict_single(sample_input, use_best=True)
        print(f"\n最佳模型预测房价: ${prediction*1000:.2f}k")
    except Exception as e:
        print(f"预测失败: {e}")
    
    # 比较所有模型
    try:
        model_comparison = predictor.compare_model_predictions(sample_input)
        print("\n所有模型预测结果:")
        print(model_comparison)
    except Exception as e:
        print(f"模型比较失败: {e}")

if __name__ == "__main__":
    main()
