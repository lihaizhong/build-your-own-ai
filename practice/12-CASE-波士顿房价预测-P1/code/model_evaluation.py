#!/usr/bin/env python3
"""
模型评估模块
负责模型性能评估、可视化和结果分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            
        Returns:
            dict: 评估指标
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'Explained_Variance': explained_variance_score(y_true, y_pred)
        }
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, save_path=None):
        """
        绘制预测值vs实际值散点图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('实际房价')
        plt.ylabel('预测房价')
        plt.title(f'{model_name} - 预测值 vs 实际值')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, model_name, save_path=None):
        """
        绘制残差图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            save_path: 保存路径
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 残差散点图
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('预测值')
        axes[0].set_ylabel('残差')
        axes[0].set_title(f'{model_name} - 残差散点图')
        axes[0].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('残差')
        axes[1].set_ylabel('频次')
        axes[1].set_title(f'{model_name} - 残差分布')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curve(self, model, X, y, model_name, save_path=None):
        """
        绘制学习曲线
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 目标数据
            model_name: 模型名称
            save_path: 保存路径
        """
        # TODO: 实现学习曲线绘制
        pass
    
    def compare_models(self, save_path=None):
        """
        对比所有模型的性能
        
        Args:
            save_path: 保存路径
        """
        if not self.evaluation_results:
            print("没有评估结果可比较")
            return
        
        df_results = pd.DataFrame(self.evaluation_results).T
        
        # 创建对比表格
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MAE对比
        axes[0, 0].bar(df_results.index, df_results['MAE'])
        axes[0, 0].set_title('平均绝对误差 (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE对比
        axes[0, 1].bar(df_results.index, df_results['RMSE'])
        axes[0, 1].set_title('均方根误差 (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R²对比
        axes[1, 0].bar(df_results.index, df_results['R2'])
        axes[1, 0].set_title('决定系数 (R²)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 解释方差对比
        axes[1, 1].bar(df_results.index, df_results['Explained_Variance'])
        axes[1, 1].set_title('解释方差')
        axes[1, 1].set_ylabel('Explained Variance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_results
    
    def save_evaluation_results(self, save_path='../prediction_result/evaluation_metrics.csv'):
        """
        保存评估结果
        
        Args:
            save_path: 保存路径
        """
        if self.evaluation_results:
            df_results = pd.DataFrame(self.evaluation_results).T
            df_results.to_csv(save_path)
            print(f"评估结果已保存到: {save_path}")
        else:
            print("没有评估结果可保存")
    
    def load_evaluation_results(self, load_path='../prediction_result/evaluation_metrics.csv'):
        """
        加载评估结果
        
        Args:
            load_path: 加载路径
        """
        try:
            self.evaluation_results = pd.read_csv(load_path, index_col=0).to_dict('index')
            print(f"评估结果已从 {load_path} 加载")
        except FileNotFoundError:
            print(f"文件 {load_path} 不存在")

def main():
    """
    主函数 - 执行模型评估流程
    """
    evaluator = ModelEvaluator()
    
    # TODO: 加载训练好的模型和测试数据
    # models = load_trained_models()
    # X_test, y_test = load_test_data()
    
    # 评估每个模型
    # for model_name, model in models.items():
    #     y_pred = model.predict(X_test)
    #     metrics = evaluator.calculate_metrics(y_test, y_pred, model_name)
    #     print(f"{model_name} 评估结果:")
    #     for metric, value in metrics.items():
    #         print(f"  {metric}: {value:.4f}")
    
    # 可视化结果
    # evaluator.compare_models(save_path='../prediction_result/model_comparison.png')
    
    # 保存评估结果
    # evaluator.save_evaluation_results()

if __name__ == "__main__":
    main()