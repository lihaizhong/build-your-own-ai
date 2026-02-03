"""
模型验证工具模块
提供模型评估、交叉验证和结果分析功能
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "模型"
) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        
    Returns:
        评估指标字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name} 评估结果")
    print(f"{'='*60}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"{'='*60}")
    
    logger.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return metrics


def cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str = 'neg_mean_absolute_error'
) -> Dict[str, Any]:
    """
    K折交叉验证
    
    Args:
        model: 模型对象
        X: 特征数据
        y: 目标变量
        n_splits: 折数
        random_state: 随机种子
        scoring: 评分指标
        
    Returns:
        交叉验证结果字典
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 执行交叉验证
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    
    # 转换为正数（因为使用的是neg_mean_absolute_error）
    cv_scores = -cv_scores
    
    result = {
        'mean_mae': cv_scores.mean(),
        'std_mae': cv_scores.std(),
        'min_mae': cv_scores.min(),
        'max_mae': cv_scores.max(),
        'all_scores': cv_scores
    }
    
    print(f"\n{'='*60}")
    print(f"交叉验证结果 ({n_splits}折)")
    print(f"{'='*60}")
    print(f"平均 MAE: {result['mean_mae']:.4f}")
    print(f"标准差:   {result['std_mae']:.4f}")
    print(f"最小值:   {result['min_mae']:.4f}")
    print(f"最大值:   {result['max_mae']:.4f}")
    print(f"各折得分: {result['all_scores']}")
    print(f"{'='*60}")
    
    logger.info(f"交叉验证 - 平均MAE: {result['mean_mae']:.4f} (±{result['std_mae']:.4f})")
    
    return result


def compare_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    比较多个模型的性能
    
    Args:
        models: 模型字典 {名称: 模型对象}
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征
        y_test: 测试目标
        
    Returns:
        比较结果数据框
    """
    results = []
    
    for name, model in models.items():
        logger.info(f"训练模型: {name}")
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # 评估
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        results.append({
            'Model': name,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Gap': test_mae - train_mae
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Test_MAE')
    
    print(f"\n{'='*60}")
    print("模型性能比较")
    print(f"{'='*60}")
    print(result_df.to_string(index=False))
    print(f"{'='*60}")
    
    return result_df


def feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    获取特征重要性
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        top_n: 显示前N个特征
        
    Returns:
        特征重要性数据框
    """
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        logger.warning("模型不支持特征重要性分析")
        return None # type: ignore
    
    # 创建数据框
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # 排序
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df = importance_df.head(top_n)
    
    print(f"\n{'='*60}")
    print(f"前 {top_n} 重要特征")
    print(f"{'='*60}")
    print(importance_df.to_string(index=False))
    print(f"{'='*60}")
    
    return importance_df


def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    """
    残差分析
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    """
    residuals = y_true - y_pred
    
    print(f"\n{'='*60}")
    print("残差分析")
    print(f"{'='*60}")
    print(f"残差均值: {residuals.mean():.4f}")
    print(f"残差标准差: {residuals.std():.4f}")
    print(f"残差最小值: {residuals.min():.4f}")
    print(f"残差最大值: {residuals.max():.4f}")
    print(f"{'='*60}")
    
    # 检查残差是否接近正态分布
    from scipy import stats
    _, p_value = stats.normaltest(residuals)
    
    print(f"残差正态性检验 p值: {p_value:.4f}")
    if p_value > 0.05:
        print("残差近似服从正态分布 ✓")
    else:
        print("残差不服从正态分布 ✗")
