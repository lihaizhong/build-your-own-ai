"""
数据分析工具模块
提供数据加载、预处理和基础分析功能
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger
from ...shared import get_project_path


def load_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载训练和测试数据
    
    Args:
        train_path: 训练数据路径，默认使用项目默认路径
        test_path: 测试数据路径，默认使用项目默认路径
        
    Returns:
        (训练数据, 测试数据)
    """
    if train_path is None:
        train_path = get_project_path("data", "used_car_train_20200313.csv") # type: ignore
    if test_path is None:
        test_path = get_project_path("data", "used_car_testB_20200421.csv") # type: ignore
    
    logger.info(f"加载训练数据: {train_path}")
    train_df = pd.read_csv(train_path, sep=' ') # type: ignore
    
    logger.info(f"加载测试数据: {test_path}")
    test_df = pd.read_csv(test_path, sep=' ') # type: ignore
    
    logger.info(f"训练数据形状: {train_df.shape}")
    logger.info(f"测试数据形状: {test_df.shape}")
    
    return train_df, test_df


def basic_data_info(df: pd.DataFrame, name: str = "数据集") -> None:
    """
    打印数据集基本信息
    
    Args:
        df: 数据框
        name: 数据集名称
    """
    print(f"\n{'='*60}")
    print(f"{name}基本信息")
    print(f"{'='*60}")
    print(f"形状: {df.shape}")
    print(f"列数: {len(df.columns)}")
    print(f"缺失值数量: {df.isnull().sum().sum()}")
    
    print(f"\n数据类型:")
    print(df.dtypes)
    
    print(f"\n缺失值统计:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("无缺失值")
    
    print(f"\n数值特征统计:")
    print(df.describe())


def check_outliers(df: pd.DataFrame, column: str, n_std: int = 3) -> pd.DataFrame:
    """
    检查异常值（基于标准差）
    
    Args:
        df: 数据框
        column: 列名
        n_std: 标准差倍数
        
    Returns:
        包含异常值的行
    """
    mean = df[column].mean()
    std = df[column].std()
    lower = mean - n_std * std
    upper = mean + n_std * std
    
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    
    logger.info(f"{column} 异常值数量: {len(outliers)}")
    return outliers


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化内存使用
    
    Args:
        df: 数据框
        
    Returns:
        优化后的数据框
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"内存优化: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% 减少)")
    
    return df


def save_submission(
    predictions: pd.Series,
    sample_path: Optional[str] = None,
    output_path: Optional[str] = None,
    suffix: str = ""
) -> pd.DataFrame:
    """
    保存预测结果到提交格式
    
    Args:
        predictions: 预测结果
        sample_path: 样本提交文件路径
        output_path: 输出路径
        suffix: 文件名后缀
        
    Returns:
        提交数据框
    """
    if sample_path is None:
        sample_path = get_project_path("data", "used_car_sample_submit.csv") # type: ignore
    
    if output_path is None:
        output_dir = get_project_path("prediction_result")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"submission{suffix}.csv" # type: ignore
    
    # 加载提交模板
    submit_df = pd.read_csv(sample_path, sep=' ') # type: ignore
    
    # 填充预测值
    submit_df['SaleID'] = predictions.index
    submit_df['price'] = predictions.values
    
    # 保存
    submit_df.to_csv(output_path, index=False, sep=' ')
    logger.info(f"预测结果已保存到: {output_path}")
    
    return submit_df
