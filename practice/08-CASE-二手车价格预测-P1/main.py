#!/usr/bin/env python3
"""
二手车价格预测项目主入口
"""

from pathlib import Path
from loguru import logger
from dotenv import load_dotenv


def get_project_path(*paths: str) -> Path:
    """获取项目路径的统一方法"""
    try:
        current_dir = Path(__file__).parent
        return current_dir.joinpath(*paths)
    except NameError:
        return Path.cwd().joinpath(*paths)


def setup_logging():
    """配置日志"""
    log_dir = get_project_path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "app.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    logger.info("日志系统初始化完成")


def load_environment():
    """加载环境变量"""
    load_dotenv(get_project_path(".env"))
    logger.info("环境变量加载完成")


def main():
    """主函数"""
    print("=" * 60)
    print("二手车价格预测项目")
    print("=" * 60)
    
    # 初始化环境
    setup_logging()
    load_environment()
    
    # 检查数据文件
    train_data = get_project_path("data", "used_car_train_20200313.csv")
    test_data = get_project_path("data", "used_car_testB_20200421.csv")
    
    if not train_data.exists():
        logger.warning(f"训练数据文件不存在: {train_data}")
    if not test_data.exists():
        logger.warning(f"测试数据文件不存在: {test_data}")
    
    logger.info("项目环境初始化完成")
    logger.info(f"项目路径: {get_project_path()}")
    
    # 显示可用模块
    print("\n可用模块:")
    print("  - 数据分析: feature/modeling_v*_analysis.py")
    print("  - 模型训练: code/modeling_v*.py")
    print("  - 可视化: feature/plot_*.py")
    
    print("\n下一步操作:")
    print("  1. 查看数据分析: 运行 feature/ 目录下的分析脚本")
    print("  2. 训练模型: 运行 code/ 目录下的建模脚本")
    print("  3. 使用 Jupyter: 打开 二手车价格预测.ipynb")


if __name__ == "__main__":
    main()