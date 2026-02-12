"""
Disney RAG问答助手 - 主程序入口
"""

import sys
import argparse
import warnings
from pathlib import Path
from loguru import logger
from typing import Optional
import torch

# 绝对导入，支持直接运行
try:
    from .config import config, load_env_config
    from .data_processor import DocumentProcessor, ImageProcessor
    from .embedding import VectorStore
    from .retrieval import HybridRetriever
    from .generator import AnswerGenerator, RAGPipeline
except ImportError:
    # 作为脚本直接运行时使用绝对导入
    from config import config, load_env_config
    from data_processor import DocumentProcessor, ImageProcessor
    from embedding import VectorStore
    from retrieval import HybridRetriever
    from generator import AnswerGenerator, RAGPipeline


def cleanup_resources():
    """清理资源，防止multiprocessing警告"""
    try:
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 关闭multiprocessing资源（兼容Python 3.14+）
        import multiprocessing.util as mp_util
        if hasattr(mp_util, '_cleanup_all_processes'):
            mp_util._cleanup_all_processes()  # type: ignore
    except Exception as e:
        logger.debug(f"资源清理时出现警告（可忽略）: {e}")


def setup_logger(log_dir: Optional[Path] = None):
    """
    配置日志系统
    
    Args:
        log_dir: 日志目录
    """
    if log_dir is None:
        log_dir = config.output_dir / "logs" # type: ignore
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.add(
        log_dir / "disney_rag_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )


def build_indexes(skip_images: bool = False):
    """构建索引
    
    Args:
        skip_images: 是否跳过图像处理
    """
    logger.info("=" * 60)
    logger.info("开始构建索引")
    logger.info("=" * 60)
    
    # 初始化向量存储
    vector_store = VectorStore()
    
    # 处理文档
    logger.info("\n【Step 1】处理文档...")
    doc_processor = DocumentProcessor()
    chunks = doc_processor.process_directory()
    
    if chunks:
        logger.info(f"共处理 {len(chunks)} 个文本块")
        vector_store.build_text_index(chunks)
    else:
        logger.warning("没有找到可处理的文档")
    
    # 处理图像（添加异常处理）
    if not skip_images:
        logger.info("\n【Step 2】处理图像...")
        try:
            img_processor = ImageProcessor()
            images = img_processor.process_directory()
            
            if images:
                logger.info(f"共处理 {len(images)} 个图像")
                vector_store.build_image_index(images)
            else:
                logger.warning("没有找到可处理的图像")
        except Exception as e:
            logger.warning(f"图像处理失败，跳过图像索引: {e}")
            logger.info("继续使用文本索引...")
    else:
        logger.info("\n【Step 2】跳过图像处理（--build-text-only）")
    
    # 保存索引
    logger.info("\n【Step 3】保存索引...")
    vector_store.save_indexes()
    
    logger.info("\n" + "=" * 60)
    logger.info("索引构建完成")
    logger.info("=" * 60)


def interactive_mode():
    """交互式问答模式"""
    logger.info("=" * 60)
    logger.info("Disney RAG问答助手 - 交互模式")
    logger.info("=" * 60)
    logger.info("输入 'quit' 或 'exit' 退出")
    logger.info("输入 'help' 查看帮助信息")
    logger.info("输入 Ctrl+C 也可以退出")
    logger.info("=" * 60 + "\n")
    
    # 加载索引
    vector_store = VectorStore()
    vector_store.load_indexes()
    
    # 创建RAG管道
    retriever = HybridRetriever(vector_store)
    generator = AnswerGenerator()
    rag_pipeline = RAGPipeline(retriever, generator)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n请输入您的问题: ").strip()
            
            if not question:
                continue
            
            # 退出命令
            if question.lower() in ['quit', 'exit', '退出']:
                logger.info("再见！")
                break
            
            # 帮助命令
            if question.lower() in ['help', '帮助']:
                print_help()
                continue
            
            # 执行查询
            print(f"\n{'='*60}")
            print(f"正在查询: {question}")
            print(f"{'='*60}")
            
            result = rag_pipeline.query(
                question=question,
                return_retrieval_results=True
            )
            
            # 显示结果
            print(f"\n{rag_pipeline.format_response(result)}")
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，正在退出...")
            logger.info("用户中断程序")
            break
        except EOFError:
            # 处理 EOF (Ctrl+D)
            print("\n\n检测到 EOF，正在退出...")
            logger.info("检测到 EOF，退出程序")
            break
        except Exception as e:
            logger.error(f"查询失败: {e}")
            print(f"\n抱歉，处理您的问题时出现了错误: {e}")


def print_help():
    """打印帮助信息"""
    help_text = """
【帮助信息】

可用命令：
  quit/exit   - 退出程序
  help        - 显示此帮助信息

使用说明：
  1. 输入您的问题，系统会基于文档和图像信息回答
  2. 如果查询包含"海报"、"图片"等关键词，会触发图像检索
  3. 系统会返回答案和相关上下文信息

示例问题：
  - 迪士尼有哪些经典动画电影？
  - 展示一下迪士尼的海报
  - 米老鼠是什么时候创造的？
  - 迪士尼乐园在哪里？
"""
    print(help_text)


def single_query(question: str):
    """单次查询模式"""
    logger.info(f"执行单次查询: {question}")
    
    # 加载索引
    vector_store = VectorStore()
    vector_store.load_indexes()
    
    # 创建RAG管道
    retriever = HybridRetriever(vector_store)
    generator = AnswerGenerator()
    rag_pipeline = RAGPipeline(retriever, generator)
    
    # 执行查询
    result = rag_pipeline.query(question, return_retrieval_results=True)
    
    # 显示结果
    print(rag_pipeline.format_response(result))
    
    return result


def main():
    """主函数"""
    # 禁用multiprocessing资源泄漏警告（不影响功能）
    warnings.filterwarnings("ignore", message=".*leaked semaphore.*")
    
    parser = argparse.ArgumentParser(
        description="Disney RAG问答助手 - 基于向量数据库的智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 构建索引
  python -m code.main --build
  
  # 交互式问答
  python -m code.main --interactive
  
  # 单次查询
  python -m code.main --query "迪士尼有哪些经典动画电影？"
        """
    )
    
    parser.add_argument(
        '--build',
        action='store_true',
        help='构建文档和图像索引'
    )
    
    parser.add_argument(
        '--build-text-only',
        action='store_true',
        help='仅构建文档索引（跳过图像处理）'
    )
    
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='启动交互式问答模式'
    )
    
    parser.add_argument(
        '--query',
        '-q',
        type=str,
        help='执行单次查询'
    )
    
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=None,
        help='日志目录'
    )
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_env_config()
    
    # 配置日志
    setup_logger(args.log_dir)
    
    # 打印欢迎信息
    print_logo()
    
    # 执行命令
    if args.build:
        build_indexes(skip_images=False)
    elif args.build_text_only:
        build_indexes(skip_images=True)
    elif args.interactive:
        interactive_mode()
    elif args.query:
        single_query(args.query)
    else:
        # 默认启动交互模式
        parser.print_help()
        print("\n未指定命令，启动交互模式...")
        interactive_mode()
    
    # 程序退出时清理资源
    cleanup_resources()


def print_logo():
    """打印Logo"""
    logo = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║            Disney RAG 问答助手 v1.0.0                   ║
║                                                          ║
║       基于向量数据库的智能问答系统                       ║
║       支持文档处理、图像检索和混合问答                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(logo)


if __name__ == "__main__":
    main()