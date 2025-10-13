"""
数据分析报告服务器
基于 Flask 框架提供静态文件服务，用于查看 YData Profiling 生成的数据分析报告
启动后会自动在浏览器中打开 data-analysis.html

使用方式:
    uv run python 启动数据分析报告.py
"""
from flask import Flask, send_from_directory, redirect
from pathlib import Path
import webbrowser
import threading
import time
from loguru import logger

# 配置参数
PORT = 8000
HOST = "0.0.0.0"
HTML_FILE = "data-analysis.html"
STATIC_DIR = "user_data"

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()
STATIC_PATH = SCRIPT_DIR / STATIC_DIR
HTML_PATH = STATIC_PATH / HTML_FILE

# 创建 Flask 应用
app = Flask(__name__)

# 配置日志
logger.info(f"工作目录: {SCRIPT_DIR}")
logger.info(f"静态文件目录: {STATIC_PATH}")


@app.route("/")
def index():
    """根路径重定向到数据分析报告"""
    return redirect(f"/{STATIC_DIR}/{HTML_FILE}")


@app.route(f"/{STATIC_DIR}/<path:filename>")
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory(STATIC_PATH, filename)


def open_browser():
    """延迟打开浏览器，确保服务器已启动"""
    time.sleep(1.5)
    url = f"http://localhost:{PORT}"
    logger.info(f"正在打开浏览器: {url}")
    webbrowser.open(url)


def main():
    """主函数 - 启动服务器"""
    # 检查文件是否存在
    if not HTML_PATH.exists():
        logger.error(f"找不到文件: {HTML_PATH}")
        logger.error(f"请先生成数据分析报告")
        return
    
    logger.success(f"找到数据分析报告: {HTML_FILE}")
    logger.info(f"启动 Flask 服务器...")
    logger.info(f"访问地址: http://localhost:{PORT}")
    logger.info(f"直接访问报告: http://localhost:{PORT}/{STATIC_DIR}/{HTML_FILE}")
    logger.info(f"按 Ctrl+C 停止服务器\n")
    
    # 在新线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动 Flask 服务器
    try:
        app.run(host=HOST, port=PORT, debug=False)
    except KeyboardInterrupt:
        logger.info("\n服务器已停止")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"端口 {PORT} 已被占用，请尝试其他端口")
        else:
            logger.error(f"服务器启动失败: {e}")


if __name__ == "__main__":
    main()
