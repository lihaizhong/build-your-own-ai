"""MCP Agent 调用示例 - 主程序

基于 Qwen-Agent 集成多个 MCP 服务，支持：
1. 自驾游规划（高德地图 MCP）
2. 网页转 Markdown（Fetch MCP）
3. 新闻检索（Bing 搜索 MCP）
4. 桌面 TXT 文件统计（自定义 MCP Server）

运行方式：
- Web GUI: python main.py
- TUI 模式: 修改 __main__ 中的调用
"""

import os
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

# 加载环境变量
load_dotenv()

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 60  # type: ignore

# 系统提示词
SYSTEM_PROMPT = '''你是一个智能助手，集成了多个 MCP 服务，可以帮助用户完成以下任务：

## 1. 自驾游规划
- 使用高德地图服务规划自驾游路线
- 查找沿途景点、加油站、餐厅等
- 提供详细的行程建议

## 2. 网页内容获取
- 获取指定网页的内容
- 将网页内容转换为 Markdown 格式
- 提取网页中的关键信息

## 3. 新闻检索
- 搜索最新的新闻资讯
- 根据关键词检索相关新闻
- 提供新闻摘要和链接

## 4. 文件管理
- 统计桌面上的 txt 文件数量
- 列出所有 txt 文件
- 读取指定 txt 文件的内容

请根据用户的需求，选择合适的工具来完成任务。回答时请使用中文，并保持专业和友好的态度。
'''


def get_mcp_config() -> dict:
    """获取 MCP 服务配置
    
    根据环境变量配置可选启用不同的 MCP 服务：
    - AMAP_API_KEY: 高德地图 MCP（自驾游规划）
    - MODELSCOPE_API_KEY: SSE 远程服务（Fetch、Bing搜索）
    - ENABLE_SSE_MCP: 是否启用 SSE 远程服务（默认禁用）
    
    注意: SSE 远程服务目前可能存在连接问题，建议先测试本地 MCP 服务。
    
    Returns:
        dict: MCP 服务配置字典
    """
    # 获取 API Keys
    amap_api_key = os.getenv('AMAP_API_KEY', '')
    modelscope_api_key = os.getenv('MODELSCOPE_API_KEY', '')
    enable_sse_mcp = os.getenv('ENABLE_SSE_MCP', 'false').lower() == 'true'
    
    # 获取当前脚本目录（用于定位 txt_counter.py）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    txt_counter_path = os.path.join(current_dir, 'mcp_servers', 'txt_counter.py')
    
    mcp_servers = {}
    
    # 高德地图 MCP（npm 包方式）- 需要 API Key
    if amap_api_key:
        mcp_servers["amap-maps"] = {
            "command": "npx",
            "args": ["-y", "@amap/amap-maps-mcp-server"],
            "env": {
                "AMAP_MAPS_API_KEY": amap_api_key
            }
        }
        logger.info("已启用高德地图 MCP 服务")
    else:
        logger.warning("未配置 AMAP_API_KEY，高德地图 MCP 服务已禁用")
    
    # SSE 远程服务（Fetch、Bing搜索）- 需要 ModelScope API Key 且显式启用
    # 注意: 此服务可能存在连接问题，默认禁用
    if enable_sse_mcp and modelscope_api_key:
        # Fetch MCP（SSE 远程服务）
        mcp_servers["fetch"] = {
            "type": "sse",
            "url": f"https://mcp.api-inference.modelscope.cn/sse/{modelscope_api_key}"
        }
        # Bing 中文搜索 MCP（SSE 远程服务）
        mcp_servers["bing-cn-mcp-server"] = {
            "type": "sse",
            "url": f"https://mcp.api-inference.modelscope.cn/sse/{modelscope_api_key}"
        }
        logger.info("已启用 Fetch 和 Bing 搜索 MCP 服务")
    else:
        if not enable_sse_mcp:
            logger.info("SSE 远程服务已禁用（设置 ENABLE_SSE_MCP=true 启用）")
        else:
            logger.warning("未配置 MODELSCOPE_API_KEY，Fetch 和 Bing 搜索 MCP 服务已禁用")
    
    # 自定义 TXT 计数器 MCP（本地 Python 服务）- 始终启用
    mcp_servers["txt-counter"] = {
        "command": "python3",
        "args": [txt_counter_path]
    }
    logger.info("已启用 TXT 文件统计 MCP 服务")
    
    return {"mcpServers": mcp_servers}


def init_agent_service() -> Assistant:
    """初始化 Agent 服务
    
    Returns:
        Assistant: 配置好的助手实例
    """
    # LLM 配置
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 60,
        'retry_count': 3,
    }
    
    # MCP 工具配置
    tools = [get_mcp_config()]
    
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='MCP 智能助手',
            description='集成高德地图/网页获取/Bing搜索/TXT文件统计的智能助手',
            system_message=SYSTEM_PROMPT,
            function_list=tools,  # type: ignore
        )
        logger.info("Agent 初始化成功")
        return bot
    except Exception as e:
        logger.error(f"Agent 初始化失败: {str(e)}")
        raise


def test(query: str = '统计一下我桌面上有多少个txt文件', file: Optional[str] = None):
    """测试模式
    
    Args:
        query: 测试查询语句
        file: 可选的输入文件路径
    """
    try:
        bot = init_agent_service()
        messages = []
        
        if not file:
            messages.append({'role': 'user', 'content': query})
        else:
            messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})
        
        logger.info(f"处理查询: {query}")
        for response in bot.run(messages):
            print('Agent 响应:', response)
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")


def app_tui():
    """终端交互模式（TUI）
    
    提供命令行交互界面，支持连续对话。
    """
    try:
        bot = init_agent_service()
        messages = []
        
        print("=" * 60)
        print("MCP 智能助手 - 终端模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60)
        
        while True:
            try:
                query = input('\n用户: ').strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not query:
                    continue
                
                file = input('文件路径 (可选，直接回车跳过): ').strip()
                
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})
                
                print("\n助手: ", end="")
                response = []
                for response in bot.run(messages):
                    if response:
                        # 提取并显示响应内容
                        for msg in response:
                            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                content = msg.get('content', '')
                                if isinstance(content, str):
                                    print(content)
                messages.extend(response)
                
            except KeyboardInterrupt:
                print("\n\n已中断，再见！")
                break
            except Exception as e:
                logger.error(f"处理请求出错: {str(e)}")
                print(f"出错: {str(e)}，请重试")
                
    except Exception as e:
        logger.error(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式（Web GUI）
    
    提供 Web 图形界面，包含预设的查询建议。
    """
    try:
        logger.info("正在启动 Web 界面...")
        bot = init_agent_service()
        
        # 预设查询建议
        chatbot_config = {
            'prompt.suggestions': [
                '规划从北京到上海的7天自驾游',
                '获取网页 https://k.sina.com.cn/article_7732457677_1cce3f0cd01901eeeq.html 的内容并转为Markdown格式',
                '搜索最新的关税新闻',
                '统计桌面上的txt文件数量',
            ]
        }
        
        logger.info("Web 界面已启动")
        WebUI(bot, chatbot_config=chatbot_config).run()
        
    except Exception as e:
        logger.error(f"启动 Web 界面失败: {str(e)}")


if __name__ == '__main__':
    # 运行模式选择
    # test()           # 测试模式
    # app_tui()        # 终端交互模式
    app_gui()          # 图形界面模式（默认）
