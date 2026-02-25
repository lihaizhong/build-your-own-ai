"""MCP Agent 调用示例项目

本项目展示了如何使用 Qwen-Agent 集成多个 MCP 服务，实现：
1. 自驾游规划（高德地图 MCP）
2. 网页转 Markdown（Fetch MCP）
3. 新闻检索（Bing 搜索 MCP）
4. 桌面 TXT 文件统计（自定义 MCP Server）
"""

from .main import init_agent_service, app_gui, app_tui, test

__all__ = [
    "init_agent_service",
    "app_gui",
    "app_tui",
    "test",
]
