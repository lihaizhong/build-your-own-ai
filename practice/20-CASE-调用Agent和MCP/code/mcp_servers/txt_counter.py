"""桌面 TXT 文件统计 MCP Server

基于 FastMCP 实现的本地 MCP 服务，提供以下功能：
- 统计桌面 .txt 文件数量
- 列出所有 .txt 文件
- 读取指定 .txt 文件内容
"""

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server 实例
mcp = FastMCP("桌面 TXT 文件统计器")


@mcp.tool()
def count_desktop_txt_files() -> int:
    """统计桌面上 .txt 文件的数量
    
    Returns:
        int: 桌面上 .txt 文件的数量
    """
    desktop_path = Path(os.path.expanduser("~/Desktop"))
    txt_files = list(desktop_path.glob("*.txt"))
    return len(txt_files)


@mcp.tool()
def list_desktop_txt_files() -> str:
    """获取桌面上所有 .txt 文件的列表
    
    Returns:
        str: 格式化的文件列表，包含文件名和数量
    """
    desktop_path = Path(os.path.expanduser("~/Desktop"))
    txt_files = list(desktop_path.glob("*.txt"))
    
    if not txt_files:
        return "桌面上没有找到 .txt 文件。"
    
    file_list = "\n".join([f"- {file.name}" for file in txt_files])
    return f"在桌面上找到 {len(txt_files)} 个 .txt 文件：\n{file_list}"


@mcp.tool()
def read_txt_file(filename: str) -> str:
    """读取指定 txt 文件的内容
    
    Args:
        filename: txt 文件的名称（例如：test.txt）
        
    Returns:
        str: 文件内容，如果文件不存在则返回错误信息
    """
    desktop_path = Path(os.path.expanduser("~/Desktop"))
    file_path = desktop_path / filename
    
    # 检查文件是否存在
    if not file_path.exists():
        return f"错误：文件 '{filename}' 不存在于桌面上。"
    
    # 检查文件是否是 txt 文件
    if file_path.suffix.lower() != '.txt':
        return f"错误：文件 '{filename}' 不是 txt 文件。"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件 '{filename}' 的内容：\n\n{content}"
    except Exception as e:
        return f"读取文件时发生错误：{str(e)}"


if __name__ == "__main__":
    # 启动 MCP Server
    mcp.run()
