"""
2-network_diagnosis_agent.py - LangChain 1.29 版本

主要变更:
1. 使用 langchain.agents.create_agent
2. Tool 使用 @tool 装饰器
3. 使用 Chat Model
"""
import os
import re

from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent
import dashscope

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key


# --- 定义网络诊断工具 - 使用 @tool 装饰器 ---

@tool
def ping(target: str) -> str:
    """
    检查本机到指定主机名或 IP 地址的网络连通性。
    输入: 目标主机名或 IP 地址
    输出: 连通性状态（成功/失败）和网络延迟
    """
    print(f"--- 模拟执行 Ping: {target} ---")
    if "unreachable" in target or target == "192.168.1.254":
        return f"Ping {target} 失败：请求超时。"
    elif target == "localhost" or target == "127.0.0.1":
        return f"Ping {target} 成功：延迟 <1ms。"
    elif "example.com" in target:
        import random
        delay = random.randint(20, 80)
        return f"Ping {target} 成功：延迟 {delay}ms。"
    else:
        import random
        delay = random.randint(5, 50)
        return f"Ping {target} 成功：延迟 {delay}ms。"


@tool
def dns_lookup(hostname: str) -> str:
    """
    解析给定的主机名，获取其对应的 IP 地址。
    输入: 要解析的主机名
    输出: IP 地址或解析失败信息
    """
    print(f"--- 模拟 DNS 查询: {hostname} ---")
    if hostname == "www.example.com":
        return f"DNS 解析 {hostname} 成功：IP 地址是 93.184.216.34"
    elif hostname == "internal.service.local":
        return f"DNS 解析 {hostname} 成功：IP 地址是 192.168.1.100"
    elif hostname == "unknown.domain.xyz":
        return f"DNS 解析 {hostname} 失败：找不到主机。"
    elif hostname == "127.0.0.1" or re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", hostname):
        return f"输入 '{hostname}' 已经是 IP 地址，无需 DNS 解析。"
    else:
        return f"DNS 解析 {hostname} 成功：IP 地址是 10.0.0.5"


@tool
def check_interface(interface_name: str = "default") -> str:
    """
    检查本机网络接口的状态（如 IP 地址、是否启用）。
    输入: 接口名称（可选，不提供则检查默认接口）
    输出: 接口状态信息
    """
    print(f"--- 模拟检查接口状态: {interface_name} ---")
    if interface_name and "eth1" in interface_name.lower():
        return f"接口 '{interface_name}' 状态：关闭 (Administratively down)"
    else:
        return f"接口 'Ethernet'/'Wi-Fi' 状态：启用, IP 地址: 192.168.1.50, 子网掩码: 255.255.255.0, 网关: 192.168.1.1"


@tool
def analyze_logs(keywords: str) -> str:
    """
    搜索系统或应用程序日志，查找与网络问题相关的条目。
    输入: 描述问题的关键词（例如 'timeout', 'connection refused', 'dns error'）
    输出: 找到的相关日志条目摘要
    """
    print(f"--- 模拟分析日志: 关键词='{keywords}' ---")
    keywords_lower = keywords.lower()
    if "timeout" in keywords_lower or "超时" in keywords_lower:
        return (f"在日志中找到 3 条与 '{keywords}' 相关的条目：\n"
                f"- [Error] 连接到 10.0.0.88:8080 超时\n"
                f"- [Warning] 对 api.external.com 的请求超时\n"
                f"- [Error] 内部服务通信超时")
    elif "connection refused" in keywords_lower or "连接被拒绝" in keywords_lower:
        return (f"在日志中找到 1 条与 '{keywords}' 相关的条目：\n"
                f"- [Error] 连接到 192.168.1.200:5432 失败：Connection refused")
    elif "dns" in keywords_lower:
        return (f"在日志中找到 2 条与 '{keywords}' 相关的条目：\n"
                f"- [Warning] DNS 服务器 8.8.8.8 响应慢\n"
                f"- [Error] 无法解析主机名 'failed.internal.service'")
    else:
        return f"在日志中未找到与 '{keywords}' 相关的明显网络错误条目。"


# --- 创建 Agent ---

def create_network_diagnosis_agent():
    """创建网络故障诊断的 Agent。"""
    # 工具列表
    tools = [ping, dns_lookup, check_interface, analyze_logs]

    # 初始化 Chat 模型
    llm = ChatTongyi(model="qwen-turbo", api_key=api_key)  # type: ignore[arg-type]

    # 创建 Agent - LangChain 1.29 方式
    agent = create_agent(llm, tools=tools)  # type: ignore[arg-type]

    return agent


# --- 使用 Agent 处理网络诊断任务 ---

def diagnose_network_issue(issue_description: str):
    """
    使用网络诊断 Agent 处理用户报告的网络问题。
    """
    try:
        print(f"\n--- 开始诊断任务 ---")
        print(f"用户问题: {issue_description}")
        agent = create_network_diagnosis_agent()
        response = agent.invoke({
            "messages": [{"role": "user", "content": issue_description}]
        })
        return response["messages"][-1].content
    except Exception as e:
        import traceback
        print(f"处理诊断任务时发生错误: {str(e)}")
        traceback.print_exc()
        return f"处理诊断任务时出错: {str(e)}"


# --- 主程序入口 ---
if __name__ == "__main__":
    # 示例 1: 无法访问特定网站
    task1 = "我无法访问 www.example.com，浏览器显示连接超时。"
    print("诊断任务 1:")
    result1 = diagnose_network_issue(task1)
    print("\n--- 诊断任务 1 结束 ---")
    print(f"最终诊断结果: {result1}")

    print("\n" + "="*50 + "\n")

    # 示例 2: 内部服务访问失败
    task2 = "连接到内部数据库服务器 失败，提示 'connection refused'。"
    print("诊断任务 2:")
    result2 = diagnose_network_issue(task2)
    print("\n--- 诊断任务 2 结束 ---")
    print(f"最终诊断结果: {result2}")
