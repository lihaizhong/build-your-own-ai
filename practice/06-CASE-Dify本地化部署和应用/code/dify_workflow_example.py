#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dify 工作流应用调用示例
专门针对工作流类型的应用
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv
from dify_agent_client import DifyAgentClient

load_dotenv(verbose=True)

def simple_workflow_example() -> None:
  """简单的工作流调用示例"""
  BASE_URL: str | None = os.getenv("DIFY_BASE_URL")
  API_KEY: str | None = os.getenv("DIFY_API_KEY")
  DEFAULT_USER_ID: str | None = os.getenv("DEFAULT_USER_ID")

  # 检查环境变量是否存在
  if not BASE_URL or not API_KEY or not DEFAULT_USER_ID:
    print("错误：请设置环境变量 DIFY_BASE_URL, DIFY_API_KEY 和 DEFAULT_USER_ID")
    return

  # 创建客户端
  client = DifyAgentClient(BASE_URL, API_KEY)
  user_input: str = "离离原上草"
  print(f"发送消息：{user_input}")
  
  result: Dict[str, Any] = client.run_workflow(
    inputs={"input": user_input},
    user_id=DEFAULT_USER_ID
  )
  
  if result.get("error"):
    print(f"调用失败: {result.get('message')}")
  else:
    print(f"工作流回复: {result.get('answer')}")
    print(f"工作流运行ID: {result.get('workflow_run_id')}")
    
if __name__ == "__main__":
  simple_workflow_example()
