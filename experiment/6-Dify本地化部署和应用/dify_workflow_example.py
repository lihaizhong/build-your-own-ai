#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dify 工作流应用调用示例
专门针对工作流类型的应用
"""
import os
from dotenv import load_dotenv
from dify_agent_client import DifyAgentClient

load_dotenv(verbose=True)

def simple_workflow_example():
  """简单的工作流调用示例"""
  BASE_URL = os.getenv("DIFY_BASE_URL")
  API_KEY = os.getenv("DIFY_API_KEY")
  DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")

  # 创建客户端
  client = DifyAgentClient(BASE_URL, API_KEY)
  user_input = "离离原上草"
  print(f"发送消息：{user_input}")
  
  result = client.run_workflow(
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
