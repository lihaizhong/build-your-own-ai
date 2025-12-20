import os
import requests
from dotenv import load_dotenv

load_dotenv(verbose=True)
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

def query_ollama(prompt, model="deepseek-r1:8b"):
  url = f"{ollama_base_url}/api/generate"
  data = {
    "model": model,
    "prompt": prompt,
    # 设置为 True 可以获取流式响应
    "stream": False
  }
  response = requests.post(url, json=data)
  if response.status_code == 200:
    return response.json()["response"]
  else:
    raise Exception(f"API 请求失败: {response.text}")

# 使用示例
response = query_ollama("你好，请介绍一下你自己")
print(response)
