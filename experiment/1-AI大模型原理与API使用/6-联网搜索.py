import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(verbose=True)
API_KEY = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
  api_key=API_KEY,
  # 填写DashScope服务的base_url
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
completion = client.chat.completions.create(
  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
  model="qwen-plus",
  messages=[
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '中国队在巴黎奥运会获得了多少枚金牌'}
  ],
  extra_body={
    "enable_search": True
  }
)

print(completion.model_dump_json())
