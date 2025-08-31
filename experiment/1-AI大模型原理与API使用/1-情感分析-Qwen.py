import os
import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role

# 自动搜索 .env 文件
load_dotenv(verbose=True)
# 从 .env 文件中获取 DASHSCOPE_API_KEY
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 封装模型响应函数
def get_response(messages):
  response = dashscope.Generation.call(
    model="deepseek-v3",
    messages=messages,
    # 将输出设置为message形式
    result_format="message"
  )
  return response

messages=[
    {"role": "system", "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向"},
    {"role": "user", "content": "这款音效特别好 给你意想不到的音质。"}
  ]

response = get_response(messages)

print(response.output.choices[0].message.content)
