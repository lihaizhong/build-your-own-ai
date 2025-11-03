import os
import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role

load_dotenv(verbose=True)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def get_response(messages):
  response = dashscope.Generation.call(
    model="deepseek-r1",
    messages=messages,
    result_format="message"
  )
  return response

# 测试对话
messages = [
  {"role": "system", "content": "You are a helpful assistant"},
  {"role": "user", "content": "你好，你是什么大模型？"}
]

response = get_response(messages)
print(response.output.choices[0].message.content)

# 情感分析
review = '这款音效特别好 给你意想不到的音质。'
messages = [
  {"role": "system", "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向"},
  {"role": "user", "content": review}
]

response = get_response(messages)
print(response.output.choices[0].message.content)
