import os
import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role

load_dotenv(verbose=True)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def get_response(messages):
  response = dashscope.MultiModalConversation.call(
    # 视觉大模型
    model="qwen-vl-plus",
    messages=messages
  )
  return response

messages=[
  {
    "role": "user",
    "content": [
      { "image": "https://aiwucai.oss-cn-huhehaote.aliyuncs.com/pdf_table.jpg" },
      { "text": "这是一个表格图片，帮我提取里面的内容，输出JSON格式" }
    ]
  }
]

response = get_response(messages)
print("===response===", response)

if response.status_code == 200: # type: ignore
  print("最终结果：", response.output.choices[0].message.content[0]["text"]) # type: ignore
else:
  print("API调用出错：", f"【{response.code}】{response.message}") # type: ignore

