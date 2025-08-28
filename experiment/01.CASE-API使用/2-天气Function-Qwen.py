import os
import json
import requests
import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role

load_dotenv(verbose=True)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

# 编写你的天气函数
def get_current_weather(location, unit="摄氏度"):
  try:
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?key={AMAP_API_KEY}&city={location}"
    response = requests.get(url)
  
    data = response.json()
    print("获取天气信息成功", data)
    live = data["lives"][0]
    weather_info = {
      "location": location,
      "temperature": live["temperature"],
      "unit": unit,
      "forecast": [live["weather"], live["winddirection"]],
    }
    
    return json.dumps(weather_info)
  except Exception as e:
    print("获取天气信息失败", str(e))

def get_response(messages):
  try:
    response = dashscope.Generation.call(
      model="qwen-max",
      functions=functions,
      messages=messages,
      result_format="message"
    )
    
    return response
  except Exception as e:
    print("API调用出错:", str(e))
    
def run_conversation():
  query = "北京的天气怎么样"
  messages = [{ "role": "user", "content": query }]
  print("===messages===", messages)
  
  # 得到第一次响应
  response = get_response(messages)
  
  if not response or not response.output:
    print("获取响应失败")
    return None
  
  print("===response1===", response)
  message = response.output.choices[0].message
  print("===message1===", message)
  print("\n\n\n")
  
  # 判断 message 是否要 call function
  if hasattr(message, "function_call") and message.function_call:
    function_call = message.function_call
    tool_name = function_call["name"]
    # 执行function call
    arguments = json.loads(function_call["arguments"])
    print("===arguments===", arguments)
    tool_response = get_current_weather(location=arguments.get("location"), unit=arguments.get("unit"))
    tool_info = { "role": "function", "name": tool_name, "content": tool_response }
    print("===tool_info===", tool_info)
    messages.append(tool_info)
    print("===messages===", messages)
    
    # 得到第二次响应
    response = get_response(messages)
    if not response or not response.output:
      print("获取第二次响应失败")
      return None
    
    print("===response2===", response)
    message = response.output.choices[0].message
    print("===message2===", message)
    return message
  return message
  
functions = [
  {
    "name": "get_current_weather",
    "description": "获取给予地区的当前天气状况",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名称或者地区名称，如：北京、上海、广州、深圳"
        },
        "unit": {
          "type": "string",
        }
      }
    }
  }
]

if __name__ == "__main__":
  result = run_conversation()
  if result:
    print("\n\n\n最终结果:", result)
  else:
    print("\n\n\n对话执行失败")
