import json
import os
from datetime import datetime
import pytz
import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role

load_dotenv(verbose=True)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def get_current_time(timezone='Europe/London'):
  # Create a datetime object in UTC (or another known time zone)
  utc_time = datetime.now(pytz.utc)

  # Convert it to a different time zone
  current_tz = pytz.timezone(timezone)
  local_time = utc_time.astimezone(current_tz)

  time_info = {
    'current_local_time': str(local_time)
  }
  return json.dumps(time_info, ensure_ascii=False)

def get_timezone():
  timezone = 'Europe/London'
  timezone_info = {
    'timezone': timezone
  }
  return json.dumps(timezone_info, ensure_ascii=False)

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_timezone",
      "description": "获取时区信息",
      "parameters": {},
      "required": []
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_current_time",
      "description": "根据时区信息获取当前时区的时间",
      # "description": "Get the current local time in a specific time zone",
      "parameters": {
        "type": "object",
        "properties": {
          "timezone": {
            "type": "string",
            "description": "The time zone"
          }
        }
      },
      # "required": ["timezone"]
      "required": []
    }
  }
]

current_locals = locals()

def get_response(messages):
  response = dashscope.Generation.call(
    model='qwen-turbo',
    messages=messages,
    tools=tools,
    # tool_choice={"type": "function", "function": {"name": "get_current_time"}},
    result_format='message'  # 将输出设置为message形式
  )
  return response

query = "Hi 我回来啦！现在几点啦？"
messages = [
  {'role': 'system', 'content': "你是一个感情陪伴聊天机器人，结合当前用户所在时区的时间，从早安、午安或者晚上好三种问候语中选择最恰当的一个对用户问好，以及根据用户发来消息的情绪对用户进行回应。"},
  {'role': 'user', 'content': query}
]

while True:
  print("messages=", messages)
  response = get_response(messages)
  message = response.output.choices[0].message # type: ignore
  messages.append(message)
  print('response=', response)

  if response.output.choices[0].finish_reason == 'stop': # type: ignore
    break
  
  # 判断用户是否要call function
  if message.tool_calls:
    # 获取fn_name, fn_arguments
    fn_name = message.tool_calls[0]['function']['name']
    fn_arguments = message.tool_calls[0]['function']['arguments']
    arguments_json = json.loads(fn_arguments)
    print(f'fn_name={fn_name} fn_arguments={fn_arguments}')
    function = current_locals[fn_name]
    tool_response = function(**arguments_json)
    tool_info = {"name": fn_name, "role":"tool", "content": tool_response}
    print('tool_info=', tool_info)
    messages.append(tool_info)
