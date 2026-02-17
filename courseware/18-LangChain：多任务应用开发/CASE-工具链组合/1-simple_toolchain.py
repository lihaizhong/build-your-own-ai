"""
1-simple_toolchain.py - LangChain 1.29 版本

主要变更:
1. 使用 langchain.agents.create_agent
2. Tool 使用 @tool 装饰器
3. 使用 Chat Model
"""
import json
import os
import dashscope
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key


# 定义工具 - 使用 @tool 装饰器
@tool
def text_analysis(text: str) -> str:
    """
    分析文本内容，提取字数、字符数和情感倾向。
    输入: 要分析的文本
    输出: 分析结果
    """
    word_count = len(text.split())
    char_count = len(text)
    
    positive_words = ["好", "优秀", "喜欢", "快乐", "成功", "美好"]
    negative_words = ["差", "糟糕", "讨厌", "悲伤", "失败", "痛苦"]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    sentiment = "积极" if positive_count > negative_count else "消极" if negative_count > positive_count else "中性"
    
    return f"文本分析结果:\n- 字数: {word_count}\n- 字符数: {char_count}\n- 情感倾向: {sentiment}"


@tool
def data_conversion(input_data: str) -> str:
    """
    在不同数据格式之间转换，如JSON、CSV等。
    输入: JSON或CSV数据，自动进行反向转换
    输出: 转换后的数据
    """
    try:
        # JSON 到 CSV
        if input_data.strip().startswith('['):
            data = json.loads(input_data)
            if isinstance(data, list) and data:
                headers = list(data[0].keys())
                csv = ",".join(headers) + "\n"
                for item in data:
                    row = [str(item.get(header, "")) for header in headers]
                    csv += ",".join(row) + "\n"
                return csv
        
        # CSV 到 JSON
        lines = input_data.strip().split("\n")
        if len(lines) >= 2 and ',' in lines[0]:
            headers = lines[0].split(",")
            result = []
            for line in lines[1:]:
                values = line.split(",")
                if len(values) == len(headers):
                    item = {headers[i]: values[i] for i in range(len(headers))}
                    result.append(item)
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        return "无法识别的数据格式，请提供 JSON 或 CSV 格式的数据"
    
    except Exception as e:
        return f"转换失败: {str(e)}"


@tool
def text_processing(operation: str) -> str:
    """
    处理文本内容，如查找、替换、统计等。
    输入格式: 操作:内容 或 操作:参数:内容
    支持操作: 统计行数、查找
    示例: "统计行数:第一行\\n第二行" 或 "查找:关键词:文本内容"
    输出: 处理结果
    """
    try:
        parts = operation.split(":", 2)
        if len(parts) < 2:
            return "格式错误，请使用: 操作:内容 或 操作:参数:内容"
        
        op = parts[0].strip().lower()
        
        if op in ["统计行数", "count_lines"]:
            content = parts[1] if len(parts) == 2 else parts[2]
            return f"文本共有 {len(content.splitlines())} 行"
        
        elif op in ["查找", "find"]:
            if len(parts) < 3:
                return "格式: 查找:关键词:文本内容"
            search_text = parts[1]
            content = parts[2]
            lines = content.splitlines()
            matches = []
            for i, line in enumerate(lines):
                if search_text in line:
                    matches.append(f"第 {i+1} 行: {line}")
            if matches:
                return f"找到 {len(matches)} 处匹配:\n" + "\n".join(matches)
            return f"未找到文本 '{search_text}'"
        
        else:
            return f"不支持的操作: {op}，支持的操作: 统计行数、查找"
    
    except Exception as e:
        return f"处理失败: {str(e)}"


# 创建工具链
def create_tool_chain():
    """创建工具链"""
    # 创建工具列表
    tools = [text_analysis, data_conversion, text_processing]
    
    # 初始化 Chat 模型
    llm = ChatTongyi(model="qwen-turbo", api_key=api_key)  # type: ignore[arg-type]
    
    # 创建 Agent - LangChain 1.29 方式
    agent = create_agent(llm, tools=tools)  # type: ignore[arg-type]
    
    return agent

# 示例：使用工具链处理任务
def process_task(task_description):
    """
    使用工具链处理任务
    """
    try:
        agent = create_tool_chain()
        response = agent.invoke({
            "messages": [{"role": "user", "content": task_description}]
        })
        # 获取最后一条消息
        return response["messages"][-1].content
    except Exception as e:
        return f"处理任务时出错: {str(e)}"

# 示例用法
if __name__ == "__main__":
    # 示例1: 文本分析与处理
    task1 = "分析以下文本的情感倾向，并统计其中的行数：'这个产品非常好用，我很喜欢它的设计，使用体验非常棒！\n价格也很合理，推荐大家购买。\n客服态度也很好，解答问题很及时。'"
    print("任务1:", task1)
    print("结果:", process_task(task1))
    
    # 示例2: 数据格式转换
    task2 = "将以下CSV数据转换为JSON格式：'name,age,comment\n张三,25,这个产品很好\n李四,30,服务态度差\n王五,28,性价比高'"
    print("\n任务2:", task2)
    print("结果:", process_task(task2))