"""
网络工程师智能助手 Agent

使用 LangChain 组合多种工具，实现网络工程师日常工作任务的自动化。
支持：
- 网络故障诊断
- 配置分析与安全检查
- 日志分析与问题排查
- 文本处理与数据转换

适配 LangChain 1.2.9+ 版本
"""

import os
from typing import List, Dict, Optional

from langchain_core.tools import Tool
from langchain_community.llms import Tongyi
from langchain_openai import ChatOpenAI

# 导入自定义工具
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    TextAnalysisTool,
    DataConversionTool,
    TextProcessingTool,
    NetworkDiagnosisTool,
    ConfigAnalysisTool,
    LogAnalysisTool,
)


class NetworkEngineerAgent:
    """网络工程师智能助手"""
    
    def __init__(
        self,
        llm_type: str = "tongyi",
        model_name: str = "qwen-turbo",
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        初始化网络工程师 Agent
        
        Args:
            llm_type: LLM 类型，支持 "tongyi" 或 "openai"
            model_name: 模型名称
            api_key: API 密钥（如未提供，从环境变量读取）
            verbose: 是否显示详细日志
        """
        self.llm_type = llm_type
        self.model_name = model_name
        self.verbose = verbose
        
        # 初始化 LLM
        self.llm = self._init_llm(llm_type, model_name, api_key)
        
        # 初始化工具
        self.tools = self._init_tools()
        self.tools_dict = {tool.name: tool for tool in self.tools}
        
        # 初始化记忆
        self.memory: List[Dict] = []
        
        # 初始化系统提示
        self.system_prompt = self._build_system_prompt()
    
    def _init_llm(
        self,
        llm_type: str,
        model_name: str,
        api_key: Optional[str]
    ):
        """初始化 LLM"""
        if llm_type == "tongyi":
            dashscope_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
            if not dashscope_api_key:
                raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
            
            return Tongyi(
                model_name=model_name,
                dashscope_api_key=dashscope_api_key,
                temperature=0.7,
            )
        
        elif llm_type == "openai":
            openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
            openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            if not openai_api_key:
                raise ValueError("请设置 OPENAI_API_KEY 环境变量或传入 api_key 参数")
            
            return ChatOpenAI(
                model=model_name,
                openai_api_key=openai_api_key, # type: ignore
                openai_api_base=openai_base_url, # type: ignore
                temperature=0.7,
            )
        
        else:
            raise ValueError(f"不支持的 LLM 类型: {llm_type}")
    
    def _init_tools(self) -> List[Tool]:
        """初始化工具列表"""
        tools = [
            Tool(
                name="文本分析",
                func=TextAnalysisTool().run,
                description=(
                    "分析文本内容的工具。"
                    "可以统计字数、字符数，进行情感分析，提取关键词。"
                    "输入：需要分析的文本内容。"
                    "输出：包含各项分析结果。"
                )
            ),
            Tool(
                name="数据转换",
                func=DataConversionTool().run,
                description=(
                    "数据格式转换工具。"
                    "支持 JSON/YAML/CSV 格式互转，数据格式验证，Cisco 配置转 JSON。"
                    "输入格式：'转换类型|数据内容'，如 'json2yaml|{\"key\": \"value\"}'。"
                )
            ),
            Tool(
                name="文本处理",
                func=TextProcessingTool().run,
                description=(
                    "文本处理工具。"
                    "支持文本清洗、分割、正则匹配、IP/URL/邮箱提取等操作。"
                    "输入格式：'处理类型|文本内容' 或 '处理类型|参数|文本内容'。"
                )
            ),
            Tool(
                name="网络诊断",
                func=NetworkDiagnosisTool().run,
                description=(
                    "网络诊断工具（模拟）。"
                    "支持 Ping 测试、DNS 解析、端口检测、路由追踪、连通性检查。"
                    "输入格式：'诊断类型|目标地址' 或 '诊断类型|目标地址|端口'。"
                )
            ),
            Tool(
                name="配置分析",
                func=ConfigAnalysisTool().run,
                description=(
                    "网络设备配置分析工具。"
                    "支持 Cisco/Juniper/Huawei 配置解析、安全检查、接口提取、路由分析等。"
                    "输入格式：'分析类型|配置内容'。"
                )
            ),
            Tool(
                name="日志分析",
                func=LogAnalysisTool().run,
                description=(
                    "日志分析工具。"
                    "支持日志统计、错误提取、时间线分析、IP 统计、模式识别、防火墙日志分析。"
                    "输入格式：'分析类型|日志内容'。"
                )
            ),
        ]
        
        return tools
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        return f"""你是一个专业的网络工程师智能助手，擅长网络故障诊断、配置分析和日志处理。

你可以使用以下工具来帮助用户解决问题：

{tool_descriptions}

当需要使用工具时，请按以下格式回复：
【使用工具：工具名称】
【输入参数：参数内容】

然后等待工具返回结果，再继续分析。

如果不需要使用工具，直接回答用户问题。

请始终用中文回复，保持专业和友好的态度。"""
    
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, str]]:
        """解析工具调用"""
        import re
        
        # 匹配工具调用格式
        tool_pattern = r'【使用工具[：:]\s*([^】]+)】'
        input_pattern = r'【输入参数[：:]\s*([^】]+)】'
        
        tool_match = re.search(tool_pattern, response)
        input_match = re.search(input_pattern, response)
        
        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_input = input_match.group(1).strip() if input_match else ""
            
            return {
                "tool_name": tool_name,
                "tool_input": tool_input
            }
        
        return None
    
    def run(self, query: str) -> str:
        """
        运行 Agent
        
        Args:
            query: 用户查询
            
        Returns:
            Agent 的响应
        """
        try:
            # 构建对话历史
            conversation = f"系统提示：{self.system_prompt}\n\n"
            
            for msg in self.memory[-5:]:  # 只保留最近5轮对话
                conversation += f"用户：{msg['user']}\n"
                conversation += f"助手：{msg['assistant']}\n"
            
            conversation += f"用户：{query}\n助手："
            
            # 获取 LLM 响应
            if self.verbose:
                print("\n🤔 思考中...")
            
            response = self.llm.invoke(conversation)
            
            if isinstance(response, dict):
                response_text = response.get("text", str(response))
            elif hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            # 检查是否需要调用工具
            tool_call = self._parse_tool_call(response_text) # type: ignore
            
            if tool_call:
                tool_name = tool_call["tool_name"]
                tool_input = tool_call["tool_input"]
                
                if tool_name in self.tools_dict:
                    if self.verbose:
                        print(f"\n🔧 调用工具：{tool_name}")
                        print(f"   输入：{tool_input[:100]}{'...' if len(tool_input) > 100 else ''}")
                    
                    # 执行工具
                    tool_result = self.tools_dict[tool_name].invoke(tool_input)
                    
                    if self.verbose:
                        print(f"   结果：{tool_result[:100]}{'...' if len(tool_result) > 100 else ''}")
                    
                    # 将工具结果反馈给 LLM
                    conversation += f"{response_text}\n\n工具返回结果：\n{tool_result}\n\n请根据工具返回的结果，给用户一个完整的回答："
                    
                    final_response = self.llm.invoke(conversation)
                    
                    if isinstance(final_response, dict):
                        final_text = final_response.get("text", str(final_response))
                    elif hasattr(final_response, "content"):
                        final_text = final_response.content
                    else:
                        final_text = str(final_response)
                    
                    # 保存记忆
                    self.memory.append({
                        "user": query,
                        "assistant": final_text
                    })
                    
                    return final_text # type: ignore
                else:
                    return f"未找到工具：{tool_name}"
            
            # 保存记忆
            self.memory.append({
                "user": query,
                "assistant": response_text
            })
            
            return response_text # type: ignore
            
        except Exception as e:
            return f"执行过程中出现错误：{str(e)}"
    
    def chat(self, query: str) -> str:
        """对话模式"""
        return self.run(query)
    
    def clear_memory(self):
        """清除对话记忆"""
        self.memory = []
    
    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return [tool.name for tool in self.tools]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """获取所有工具描述"""
        return {tool.name: tool.description for tool in self.tools}


def create_network_engineer_agent(
    llm_type: str = "tongyi",
    model_name: str = "qwen-turbo",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> NetworkEngineerAgent:
    """创建网络工程师 Agent 的工厂函数"""
    return NetworkEngineerAgent(
        llm_type=llm_type,
        model_name=model_name,
        api_key=api_key,
        verbose=verbose,
    )


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("网络工程师智能助手")
    print("=" * 60)
    
    agent = NetworkEngineerAgent(verbose=True)
    
    print("\n可用工具列表：")
    for name in agent.get_tool_names():
        print(f"  - {name}")
    
    print("\n" + "=" * 60)
    print("开始对话（输入 'quit' 退出，'clear' 清除记忆）")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("再见！")
                break
            
            if user_input.lower() == "clear":
                agent.clear_memory()
                print("对话记忆已清除")
                continue
            
            response = agent.chat(user_input)
            print(f"\n助手: {response}")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")