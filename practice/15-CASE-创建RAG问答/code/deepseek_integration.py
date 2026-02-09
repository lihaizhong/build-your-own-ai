"""DashScope API集成模块
用于在RAG系统中集成DashScope大模型和嵌入模型
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi

load_dotenv(verbose=True)

class DashScopeIntegration:
    """DashScope API集成管理器"""
    
    def __init__(self):
        """初始化DashScope集成"""
        
        # 获取API密钥
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        
        # 验证必要密钥
        if not self.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    
    def get_llm(self, model_name: str = "qwen-turbo", temperature: float = 0.0):
        """
        获取DashScope LLM实例
        
        Args:
            model_name: 模型名称,默认为"qwen-turbo"
            temperature: 温度参数,控制输出随机性
            
        Returns:
            ChatTongyi实例
        """
        print(f"使用DashScope模型: {model_name}")
        return ChatTongyi(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature # type: ignore
        )
    
    def get_embeddings(self, model_name: str = "text-embedding-v2"):
        """
        获取嵌入模型实例
        
        Args:
            model_name: 嵌入模型名称
            
        Returns:
            嵌入模型实例
        """
        print(f"使用DashScope嵌入模型: {model_name}")
        return DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=self.api_key
        ) # type: ignore
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        获取配置信息
        
        Returns:
            包含配置信息的字典
        """
        return {
            'dashscope_api_key_set': bool(self.api_key),
            'using_dashscope_llm': True,
            'using_dashscope_embeddings': True
        }
    
    def validate_api_keys(self) -> bool:
        """
        验证API密钥是否有效
        
        Returns:
            验证结果
        """
        try:
            # 测试LLM连接
            llm = self.get_llm()
            test_response = llm.invoke("Hello, this is a test message.")
            print("DashScope LLM API连接成功")
            
            # 测试嵌入模型
            embeddings = self.get_embeddings()
            test_embedding = embeddings.embed_query("Hello, this is a test message.")
            print("DashScope嵌入模型连接成功")
            
            return True
            
        except Exception as e:
            print(f"API密钥验证失败: {str(e)}")
            return False


# 全局集成实例（延迟初始化）
_dashscope_integration = None


def get_dashscope_llm(model_name: str = "qwen-turbo", temperature: float = 0.0):
    """
    获取DashScope LLM的便捷函数
    
    Args:
        model_name: 模型名称
        temperature: 温度参数
        
    Returns:
        LLM实例
    """
    global _dashscope_integration
    if _dashscope_integration is None:
        _dashscope_integration = DashScopeIntegration()
    return _dashscope_integration.get_llm(model_name, temperature)


def get_embeddings(model_name: str = "text-embedding-v2"):
    """
    获取嵌入模型的便捷函数
    
    Args:
        model_name: 嵌入模型名称
        
    Returns:
        嵌入模型实例
    """
    global _dashscope_integration
    if _dashscope_integration is None:
        _dashscope_integration = DashScopeIntegration()
    return _dashscope_integration.get_embeddings(model_name)


def get_integration_info() -> Dict[str, Any]:
    """
    获取集成信息的便捷函数
    
    Returns:
        配置信息字典
    """
    global _dashscope_integration
    if _dashscope_integration is None:
        _dashscope_integration = DashScopeIntegration()
    return _dashscope_integration.get_config_info()


if __name__ == "__main__":
    # 测试集成
    try:
        info = get_integration_info()
        print("DashScope集成配置:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 验证API密钥
        if _dashscope_integration and _dashscope_integration.validate_api_keys():
            print("\n所有API密钥验证成功！")
        else:
            print("\nAPI密钥验证失败,请检查.env文件中的配置。")
            
    except Exception as e:
        print(f"初始化DashScope集成时出错: {str(e)}")