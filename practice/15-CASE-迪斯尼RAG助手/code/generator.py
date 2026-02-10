"""
生成层模块
Step4: 上下文组织和结构化提示
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI

from .config import config
from .retrieval import RetrievalResult


class PromptBuilder:
    """提示词构建器"""
    
    def __init__(self):
        """初始化提示词构建器"""
        self.system_prompt = self._build_system_prompt()
        logger.info("提示词构建器初始化完成")
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的迪士尼知识问答助手，能够基于提供的文档和图像信息回答用户的问题。

你的职责：
1. 基于提供的上下文信息准确回答用户问题
2. 如果上下文中没有相关信息，明确说明你不知道
3. 回答要清晰、简洁、准确
4. 对于图像相关信息，可以描述图像内容
5. 保持专业和友好的态度

注意事项：
- 优先使用提供的上下文信息
- 不要编造不存在的信息
- 如果信息不确定，可以适当提及不确定性
- 可以引用多个上下文片段来支持回答
"""
    
    def build_context(
        self, 
        results: Dict[str, List[RetrievalResult]]
    ) -> str:
        """
        构建上下文文本
        
        Args:
            results: 检索结果字典
        
        Returns:
            上下文文本
        """
        context_parts = []
        
        # 添加文本上下文
        if results.get("text"):
            context_parts.append("【相关文档信息】")
            for i, result in enumerate(results["text"], 1):
                context_parts.append(
                    f"\n文档{i} (来源: {result.source}, 相似度: {result.score:.3f}):\n"
                    f"{result.content}"
                )
        
        # 添加图像上下文
        if results.get("image"):
            context_parts.append("\n【相关图像信息】")
            for i, result in enumerate(results["image"], 1):
                context_parts.append(
                    f"\n图像{i} (来源: {result.source}, 相似度: {result.score:.3f}):\n"
                    f"{result.content}"
                )
        
        return "\n".join(context_parts)
    
    def build_prompt(
        self, 
        query: str, 
        results: Dict[str, List[RetrievalResult]],
        include_empty_context: bool = False
    ) -> str:
        """
        构建完整提示词
        
        Args:
            query: 用户查询
            results: 检索结果
            include_empty_context: 是否在没有结果时仍然包含上下文部分
        
        Returns:
            完整提示词
        """
        context = self.build_context(results)
        
        if not context and include_empty_context:
            context = "【相关文档信息】\n\n没有找到相关的文档信息。"
        elif not context:
            context = "没有找到相关的文档或图像信息。"
        
        prompt = f"""{context}

用户问题：{query}

请基于以上信息回答用户问题。"""
        
        return prompt
    
    def build_stream_prompt(
        self, 
        query: str, 
        results: Dict[str, List[RetrievalResult]]
    ) -> Dict[str, str]:
        """
        构建流式提示词（分别返回系统提示和用户提示）
        
        Args:
            query: 用户查询
            results: 检索结果
        
        Returns:
            {"system": 系统提示, "user": 用户提示}
        """
        context = self.build_context(results)
        
        if not context:
            context = "没有找到相关的文档或图像信息。"
        
        user_prompt = f"""上下文信息：
{context}

用户问题：{query}

请基于以上信息回答用户问题。"""
        
        return {
            "system": self.system_prompt,
            "user": user_prompt
        }


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化答案生成器
        
        Args:
            api_key: API密钥
            model: 模型名称
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model or config.llm_model
        
        # 初始化OpenAI客户端（兼容DashScope）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.prompt_builder = PromptBuilder()
        
        logger.info(f"答案生成器初始化完成 (模型: {self.model})")
    
    def generate(
        self, 
        query: str, 
        results: Dict[str, List[RetrievalResult]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        生成答案
        
        Args:
            query: 用户查询
            results: 检索结果
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            生成的答案
        """
        temperature = temperature or config.llm_temperature
        max_tokens = max_tokens or config.llm_max_tokens
        
        # 构建提示词
        prompt_dict = self.prompt_builder.build_stream_prompt(query, results)
        
        logger.info(f"生成答案: '{query}' (温度: {temperature}, 最大tokens: {max_tokens})")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_dict["system"]},
                    {"role": "user", "content": prompt_dict["user"]}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.info("答案生成完成")
            
            return answer
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return "抱歉，生成答案时出现了错误。"
    
    def generate_stream(
        self, 
        query: str, 
        results: Dict[str, List[RetrievalResult]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        流式生成答案
        
        Args:
            query: 用户查询
            results: 检索结果
            temperature: 温度参数
            max_tokens: 最大token数
        
        Yields:
            生成的文本片段
        """
        temperature = temperature or config.llm_temperature
        max_tokens = max_tokens or config.llm_max_tokens
        
        # 构建提示词
        prompt_dict = self.prompt_builder.build_stream_prompt(query, results)
        
        logger.info(f"流式生成答案: '{query}'")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_dict["system"]},
                    {"role": "user", "content": prompt_dict["user"]}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            logger.info("流式答案生成完成")
            
        except Exception as e:
            logger.error(f"流式答案生成失败: {e}")
            yield "抱歉，生成答案时出现了错误。"


class RAGPipeline:
    """RAG流程管道"""
    
    def __init__(
        self, 
        retriever,  # HybridRetriever
        generator: Optional[AnswerGenerator] = None
    ):
        """
        初始化RAG管道
        
        Args:
            retriever: 检索器
            generator: 生成器
        """
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()
        self.prompt_builder = PromptBuilder()
        
        logger.info("RAG管道初始化完成")
    
    def query(
        self, 
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_retrieval_results: bool = False
    ) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            top_k: 检索结果数量
            score_threshold: 分数阈值
            temperature: 温度参数
            max_tokens: 最大token数
            return_retrieval_results: 是否返回检索结果
        
        Returns:
            {"answer": 答案, "context": 上下文, "retrieval_results": 检索结果}
        """
        logger.info(f"RAG查询: '{question}'")
        
        # Step 1: 检索
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        # Step 2: 构建上下文
        context = self.prompt_builder.build_context(retrieval_results)
        
        # Step 3: 生成答案
        answer = self.generator.generate(
            query=question,
            results=retrieval_results,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = {
            "answer": answer,
            "context": context,
            "question": question
        }
        
        if return_retrieval_results:
            result["retrieval_results"] = retrieval_results
        
        logger.info("RAG查询完成")
        
        return result
    
    def query_stream(
        self, 
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        流式执行RAG查询
        
        Args:
            question: 用户问题
            top_k: 检索结果数量
            score_threshold: 分数阈值
            temperature: 温度参数
            max_tokens: 最大token数
        
        Yields:
            生成的文本片段
        """
        logger.info(f"流式RAG查询: '{question}'")
        
        # Step 1: 检索
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        # Step 2: 流式生成答案
        for chunk in self.generator.generate_stream(
            query=question,
            results=retrieval_results,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            yield chunk
        
        logger.info("流式RAG查询完成")
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        格式化响应
        
        Args:
            result: 查询结果
        
        Returns:
            格式化的响应字符串
        """
        formatted = []
        formatted.append(f"问题: {result['question']}")
        formatted.append(f"\n答案:\n{result['answer']}")
        
        if result.get('context') and result['context'].strip():
            formatted.append(f"\n\n参考上下文:\n{result['context']}")
        
        return "\n".join(formatted)


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    from .config import load_env_config
    from .retrieval import HybridRetriever
    from .embedding import VectorStore
    
    logger.add("logs/generator.log", rotation="1 day")
    
    # 加载环境变量
    load_env_config()
    
    # 加载索引
    vector_store = VectorStore()
    vector_store.load_indexes()
    
    # 创建检索器和RAG管道
    retriever = HybridRetriever(vector_store)
    rag_pipeline = RAGPipeline(retriever)
    
    # 测试查询
    questions = [
        "迪士尼有哪些经典动画电影？",
        "迪士尼乐园在哪里？",
        "米老鼠是什么时候创造的？"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        result = rag_pipeline.query(question)
        print(rag_pipeline.format_response(result))
