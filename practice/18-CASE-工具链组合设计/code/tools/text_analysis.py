"""
文本分析工具 - TextAnalysisTool

功能：
- 统计文本字数、字符数
- 情感分析
- 关键词提取
- 语言检测
"""

import re
from collections import Counter
from typing import Dict, List, Any


class TextAnalysisTool:
    """文本分析工具类"""
    
    def __init__(self):
        self.name = "文本分析工具"
        self.description = (
            "分析文本内容的工具。"
            "可以统计字数、字符数，进行情感分析，提取关键词。"
            "输入：需要分析的文本内容。"
            "输出：包含各项分析结果的字典。"
        )
    
    def run(self, text: str) -> str:
        """
        运行文本分析
        
        Args:
            text: 待分析的文本内容
            
        Returns:
            分析结果字符串
        """
        if not text or not isinstance(text, str):
            return "错误：请提供有效的文本内容"
        
        try:
            result = self._analyze(text)
            return self._format_result(result)
        except Exception as e:
            return f"分析过程中出现错误：{str(e)}"
    
    def _analyze(self, text: str) -> Dict[str, Any]:
        """执行文本分析"""
        # 基础统计
        char_count = len(text)
        char_count_no_space = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        
        # 中文和英文分别统计
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_words = re.findall(r'[a-zA-Z]+', text)
        numbers = re.findall(r'\d+', text)
        
        # 行数统计
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 段落统计（以空行分隔）
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # 关键词提取（简单实现：提取出现频率高的中文词）
        keywords = self._extract_keywords(text)
        
        # 情感分析（简单实现：基于关键词）
        sentiment = self._simple_sentiment(text)
        
        return {
            "字符总数": char_count,
            "字符数(不含空白)": char_count_no_space,
            "中文字符数": len(chinese_chars),
            "英文单词数": len(english_words),
            "数字个数": len(numbers),
            "总行数": len(lines),
            "非空行数": len(non_empty_lines),
            "段落数": len(paragraphs),
            "高频关键词": keywords,
            "情感倾向": sentiment,
        }
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """提取关键词（简单实现）"""
        # 提取中文词组（2-4个字符）
        chinese_pattern = r'[\u4e00-\u9fff]{2,4}'
        words = re.findall(chinese_pattern, text)
        
        # 过滤常见停用词
        stopwords = {'的', '是', '在', '了', '和', '与', '或', '等', '及', '中', '对', '为'}
        words = [w for w in words if w not in stopwords]
        
        # 统计频率
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def _simple_sentiment(self, text: str) -> str:
        """简单情感分析"""
        positive_words = ['好', '优秀', '成功', '正常', '稳定', '快速', '高效', '完美']
        negative_words = ['错误', '失败', '异常', '问题', '故障', '慢', '崩溃', '超时']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "积极"
        elif negative_count > positive_count:
            return "消极"
        else:
            return "中性"
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """格式化输出结果"""
        lines = ["📊 文本分析结果："]
        lines.append("-" * 40)
        
        for key, value in result.items():
            if isinstance(value, list):
                value_str = "、".join(value) if value else "无"
            else:
                value_str = str(value)
            lines.append(f"  {key}：{value_str}")
        
        return "\n".join(lines)


# 用于 LangChain Tool 包装的工厂函数
def create_text_analysis_tool():
    """创建 LangChain Tool 实例"""
    from langchain_core.tools import Tool
    
    tool = TextAnalysisTool()
    return Tool(
        name=tool.name,
        func=tool.run,
        description=tool.description
    )
