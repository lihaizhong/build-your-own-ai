"""
文本处理工具 - TextProcessingTool

功能：
- 文本清洗（去除特殊字符、空白等）
- 文本分割（按段落、句子分割）
- 正则表达式处理
- 日志文本解析
"""

import re
from typing import List


class TextProcessingTool:
    """文本处理工具类"""
    
    def __init__(self):
        self.name = "文本处理工具"
        self.description = (
            "文本处理工具。"
            "支持文本清洗、分割、正则匹配等操作。"
            "输入：格式为 '处理类型|文本内容' 或 '处理类型|参数|文本内容'。"
            "输出：处理后的文本或错误信息。"
        )
    
    def run(self, input_str: str) -> str:
        """
        运行文本处理
        
        Args:
            input_str: 格式为 "处理类型|文本内容" 或 "处理类型|参数|文本内容"
            
        Returns:
            处理结果字符串
        """
        if not input_str:
            return self._show_usage()
        
        try:
            parts = input_str.split("|")
            if len(parts) < 2:
                return self._show_usage()
            
            process_type = parts[0].strip().lower()
            
            if len(parts) == 2:
                data = parts[1]
                return self._process(process_type, data)
            else:
                param = parts[1].strip()
                data = "|".join(parts[2:])
                return self._process_with_param(process_type, param, data)
        except Exception as e:
            return f"处理过程中出现错误：{str(e)}"
    
    def _show_usage(self) -> str:
        """显示使用说明"""
        return """📋 文本处理工具使用说明：
格式：处理类型|文本内容 或 处理类型|参数|文本内容

支持的处理类型：
  clean       - 清洗文本（去除多余空白、特殊字符）
  split_line  - 按行分割
  split_para  - 按段落分割
  regex       - 正则匹配（需提供正则表达式参数）
  extract_ip  - 提取所有 IP 地址
  extract_url - 提取所有 URL
  extract_email - 提取所有邮箱地址
  extract_time - 提取时间戳
  lowercase   - 转小写
  uppercase   - 转大写
  remove_digits - 移除数字
  remove_punctuation - 移除标点符号

示例：
  clean|  Hello   World!  
  extract_ip|服务器日志：192.168.1.1 连接失败，10.0.0.1 正常
  regex|\d{4}-\d{2}-\d{2}|日志日期：2024-01-15""" # type: ignore
    
    def _process(self, process_type: str, data: str) -> str:
        """执行处理（无参数）"""
        processors = {
            "clean": self._clean_text,
            "split_line": self._split_by_line,
            "split_para": self._split_by_paragraph,
            "extract_ip": self._extract_ips,
            "extract_url": self._extract_urls,
            "extract_email": self._extract_emails,
            "extract_time": self._extract_timestamps,
            "lowercase": lambda x: x.lower(),
            "uppercase": lambda x: x.upper(),
            "remove_digits": lambda x: re.sub(r'\d+', '', x),
            "remove_punctuation": lambda x: re.sub(r'[^\w\s]', '', x),
        }
        
        if process_type not in processors:
            return f"不支持的处理类型：{process_type}\n{self._show_usage()}"
        
        result = processors[process_type](data)
        
        if isinstance(result, list):
            return "\n".join(f"  {i+1}. {item}" for i, item in enumerate(result))
        return result
    
    def _process_with_param(self, process_type: str, param: str, data: str) -> str:
        """执行处理（带参数）"""
        if process_type == "regex":
            return self._regex_match(param, data)
        else:
            return f"处理类型 '{process_type}' 不支持额外参数\n{self._show_usage()}"
    
    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        # 去除多余的空白
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空白
        text = text.strip()
        # 去除特殊控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text
    
    def _split_by_line(self, text: str) -> List[str]:
        """按行分割"""
        return [line for line in text.split('\n') if line.strip()]
    
    def _split_by_paragraph(self, text: str) -> List[str]:
        """按段落分割"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _extract_ips(self, text: str) -> List[str]:
        """提取 IP 地址"""
        ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        return re.findall(ipv4_pattern, text)
    
    def _extract_urls(self, text: str) -> List[str]:
        """提取 URL"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)
    
    def _extract_emails(self, text: str) -> List[str]:
        """提取邮箱地址"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def _extract_timestamps(self, text: str) -> List[str]:
        """提取时间戳"""
        patterns = [
            r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',  # ISO 格式
            r'\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2}',      # 常见格式
            r'\d{2}:\d{2}:\d{2}',                          # 时间
            r'\d{4}-\d{2}-\d{2}',                          # 日期
        ]
        results = []
        for pattern in patterns:
            results.extend(re.findall(pattern, text))
        return results
    
    def _regex_match(self, pattern: str, text: str) -> str:
        """正则匹配"""
        try:
            matches = re.findall(pattern, text)
            if matches:
                return "\n".join(f"  {i+1}. {m}" for i, m in enumerate(matches))
            return "未找到匹配项"
        except re.error as e:
            return f"正则表达式错误：{str(e)}"


def create_text_processing_tool():
    """创建 LangChain Tool 实例"""
    from langchain_core.tools import Tool
    
    tool = TextProcessingTool()
    return Tool(
        name=tool.name,
        func=tool.run,
        description=tool.description
    )
