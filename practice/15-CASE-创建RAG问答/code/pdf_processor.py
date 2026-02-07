
"""
PDF处理器模块
用于从PDF文件中提取文本并记录页码信息
"""

import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document



class PDFProcessor:
    """PDF文档处理器，支持文本提取和页码记录"""
    
    def __init__(self, pdf_path: str):
        """
        初始化PDF处理器
        
        Args:
            pdf_path: PDF文件路径
        """
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        
    def load_and_process(self) -> List[Document]:
        """
        加载PDF并处理文档，为每个文档块添加页码信息
        
        Returns:
            处理后的文档列表，每个文档包含页码元数据
        """
        try:
            # 加载PDF文档
            raw_documents = self.loader.load()
            print(f"成功加载 {len(raw_documents)} 个文档块")
            
            # 为每个文档添加页码信息
            processed_documents = []
            for doc in raw_documents:
                # 创建新的文档对象，保留原始内容但添加页码
                page_number = doc.metadata.get('page', 0)
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'page_number': page_number,
                        'source': os.path.basename(self.pdf_path)
                    }
                )
                processed_documents.append(new_doc)
            
            return processed_documents
            
        except Exception as e:
            print(f"处理PDF时出错: {str(e)}")
            raise
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        获取文档统计信息
        
        Returns:
            包含文档统计信息的字典
        """
        try:
            raw_documents = self.loader.load()
            total_pages = len(set(doc.metadata.get('page', 0) for doc in raw_documents))
            total_chars = sum(len(doc.page_content) for doc in raw_documents)
            
            return {
                'total_documents': len(raw_documents),
                'total_pages': total_pages,
                'total_characters': total_chars,
                'pdf_file': os.path.basename(self.pdf_path)
            }
        except Exception as e:
            print(f"获取文档统计信息时出错: {str(e)}")
            return {}
    
    def save_processed_documents(self, output_path: str):
        """
        保存处理后的文档到文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            documents = self.load_and_process()
            
            # 创建输出目录（如果不存在）
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存文档信息
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"PDF文档处理结果 - {os.path.basename(self.pdf_path)}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, doc in enumerate(documents, 1):
                    page_num = doc.metadata.get('page_number', '未知')
                    f.write(f"文档块 {i} (页码: {page_num}):\n")
                    f.write("-" * 30 + "\n")
                    f.write(doc.page_content[:500] + "...\n\n")
            
            print(f"处理结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存处理结果时出错: {str(e)}")
            raise


def process_pdf_with_page_numbers(pdf_path: str) -> List[Document]:
    """
    处理PDF文件并返回带页码的文档列表
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        带页码信息的文档列表
    """
    processor = PDFProcessor(pdf_path)
    return processor.load_and_process()


if __name__ == "__main__":
    # 示例用法
    pdf_path = "data/AI产品经理面试题65道.pdf"
    if os.path.exists(pdf_path):
        processor = PDFProcessor(pdf_path)
        stats = processor.get_document_stats()
        print("文档统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 保存处理结果
        output_path = "output/processed_documents.txt"
        processor.save_processed_documents(output_path)
    else:
        print(f"PDF文件不存在: {pdf_path}")
