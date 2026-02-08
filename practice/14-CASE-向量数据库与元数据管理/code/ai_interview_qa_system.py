"""
AI产品经理面试题向量数据库系统
功能：
1. 读取PDF文件并解析65道面试题
2. 使用text-embedding-v4模型（兼容中文模型）生成向量
3. 使用Faiss存储向量数据
4. 实现元数据管理系统
5. 实现查询功能并测试匹配准确度
"""

import os
import re
import json
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer # type: ignore
from ...shared import get_project_path


class AIInterviewQASystem:
    """AI产品经理面试题向量数据库系统"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化系统
        
        Args:
            model_name: 嵌入模型名称（使用中文多语言模型作为text-embedding-v4的替代）
        """
        self.model_name = model_name
        
        # 初始化嵌入模型
        print(f"加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Faiss索引
        self.index = None
        
        # 元数据存储（metadata_store）
        self.metadata_store = []  # 存储所有元数据
        
        # 数据路径
        self.data_dir = get_project_path('data')
        self.model_dir = get_project_path('model')
        self.output_dir = get_project_path('output')
        
        # 确保目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"系统初始化完成，向量维度: {self.dimension}")
    
    def parse_pdf_content(self, pdf_text: str) -> List[Dict[str, Any]]:
        """
        解析PDF文本，提取问题和答案
        
        Args:
            pdf_text: PDF文本内容
            
        Returns:
            解析后的问答数据列表
        """
        qa_pairs = []
        
        # 定义类别映射
        category_patterns = [
            (r'一、个人情况类', '个人情况类'),
            (r'二、个人经历类', '个人经历类'),
            (r'三、产品素质类', '产品素质类'),
            (r'四、经典算法类', '经典算法类'),
            (r'五、深度学习类', '深度学习类'),
            (r'六、大数据模型类', '大数据模型类'),
            (r'七、技术基础类', '技术基础类'),
            (r'八、工作场景类', '工作场景类'),
            (r'九、行业认知类', '行业认知类')
        ]
        
        # 找出所有类别位置
        category_positions = []
        for pattern, category_name in category_patterns:
            matches = list(re.finditer(pattern, pdf_text))
            for match in matches:
                category_positions.append((match.start(), category_name))
        
        # 按位置排序
        category_positions.sort()
        
        # 使用正则表达式匹配问题和答案
        # 改进匹配模式：匹配 (一)问题标题 或 (一)问题
        question_pattern = r'\(([一二三四五六七八九十]+)\)\s*([^\n]+)'
        
        # 查找所有问题
        questions = list(re.finditer(question_pattern, pdf_text))
        
        print(f"找到 {len(questions)} 个问题匹配")
        
        current_category = "未分类"
        question_count = 0
        
        for match in questions:
            # 确定当前类别
            for pos, category_name in category_positions:
                if match.start() >= pos:
                    current_category = category_name
                else:
                    break
            
            question_number = match.group(1)  # 中文数字
            question_text = match.group(2).strip() if match.group(2) else ""  # 问题内容
            
            # 清理问题文本：去除页码、点号、问号
            question_text = re.sub(r'\.{3,}.*?\d+\s*$', '', question_text)  # 去除页码
            question_text = re.sub(r'\s+', ' ', question_text)  # 合并多个空格
            question_text = question_text.strip()
            
            # 如果问题文本为空或太短，跳过
            if not question_text or len(question_text) < 3:
                continue
            
            # 查找答案（在问题之后，下一个问题之前）
            start_pos = match.end()
            next_match = re.search(question_pattern, pdf_text[start_pos:])
            
            if next_match:
                answer_text = pdf_text[start_pos:start_pos + next_match.start()].strip()
            else:
                answer_text = pdf_text[start_pos:].strip()
            
            # 提取"参考答案"部分 - 改进正则表达式
            answer_pattern = r'参考答案[:：]\s*(.*?)(?=(?:\([一二三四五六七八九十]+\)|一、|二、|三、|四、|五、|六、|七、|八、|九、|目录|$))'
            answer_match = re.search(answer_pattern, answer_text, re.DOTALL)
            
            if answer_match:
                answer_content = answer_match.group(1).strip()
            else:
                answer_content = answer_text
            
            # 清理答案文本
            answer_content = re.sub(r'\s+', ' ', answer_content)
            answer_content = re.sub(r'\.{3,}.*?\d+\s*$', '', answer_content)  # 去除页码
            answer_content = answer_content[:5000]  # 限制长度
            
            # 调试信息：显示前5个问答对
            if question_count < 5:
                print(f"\n问答对 {question_count}:")
                print(f"  类别: {current_category}")
                print(f"  问题: {question_text}")
                print(f"  答案长度: {len(answer_content)} 字符")
                if answer_content:
                    print(f"  答案预览: {answer_content[:100]}...")
                else:
                    print(f"  答案: (空)")
            
            # 创建问答对象
            qa_pair = {
                "id": f"qa_{question_count:03d}",
                "category": current_category,
                "question_number": question_number,
                "question_title": "",  # 不再使用question_title
                "question": question_text,
                "answer": answer_content,
                "full_text": f"{question_text} {answer_content}"  # 用于生成嵌入的完整文本
            }
            
            qa_pairs.append(qa_pair)
            question_count += 1
        
        print(f"成功解析 {len(qa_pairs)} 个问答对")
        
        # 打印类别统计
        category_stats = {}
        for qa in qa_pairs:
            cat = qa['category']
            category_stats[cat] = category_stats.get(cat, 0) + 1
        
        print("\n类别统计:")
        for cat, count in category_stats.items():
            print(f"  {cat}: {count} 个问题")
        
        return qa_pairs
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量数组
        """
        print(f"正在生成 {len(texts)} 个文本的嵌入向量...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray):
        """
        创建Faiss索引
        
        Args:
            embeddings: 嵌入向量数组
        """
        print("创建Faiss索引...")
        
        num_vectors = embeddings.shape[0]
        dimension = embeddings.shape[1]
        
        # 使用IndexFlatL2 (L2距离)
        self.index = faiss.IndexFlatL2(dimension)
        
        # 添加向量到索引（使用显式参数名避免类型检查错误）
        self.index.add(embeddings) # type: ignore
        
        # 保存索引
        index_path = os.path.join(self.model_dir, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        
        print(f"Faiss索引创建完成，包含 {num_vectors} 个向量")
        print(f"索引已保存到: {index_path}")
    
    def save_metadata(self, qa_pairs: List[Dict[str, Any]]):
        """
        保存元数据到metadata_store
        
        Args:
            qa_pairs: 问答数据列表
        """
        print("保存元数据到metadata_store...")
        
        # 保存为JSON
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 保存为pickle（便于快速加载）
        pickle_path = os.path.join(self.model_dir, 'metadata.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(qa_pairs, f)
        
        # 存储到metadata_store
        self.metadata_store = qa_pairs
        
        print(f"元数据已保存到metadata_store:")
        print(f"  - JSON: {metadata_path}")
        print(f"  - Pickle: {pickle_path}")
        print(f"  - 内存: {len(self.metadata_store)} 个问答对")
    
    def load_system(self):
        """
        加载已保存的系统（Faiss索引和元数据）
        """
        print("加载已保存的系统...")
        
        # 加载Faiss索引
        index_path = os.path.join(self.model_dir, 'faiss_index.bin')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Faiss索引已加载: {index_path}")
        else:
            print("未找到Faiss索引文件")
        
        # 加载元数据
        pickle_path = os.path.join(self.model_dir, 'metadata.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                self.metadata_store = pickle.load(f)
            
            print(f"元数据已加载到metadata_store: {pickle_path}")
        else:
            print("未找到元数据文件")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        搜索相似问题
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值（默认0.1，适应384维向量的L2距离）
            
        Returns:
            搜索结果列表
        """
        if self.index is None:
            print("错误：索引未初始化，请先加载或创建索引")
            return []
        
        print(f"\n搜索查询: {query}")
        print(f"相似度阈值: {threshold}")
        
        # 生成查询向量
        query_embedding = self.model.encode([query])
        
        # 在Faiss中搜索
        distances, indices = self.index.search(query_embedding, top_k) # type: ignore
        
        print(f"搜索到 {top_k} 个候选结果，正在计算相似度...")
        
        # 转换距离为相似度（L2距离越小，相似度越高）
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 / (1 + distance)  # L2距离转相似度
            
            # 调试信息：显示前10个结果的相似度
            if i < 10:
                print(f"  候选{i+1}: L2距离={distance:.4f}, 相似度={similarity:.4f}, 阈值={threshold}")
            
            if similarity < threshold:
                continue
            
            if idx < len(self.metadata_store):
                qa = self.metadata_store[idx]
                results.append({
                    'rank': i + 1,
                    'similarity': similarity,
                    'distance': distance,
                    'qa_data': qa
                })
        
        print(f"找到 {len(results)} 个相似结果")
        return results
    
    def display_results(self, results: List[Dict[str, Any]]):
        """
        显示搜索结果和元数据
        
        Args:
            results: 搜索结果列表
        """
        print("\n搜索结果:")
        print("=" * 80)
        
        for result in results:
            qa = result['qa_data']
            print(f"\n排名: {result['rank']}")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"距离: {result['distance']:.4f}")
            print(f"元数据:")
            print(f"  - ID: {qa['id']}")
            print(f"  - 类别: {qa['category']}")
            print(f"  - 问题编号: {qa['question_number']}")
            print(f"  - 问题标题: {qa['question_title']}")
            print(f"问题: {qa['question']}")
            print(f"答案摘要: {qa['answer'][:100]}...")
            print("-" * 80)
    
    def export_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        导出搜索结果到文件
        
        Args:
            results: 搜索结果列表
            output_file: 输出文件路径
        """
        export_data = []
        for result in results:
            qa = result['qa_data']
            export_data.append({
                'rank': result['rank'],
                'similarity': float(result['similarity']),
                'distance': float(result['distance']),
                'metadata': {
                    'id': qa['id'],
                    'category': qa['category'],
                    'question_number': qa['question_number'],
                    'question_title': qa['question_title']
                },
                'question': qa['question'],
                'answer': qa['answer']
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"结果已导出到: {output_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("AI产品经理面试题向量数据库系统")
    print("=" * 80)
    
    # 创建系统实例
    system = AIInterviewQASystem()
    
    # 读取PDF文件
    pdf_path = get_project_path('data', 'AI产品经理面试题65道.pdf')
    print(f"\n读取PDF文件: {pdf_path}")
    
    # 检查PDF文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误：PDF文件不存在: {pdf_path}")
        print("请确保PDF文件位于正确的路径")
        return
    
    # 使用PyMuPDF读取PDF
    try:
        import fitz # type: ignore
        doc = fitz.open(pdf_path)
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        doc.close()
        print(f"成功读取PDF文件，共 {len(pdf_text)} 个字符")
    except ImportError:
        print("警告：无法导入PyMuPDF，尝试使用其他方法")
        # 尝试使用pypdf
        try:
            from pypdf import PdfReader # type: ignore
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()
            print(f"成功读取PDF文件，共 {len(pdf_text)} 个字符")
        except ImportError:
            print("错误：无法读取PDF文件，请安装PyMuPDF或pypdf")
            print("运行: uv add pymupdf 或 uv add pypdf")
            return
    
    # 解析PDF内容
    print("\n" + "=" * 80)
    print("解析PDF内容")
    print("=" * 80)
    qa_pairs = system.parse_pdf_content(pdf_text)
    
    if not qa_pairs:
        print("未找到任何问答对")
        return
    
    # 生成嵌入向量
    print("\n" + "=" * 80)
    print("生成嵌入向量")
    print("=" * 80)
    texts = [qa['full_text'] for qa in qa_pairs]
    embeddings = system.generate_embeddings(texts)
    
    # 创建Faiss索引
    print("\n" + "=" * 80)
    print("创建Faiss索引")
    print("=" * 80)
    system.create_faiss_index(embeddings)
    
    # 保存元数据
    print("\n" + "=" * 80)
    print("保存元数据到metadata_store")
    print("=" * 80)
    system.save_metadata(qa_pairs)
    
    # 保存系统配置
    print("\n" + "=" * 80)
    print("保存系统配置")
    print("=" * 80)
    config = {
        'model_name': system.model_name,
        'dimension': system.dimension,
        'total_qa_pairs': len(qa_pairs),
        'faiss_index_path': os.path.join(system.model_dir, 'faiss_index.bin'),
        'metadata_path': os.path.join(system.model_dir, 'metadata.json'),
        'created_at': '2026-01-21'
    }
    
    # 修复：使用 system.model_dir 而不是 self.model_dir
    config_path = os.path.join(system.model_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"系统配置已保存到: {config_path}")
    
    # 显示系统摘要
    print("\n" + "=" * 80)
    print("系统构建完成！")
    print("=" * 80)
    print(f"总问答对数: {len(qa_pairs)}")
    print(f"向量维度: {system.dimension}")
    print(f"Faiss索引: {os.path.join(system.model_dir, 'faiss_index.bin')}")
    print(f"元数据: {os.path.join(system.model_dir, 'metadata.json')}")
    print(f"配置文件: {config_path}")
    
    # 测试查询功能
    print("\n" + "=" * 80)
    print("测试查询功能")
    print("=" * 80)
    
    test_queries = [
        "如何进行自我介绍？",
        "AI产品经理和传统产品经理有什么区别？",
        "什么是机器学习？",
        "KNN算法的优缺点是什么？",
        "如何处理离职原因？"
    ]
    
    for query in test_queries:
        results = system.search(query, top_k=3)
        system.display_results(results)
        
        # 导出结果
        output_file = os.path.join(system.output_dir, f"query_{query[:10]}_results.json")
        system.export_results(results, output_file)
    
    print("\n" + "=" * 80)
    print("所有任务完成！")
    print("=" * 80)
    print("\n使用方法：")
    print("1. 重新加载系统：system.load_system()")
    print("2. 搜索问题：system.search('你的问题', top_k=5)")
    print("3. 显示结果：system.display_results(results)")
    print("4. 导出结果：system.export_results(results, 'output.json')")


if __name__ == "__main__":
    main()