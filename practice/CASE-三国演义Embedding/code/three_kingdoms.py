"""
三国演义 Word2Vec 词嵌入分析
使用Gensim训练Word2Vec模型，分析人物关系
"""

import os
from typing import Optional
import jieba
from gensim.models import Word2Vec
from collections import Counter


def get_project_path(*paths: str) -> str:
    """
    获取项目路径的统一方法
    类似TypeScript中的path.join(__dirname, ...paths)
    
    Args:
        *paths: 路径组件
        
    Returns:
        完整的项目路径
    """
    try:
        # __file__ 类似于 TypeScript 中的 __filename
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录(向上一级)
        project_dir = os.path.dirname(current_dir)
        
        return os.path.join(project_dir, *paths)
    except NameError:
        # 在某些环境下(如Jupyter)__file__不可用,使用当前工作目录
        return os.path.join(os.getcwd(), *paths)

class SanguoWord2VecAnalyzer:
    """三国演义Word2Vec分析器"""
    
    def __init__(self, data_path: str) -> None:
        """
        初始化分析器
        
        Args:
            data_path: 三国演义文本文件路径
        """
        self.data_path: str = data_path
        self.sentences: list[list[str]] = []
        self.model: Optional[Word2Vec] = None
    
    def get_model_path(self, model_name: str = 'three_kingdoms.model') -> str:
        """
        获取模型保存路径
        
        Args:
            model_name: 模型文件名
            
        Returns:
            模型文件的完整路径
        """
        return get_project_path('model', model_name)
        
    def load_and_preprocess(self) -> None:
        """加载和预处理文本数据"""
        print("=" * 60)
        print("1. 加载三国演义文本")
        print("=" * 60)
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"文本总长度: {len(text)} 字符")
        
        # 分词处理
        print("\n正在分词...")
        words = jieba.lcut(text)
        print(f"分词结果: {len(words)} 个词")
        
        # 构建句子（每10个词作为一个句子）
        print("\n构建训练句子...")
        for i in range(0, len(words) - 10, 10):
            sentence = words[i:i+10]
            self.sentences.append(sentence)
        
        print(f"总句子数: {len(self.sentences)}")
        
        # 统计词频
        print("\n词频统计（前20）:")
        word_counts = Counter(words)
        for word, count in word_counts.most_common(20):
            print(f"  {word}: {count}")
        
    def train_word2vec(self) -> None:
        """训练Word2Vec模型"""
        print("\n" + "=" * 60)
        print("2. 训练Word2Vec模型")
        print("=" * 60)
        
        # 训练参数
        params: dict[str, int] = {
            'vector_size': 100,      # 向量维度
            'window': 5,              # 上下文窗口大小
            'min_count': 5,           # 最小词频
            'workers': 4,             # 并行数
            'sg': 0,                  # 0=CBOW, 1=Skip-gram
            'epochs': 100,            # 训练轮数
            'seed': 42                # 随机种子
        }
        
        print(f"训练参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 训练模型
        print("\n开始训练...")
        self.model = Word2Vec(self.sentences, **params)  # type: ignore[arg-type]
        
        print(f"训练完成！")
        print(f"词汇表大小: {len(self.model.wv)}")
        print(f"向量维度: {self.model.wv.vector_size}")
        
    def analyze_similar_words(self, target_word: str, topn: int = 10) -> list[tuple[str, float]]:
        """
        分析与目标词最相似的词
        
        Args:
            target_word: 目标词
            topn: 返回最相似的词数量
            
        Returns:
            相似词列表，每个元素为(词, 相似度)的元组
        """
        print("\n" + "=" * 60)
        print(f"3. 分析与'{target_word}'最相近的词")
        print("=" * 60)
        
        if self.model is None or target_word not in self.model.wv:
            print(f"警告: '{target_word}' 不在词汇表中")
            return []
        
        similar_words = self.model.wv.most_similar(target_word, topn=topn)
        
        print(f"\n与 '{target_word}' 最相近的 {topn} 个词:")
        for i, (word, similarity) in enumerate(similar_words, 1):
            print(f"  {i}. {word}: {similarity:.4f}")
        
        return similar_words
    
    def word_analogy(self, positive_words: list[str], negative_words: list[str], topn: int = 5) -> list[tuple[str, float]]:
        """
        词类比计算
        
        Args:
            positive_words: 正面词列表
            negative_words: 负面词列表
            topn: 返回结果数量
            
        Returns:
            类似词列表，每个元素为(词, 相似度)的元组
        """
        print("\n" + "=" * 60)
        print("4. 词类比计算")
        print("=" * 60)
        
        pos_str = " + ".join(positive_words)
        neg_str = " - ".join(negative_words)
        print(f"\n计算: {pos_str} - {neg_str} = ?")
        
        if self.model is None:
            print("错误: 模型未加载")
            return []
        
        try:
            result = self.model.wv.most_similar(
                positive=positive_words,
                negative=negative_words,
                topn=topn
            )
            
            print(f"\n结果:")
            for i, (word, similarity) in enumerate(result, 1):
                print(f"  {i}. {word}: {similarity:.4f}")
            
            return result
        except KeyError as e:
            print(f"错误: 词汇表中缺少词 - {e}")
            return []
    
    def save_model(self, model_path: Optional[str] = None) -> None:
        """
        保存模型
        
        Args:
            model_path: 模型保存路径，如果为None则使用默认路径
        """
        if model_path is None:
            model_path = self.get_model_path()
        
        # 确保模型目录存在
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\n保存模型到: {model_path}")
        if self.model is not None:
            self.model.save(model_path)
        print("模型保存完成！")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
        """
        if model_path is None:
            model_path = self.get_model_path()
        
        print(f"\n从 {model_path} 加载模型")
        self.model = Word2Vec.load(model_path)
        print("模型加载完成！")


def main() -> None:
    """主函数"""
    # 使用 get_project_path 构建路径
    data_path = get_project_path('data', 'three_kingdoms.txt')
    model_path = get_project_path('model', 'three_kingdoms.model')
    
    print(f"数据路径: {data_path}")
    print(f"模型路径: {model_path}")
    
    # 创建分析器
    analyzer = SanguoWord2VecAnalyzer(data_path)
    
    # 加载和预处理数据
    analyzer.load_and_preprocess()
    
    # 训练Word2Vec模型
    analyzer.train_word2vec()
    
    # 保存模型
    analyzer.save_model(model_path)
    
    # 分析与曹操最相近的词
    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)
    
    similar_to_caocao = analyzer.analyze_similar_words('曹操', topn=15)
    
    print(f"\n相似词结果已保存到变量 similar_to_caocao")
    print(f"共找到 {len(similar_to_caocao)} 个相似词")
    
    # 词类比: 曹操 + 刘备 - 张飞 = ?
    analogy_result = analyzer.word_analogy(
        positive_words=['曹操', '刘备'],
        negative_words=['张飞'],
        topn=10
    )
    
    print(f"\n词类比结果已保存到变量 analogy_result")
    print(f"共找到 {len(analogy_result)} 个类比结果")
    
    # 更多有趣的词类比
    print("\n" + "=" * 60)
    print("更多词类比分析")
    print("=" * 60)
    
    # 诸葛亮 + 刘备 - 关羽 = ?
    analyzer.word_analogy(
        positive_words=['诸葛亮', '刘备'],
        negative_words=['关羽'],
        topn=5
    )
    
    # 曹操 - 董卓 = ?
    analyzer.word_analogy(
        positive_words=['曹操'],
        negative_words=['董卓'],
        topn=5
    )
    
    # 刘备 + 关羽 - 张飞 = ?
    analyzer.word_analogy(
        positive_words=['刘备', '关羽'],
        negative_words=['张飞'],
        topn=5
    )
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()