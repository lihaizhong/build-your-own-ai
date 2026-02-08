"""
Word2Vec 模型使用示例
展示如何加载已训练的模型并进行词向量分析
"""

from gensim.models import Word2Vec
from ...shared import get_project_path


def load_model():
    """
    加载已训练的Word2Vec模型
    """
    model_path = get_project_path('model', 'three_kingdoms.model')
    print(f"从 {model_path} 加载模型...")
    model = Word2Vec.load(model_path)
    print(f"模型加载成功！词汇表大小: {len(model.wv)}")
    return model


def analyze_word_similarity(model, word, topn=10):
    """
    分析词的相似度
    
    Args:
        model: Word2Vec模型
        word: 目标词
        topn: 返回最相似的词数量
    """
    if word not in model.wv:
        print(f"警告: '{word}' 不在词汇表中")
        return
    
    print(f"\n与 '{word}' 最相近的 {topn} 个词:")
    similar_words = model.wv.most_similar(word, topn=topn)
    for i, (w, sim) in enumerate(similar_words, 1):
        print(f"  {i}. {w}: {sim:.4f}")


def word_analogy_demo(model):
    """
    词类比演示
    """
    print("\n" + "=" * 60)
    print("词类比演示")
    print("=" * 60)
    
    # 示例1: 曹操 + 刘备 - 张飞 = ?
    print("\n示例1: 曹操 + 刘备 - 张飞 = ?")
    result = model.wv.most_similar(
        positive=['曹操', '刘备'],
        negative=['张飞'],
        topn=5
    )
    for i, (word, sim) in enumerate(result, 1):
        print(f"  {i}. {word}: {sim:.4f}")
    
    # 示例2: 诸葛亮 + 刘备 - 关羽 = ?
    print("\n示例2: 诸葛亮 + 刘备 - 关羽 = ?")
    result = model.wv.most_similar(
        positive=['诸葛亮', '刘备'],
        negative=['关羽'],
        topn=5
    )
    for i, (word, sim) in enumerate(result, 1):
        print(f"  {i}. {word}: {sim:.4f}")
    
    # 示例3: 曹操 - 董卓 = ?
    print("\n示例3: 曹操 - 董卓 = ?")
    result = model.wv.most_similar(
        positive=['曹操'],
        negative=['董卓'],
        topn=5
    )
    for i, (word, sim) in enumerate(result, 1):
        print(f"  {i}. {word}: {sim:.4f}")


def word_vector_demo(model):
    """
    词向量演示
    """
    print("\n" + "=" * 60)
    print("词向量演示")
    print("=" * 60)
    
    # 获取词向量
    word = '曹操'
    if word in model.wv:
        vector = model.wv[word]
        print(f"\n'{word}' 的词向量:")
        print(f"  向量维度: {len(vector)}")
        print(f"  向量前10维: {vector[:10]}")
        print(f"  向量范数: {vector_norm(vector):.4f}")
    
    # 计算词之间的余弦相似度
    print("\n词之间的相似度:")
    word_pairs = [
        ('曹操', '刘备'),
        ('曹操', '董卓'),
        ('诸葛亮', '刘备'),
        ('关羽', '张飞')
    ]
    
    for word1, word2 in word_pairs:
        if word1 in model.wv and word2 in model.wv:
            similarity = model.wv.similarity(word1, word2)
            print(f"  {word1} - {word2}: {similarity:.4f}")


def vector_norm(vector):
    """
    计算向量的L2范数
    """
    import numpy as np
    return np.linalg.norm(vector)


def vocabulary_info(model):
    """
    词汇表信息
    """
    print("\n" + "=" * 60)
    print("词汇表信息")
    print("=" * 60)
    
    print(f"\n词汇表大小: {len(model.wv)}")
    print(f"向量维度: {model.wv.vector_size}")
    
    # 统计人物名
    person_names = ['曹操', '刘备', '孙权', '诸葛亮', '关羽', '张飞', 
                    '周瑜', '赵云', '马超', '黄忠', '董卓', '吕布',
                    '袁绍', '孙策', '荀彧', '司马懿']
    
    print(f"\n常见人物词:")
    for name in person_names:
        if name in model.wv:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (不在词汇表中)")


def main():
    """主函数"""
    print("=" * 60)
    print("三国演义 Word2Vec 模型使用示例")
    print("=" * 60)
    
    # 加载模型
    model = load_model()
    
    # 词汇表信息
    vocabulary_info(model)
    
    # 词相似度分析
    print("\n" + "=" * 60)
    print("词相似度分析")
    print("=" * 60)
    
    analyze_word_similarity(model, '曹操', topn=10)
    analyze_word_similarity(model, '刘备', topn=10)
    analyze_word_similarity(model, '诸葛亮', topn=10)
    
    # 词向量演示
    word_vector_demo(model)
    
    # 词类比演示
    word_analogy_demo(model)
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()