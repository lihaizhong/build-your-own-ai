"""
环境测试脚本 - 验证三国演义 Embedding 项目的依赖安装
"""

def test_environment():
    """测试所有关键依赖是否正确安装"""
    print("=" * 60)
    print("三国演义 Embedding 项目 - 环境测试")
    print("=" * 60)
    print()

    # 测试 Python 版本
    import sys
    print(f"✅ Python 版本: {sys.version}")
    print()

    # 测试核心库
    try:
        import numpy as np
        print(f"✅ NumPy 版本: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy 导入失败: {e}")

    try:
        import pandas as pd
        print(f"✅ Pandas 版本: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas 导入失败: {e}")

    try:
        import faiss
        print(f"✅ FAISS 版本: {faiss.__version__}")
    except ImportError as e:
        print(f"❌ FAISS 导入失败: {e}")

    try:
        import sentence_transformers
        print(f"✅ Sentence-Transformers 版本: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ Sentence-Transformers 导入失败: {e}")

    try:
        import jieba
        print(f"✅ Jieba 版本: {jieba.__version__}")
    except ImportError as e:
        print(f"❌ Jieba 导入失败: {e}")

    try:
        import sklearn
        print(f"✅ Scikit-learn 版本: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn 导入失败: {e}")

    try:
        import matplotlib
        print(f"✅ Matplotlib 版本: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib 导入失败: {e}")

    try:
        import seaborn
        print(f"✅ Seaborn 版本: {seaborn.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn 导入失败: {e}")

    print()
    print("=" * 60)
    print("功能测试")
    print("=" * 60)
    print()

    # 测试中文分词
    try:
        import jieba
        text = "三国演义是中国古典四大名著之一"
        words = jieba.lcut(text)
        print(f"✅ 中文分词测试: {text}")
        print(f"   分词结果: {words}")
    except Exception as e:
        print(f"❌ 中文分词测试失败: {e}")

    print()

    # 测试 Embedding 生成
    try:
        from sentence_transformers import SentenceTransformer
        print("⏳ 正在加载 Embedding 模型...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        text = "三国演义是中国古典四大名著之一"
        embedding = model.encode(text)
        print(f"✅ Embedding 生成测试: {text}")
        print(f"   向量维度: {embedding.shape}")
    except Exception as e:
        print(f"❌ Embedding 生成测试失败: {e}")

    print()

    # 测试 FAISS 索引创建
    try:
        import faiss
        import numpy as np
        print("⏳ 正在创建 FAISS 索引...")
        dimension = 384  # paraphrase-multilingual-MiniLM-L12-v2 的向量维度
        index = faiss.IndexFlatL2(dimension)
        test_vector = np.random.random((1, dimension)).astype('float32')
        index.add(test_vector)
        print(f"✅ FAISS 索引创建测试")
        print(f"   索引维度: {dimension}")
        print(f"   向量数量: {index.ntotal}")
    except Exception as e:
        print(f"❌ FAISS 索引创建测试失败: {e}")

    print()
    print("=" * 60)
    print("环境测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_environment()