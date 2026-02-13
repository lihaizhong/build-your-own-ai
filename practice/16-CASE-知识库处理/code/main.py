#!/usr/bin/env python3
"""
知识库处理主程序
提供命令行接口运行各种知识库处理功能
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from .config import config
from .question_generator import KnowledgeBaseOptimizer
from .conversation_extractor import ConversationKnowledgeExtractor
from .health_checker import KnowledgeBaseHealthChecker
from .version_manager import KnowledgeBaseVersionManager
from .utils import save_json, load_json


# 示例知识库（迪士尼主题乐园）
SAMPLE_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "kb_001",
        "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区：米奇大街、奇想花园、探险岛、宝藏湾、明日世界、梦幻世界和迪士尼小镇。",
        "category": "基本信息",
        "last_updated": "2024-01-15"
    },
    {
        "id": "kb_002", 
        "content": "上海迪士尼乐园的门票价格根据季节和日期有所不同。平日成人票价为399元，周末和节假日为499元。儿童票（1.0-1.4米）平日为299元，周末和节假日为374元。1.0米以下儿童免费入园。",
        "category": "价格信息",
        "last_updated": "2024-01-01"
    },
    {
        "id": "kb_003",
        "content": "上海迪士尼乐园的营业时间通常为上午8:00至晚上8:00，但具体时间会根据季节和特殊活动进行调整。建议游客在出发前查看官方网站或APP获取最新的营业时间信息。",
        "category": "营业信息",
        "last_updated": "2024-01-20"
    },
    {
        "id": "kb_004",
        "content": "从上海市区到上海迪士尼乐园有多种交通方式：1. 地铁11号线迪士尼站下车；2. 乘坐迪士尼专线巴士；3. 打车约40-60分钟；4. 自驾车可停在乐园停车场，停车费为100元/天。",
        "category": "交通信息",
        "last_updated": "2024-01-10"
    },
    {
        "id": "kb_005",
        "content": "上海迪士尼乐园的特色项目包括：创极速光轮（明日世界）、七个小矮人矿山车（梦幻世界）、加勒比海盗：战争之潮（宝藏湾）、翱翔·飞越地平线（探险岛）等。这些项目都有不同的身高和年龄限制。",
        "category": "游乐项目",
        "last_updated": "2024-02-01"
    },
    {
        "id": "kb_006",
        "content": "上海迪士尼乐园提供多种餐饮选择，包括米奇大街的皇家宴会厅、奇想花园的漫月轩、宝藏湾的巴波萨烧烤等。园内餐厅价格相对较高，人均消费约150-300元。建议游客可以携带密封包装的零食和水入园。",
        "category": "餐饮信息",
        "last_updated": "2024-01-25"
    }
]

# 示例对话
SAMPLE_CONVERSATIONS = [
    """
用户: "我想去上海迪士尼乐园玩，门票多少钱？"
AI: "上海迪士尼乐园的门票价格根据日期有所不同。平日成人票价为399元，周末和节假日为499元。儿童票（1.0-1.4米）平日为299元，周末为374元。1.0米以下儿童免费。"
用户: "需要提前预订吗？"
AI: "建议提前预订，特别是周末和节假日，可以通过官方网站或第三方平台预订。"
用户: "从浦东机场怎么去？"
AI: "从浦东机场到迪士尼乐园可以乘坐地铁2号线到广兰路站，然后换乘11号线到迪士尼站，全程约1小时。也可以打车，约40分钟。"
""",
    """
用户: "迪士尼乐园今天开放吗？"
AI: "上海迪士尼乐园通常每天开放，营业时间为上午8:00至晚上8:00。但建议您出发前查看官方网站或APP确认具体营业时间，因为可能会因特殊活动或维护调整。"
用户: "有什么必玩的项目？"
AI: "上海迪士尼乐园的必玩项目包括：创极速光轮（明日世界）、七个小矮人矿山车（梦幻世界）、加勒比海盗：战争之潮（宝藏湾）、翱翔·飞越地平线（探险岛）等。"
"""
]

# 测试查询
TEST_QUERIES = [
    {"query": "上海迪士尼乐园在哪里？", "expected_answer": "浦东新区"},
    {"query": "门票多少钱？", "expected_answer": "价格"},
    {"query": "营业时间是什么？", "expected_answer": "8:00"},
    {"query": "怎么去迪士尼？", "expected_answer": "地铁"},
    {"query": "有什么好玩的项目？", "expected_answer": "项目"},
    {"query": "可以带食物进去吗？", "expected_answer": "零食"}
]


def run_question_generation(knowledge_base: List[Dict[str, Any]]) -> None:
    """运行问题生成与检索优化演示"""
    print("\n" + "="*60)
    print("知识库问题生成与检索优化（BM25版本）")
    print("="*60 + "\n")
    
    optimizer = KnowledgeBaseOptimizer()
    
    # 示例1: 为知识切片生成问题
    print("示例1: 为知识切片生成多样化问题")
    test_chunk = knowledge_base[0]['content']
    print(f"知识内容: {test_chunk[:100]}...")
    
    questions = optimizer.generate_questions_for_chunk(test_chunk, num_questions=5)
    print(f"\n生成的5个问题:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['question']} (类型: {q['question_type']}, 难度: {q['difficulty']})")
    
    print("\n" + "-"*40 + "\n")
    
    # 示例2: 评估检索方法
    print("示例2: 评估两种检索方法的准确度")
    
    # 为知识库生成问题
    print('正在为知识库生成问题...')
    for chunk in knowledge_base:
        chunk['generated_questions'] = optimizer.generate_questions_for_chunk(
            chunk['content']
        )
    print('为知识库生成问题完毕')
    
    # 评估
    results = optimizer.evaluate_retrieval_methods(
        knowledge_base, 
        TEST_QUERIES[:3]  # 使用前3个测试查询
    )
    
    print(f"测试查询数量: {len(TEST_QUERIES[:3])}")
    print(f"BM25原文检索准确率: {sum(results['content_similarity'])/len(results['content_similarity'])*100:.1f}%")
    print(f"BM25问题检索准确率: {sum(results['question_similarity'])/len(results['question_similarity'])*100:.1f}%")


def run_conversation_extraction(conversations: List[str]) -> None:
    """运行对话知识提取演示"""
    print("\n" + "="*60)
    print("对话知识提取与沉淀")
    print("="*60 + "\n")
    
    extractor = ConversationKnowledgeExtractor()
    
    # 示例1: 从单次对话中提取知识
    print("示例1: 从单次对话中提取知识")
    conversation = conversations[0]
    
    extracted = extractor.extract_knowledge_from_conversation(conversation)
    print(f"\n提取的知识点:")
    for i, knowledge in enumerate(extracted['extracted_knowledge'], 1):
        print(f"  {i}. 类型: {knowledge['knowledge_type']}")
        print(f"     内容: {knowledge['content'][:50]}...")
        print(f"     置信度: {knowledge['confidence']}")
    
    print(f"\n对话摘要: {extracted['conversation_summary']}")
    print(f"用户意图: {extracted['user_intent']}")
    
    print("\n" + "-"*40 + "\n")
    
    # 示例2: 批量提取并合并知识
    print("示例2: 批量提取知识并合并")
    all_knowledge = extractor.batch_extract_knowledge(conversations)
    print(f"总共提取了 {len(all_knowledge)} 个知识点")
    
    merged_knowledge = extractor.merge_similar_knowledge(all_knowledge)
    print(f"合并后剩余 {len(merged_knowledge)} 个知识点")


def run_health_check(
    knowledge_base: List[Dict[str, Any]], 
    test_queries: List[Dict[str, Any]]
) -> None:
    """运行健康度检查演示"""
    print("\n" + "="*60)
    print("知识库健康度检查")
    print("="*60 + "\n")
    
    checker = KnowledgeBaseHealthChecker()
    
    # 生成健康度报告
    health_report = checker.generate_health_report(knowledge_base, test_queries)
    
    # 显示报告
    print(f"整体健康度评分: {health_report['overall_health_score']:.2f}")
    print(f"健康等级: {health_report['health_level']}")
    print(f"检查时间: {health_report['check_date']}")
    
    print("\n" + "-"*40 + "\n")
    
    # 详细分析
    print("详细分析:")
    
    # 1. 缺少的知识
    missing = health_report['missing_knowledge']
    print(f"\n1. 缺少的知识分析:")
    print(f"   覆盖率: {missing.get('coverage_score', 0)*100:.1f}%")
    print(f"   缺少知识点数量: {len(missing.get('missing_knowledge', []))}")
    
    # 2. 过期的知识
    outdated = health_report['outdated_knowledge']
    print(f"\n2. 过期的知识分析:")
    print(f"   新鲜度评分: {outdated.get('freshness_score', 0):.2f}")
    print(f"   过期知识点数量: {len(outdated.get('outdated_knowledge', []))}")
    
    # 3. 冲突的知识
    conflicting = health_report['conflicting_knowledge']
    print(f"\n3. 冲突的知识分析:")
    print(f"   一致性评分: {conflicting.get('consistency_score', 0):.2f}")
    print(f"   冲突数量: {len(conflicting.get('conflicting_knowledge', []))}")
    
    print("\n" + "-"*40 + "\n")
    
    # 改进建议
    print("改进建议:")
    for i, recommendation in enumerate(health_report['recommendations'], 1):
        print(f"  {i}. {recommendation}")


def run_version_management(knowledge_base: List[Dict[str, Any]]) -> None:
    """运行版本管理演示"""
    print("\n" + "="*60)
    print("知识库版本管理与性能比较")
    print("="*60 + "\n")
    
    version_manager = KnowledgeBaseVersionManager()
    
    # 创建版本1（基础版本）
    knowledge_base_v1 = knowledge_base[:3]  # 前3个
    
    # 创建版本2（增强版本）
    knowledge_base_v2 = knowledge_base.copy()
    
    # 功能1: 创建版本
    print("功能1: 创建知识库版本")
    v1_info = version_manager.create_version(knowledge_base_v1, "v1.0", "基础版本")
    v2_info = version_manager.create_version(knowledge_base_v2, "v2.0", "增强版本")
    
    print(f"版本1信息:")
    print(f"  版本名: {v1_info['version_name']}")
    print(f"  描述: {v1_info['description']}")
    print(f"  知识切片数量: {v1_info['statistics']['total_chunks']}")
    
    print(f"\n版本2信息:")
    print(f"  版本名: {v2_info['version_name']}")
    print(f"  描述: {v2_info['description']}")
    print(f"  知识切片数量: {v2_info['statistics']['total_chunks']}")
    
    print("\n" + "-"*40 + "\n")
    
    # 功能2: 版本比较
    print("功能2: 版本差异比较")
    comparison = version_manager.compare_versions("v1.0", "v2.0")
    
    changes = comparison['changes']
    print(f"  新增知识切片: {len(changes['added_chunks'])}个")
    print(f"  删除知识切片: {len(changes['removed_chunks'])}个")
    print(f"  修改知识切片: {len(changes['modified_chunks'])}个")
    
    print("\n" + "-"*40 + "\n")
    
    # 功能3: 性能评估
    print("功能3: 版本性能评估")
    
    perf_v1 = version_manager.evaluate_version_performance("v1.0", TEST_QUERIES)
    perf_v2 = version_manager.evaluate_version_performance("v2.0", TEST_QUERIES)
    
    print(f"版本1性能:")
    print(f"  准确率: {perf_v1['overall_metrics']['accuracy']*100:.1f}%")
    print(f"  平均响应时间: {perf_v1['overall_metrics']['avg_response_time']*1000:.1f}ms")
    
    print(f"\n版本2性能:")
    print(f"  准确率: {perf_v2['overall_metrics']['accuracy']*100:.1f}%")
    print(f"  平均响应时间: {perf_v2['overall_metrics']['avg_response_time']*1000:.1f}ms")
    
    print("\n" + "-"*40 + "\n")
    
    # 功能4: 性能比较与建议
    print("功能4: 性能比较与建议")
    perf_comparison = version_manager.compare_version_performance(
        "v1.0", "v2.0", TEST_QUERIES
    )
    
    comp = perf_comparison['performance_comparison']
    print(f"  准确率提升: {comp['accuracy']['improvement']*100:.1f}%")
    print(f"  建议: {perf_comparison['recommendation']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="知识库处理工具 - 迪士尼RAG助手"
    )
    parser.add_argument(
        "--question", "-q",
        action="store_true",
        help="运行问题生成与检索优化"
    )
    parser.add_argument(
        "--conversation", "-c",
        action="store_true", 
        help="运行对话知识提取"
    )
    parser.add_argument(
        "--health", "-H",
        action="store_true",
        help="运行知识库健康度检查"
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="运行知识库版本管理"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="运行所有功能演示"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="指定知识库数据文件路径（JSON格式）"
    )
    
    args = parser.parse_args()
    
    # 加载知识库数据
    if args.data:
        data_path = Path(args.data)
        if data_path.exists():
            knowledge_base = load_json(data_path)
            logger.info(f"已加载知识库数据: {data_path}")
        else:
            logger.warning(f"数据文件不存在: {data_path}，使用示例数据")
            knowledge_base = SAMPLE_KNOWLEDGE_BASE
    else:
        knowledge_base = SAMPLE_KNOWLEDGE_BASE
    
    # 如果没有指定任何功能，显示帮助
    if not (args.question or args.conversation or args.health or args.version or args.all):
        parser.print_help()
        return
    
    print("="*60)
    print("知识库处理工具 - 迪士尼RAG助手")
    print("="*60)
    
    # 运行选定的功能
    if args.question or args.all:
        run_question_generation(knowledge_base)
    
    if args.conversation or args.all:
        run_conversation_extraction(SAMPLE_CONVERSATIONS)
    
    if args.health or args.all:
        run_health_check(knowledge_base, TEST_QUERIES)
    
    if args.version or args.all:
        run_version_management(knowledge_base)
    
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)


if __name__ == "__main__":
    main()
