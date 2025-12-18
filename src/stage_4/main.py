"""
Stage 4 主入口

演示 GraphRAG 和微调功能。
"""

import os
import sys
from pathlib import Path

from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage_1.document_loader import DocumentLoader
from src.stage_1.chunker import TextChunker
from src.stage_4.config import get_stage4_config
from src.stage_4.graph_rag import GraphRAGChain
from src.stage_4.ultimate_rag_chain import UltimateRAGChain, RetrievalMode


def load_documents(data_dir: str = "./data/documents") -> list:
    """加载文档"""
    loader = DocumentLoader()
    documents = loader.load_directory(data_dir)
    logger.info(f"加载了 {len(documents)} 个文档")
    return documents


def chunk_documents(documents: list) -> list:
    """分块文档"""
    chunker = TextChunker()
    chunks = chunker.split_documents(documents)
    logger.info(f"生成了 {len(chunks)} 个文档块")
    return chunks


def demo_graph_rag():
    """
    演示 GraphRAG 功能
    """
    print("\n" + "=" * 60)
    print("🎯 GraphRAG 演示")
    print("=" * 60 + "\n")
    
    # 加载文档
    documents = load_documents()
    if not documents:
        print("❌ 没有找到文档，请在 data/documents 目录下放置文档")
        return
    
    # 分块
    chunks = chunk_documents(documents)
    
    # 初始化 GraphRAG
    print("\n📊 初始化 GraphRAG...")
    config = get_stage4_config()
    
    graph_rag = GraphRAGChain(
        documents=chunks,
        config=config,
        graph_name="demo_graph",
        force_rebuild=True,  # 演示时强制重建
    )
    
    # 构建知识图谱
    print("\n🔨 构建知识图谱...")
    graph_rag.build_knowledge_graph(chunks)
    
    # 显示图谱统计
    stats = graph_rag.get_statistics()
    print(f"\n📈 知识图谱统计:")
    print(f"   - 实体数量: {stats['num_nodes']}")
    print(f"   - 关系数量: {stats['num_edges']}")
    print(f"   - 实体类型: {stats.get('entity_type_counts', {})}")
    print(f"   - 关系类型: {stats.get('relation_type_counts', {})}")
    
    # 生成全局摘要
    print("\n📝 生成全局摘要...")
    summary = graph_rag.generate_global_summary()
    print(f"\n{summary}")
    
    # 交互式问答
    print("\n" + "-" * 60)
    print("💬 进入交互式问答模式（输入 'quit' 退出）")
    print("-" * 60)
    
    while True:
        question = input("\n您的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not question:
            continue
        
        result = graph_rag.ask(question)
        
        print(f"\n📚 答案:")
        print(result.answer)
        
        if result.matched_entities:
            print(f"\n🔍 相关实体: {[e['name'] for e in result.matched_entities]}")
        
        if result.related_relations:
            print(f"🔗 相关关系: {len(result.related_relations)} 条")


def demo_ultimate_rag():
    """
    演示终极 RAG 功能
    """
    print("\n" + "=" * 60)
    print("🚀 终极 RAG 演示")
    print("=" * 60 + "\n")
    
    # 加载文档
    documents = load_documents()
    if not documents:
        print("❌ 没有找到文档，请在 data/documents 目录下放置文档")
        return
    
    # 分块
    chunks = chunk_documents(documents)
    
    # 初始化终极 RAG
    print("\n🎯 初始化终极 RAG...")
    config = get_stage4_config()
    
    ultimate_rag = UltimateRAGChain(
        documents=chunks,
        config=config,
        graph_name="ultimate_demo",
        enable_routing=True,
        enable_self_rag=True,
        enable_graph_rag=True,
        enable_reranking=True,
        enable_compression=True,
        force_rebuild_graph=True,
    )
    
    # 显示统计
    stats = ultimate_rag.get_statistics()
    print(f"\n📈 系统组件状态:")
    for component, enabled in stats["components"].items():
        status = "✅" if enabled else "❌"
        print(f"   {status} {component}")
    
    if "knowledge_graph" in stats:
        kg_stats = stats["knowledge_graph"]
        print(f"\n📊 知识图谱:")
        print(f"   - 实体: {kg_stats['num_nodes']}")
        print(f"   - 关系: {kg_stats['num_edges']}")
    
    # 交互式问答
    print("\n" + "-" * 60)
    print("💬 进入交互式问答模式")
    print("   - 输入 'quit' 退出")
    print("   - 输入 'vector' 切换到向量检索")
    print("   - 输入 'graph' 切换到图检索")
    print("   - 输入 'fusion' 切换到融合检索")
    print("   - 输入 'auto' 切换到自动模式")
    print("-" * 60)
    
    current_mode = RetrievalMode.AUTO
    
    while True:
        question = input(f"\n[{current_mode.value}] 您的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        # 模式切换
        if question.lower() in ['vector', 'graph', 'fusion', 'auto']:
            current_mode = RetrievalMode(question.lower())
            print(f"🔄 已切换到 {current_mode.value} 模式")
            continue
        
        if not question:
            continue
        
        result = ultimate_rag.ask(question, retrieval_mode=current_mode)
        
        print(f"\n📚 答案:")
        print(result.answer)
        
        print(f"\n📊 详细信息:")
        print(f"   - 检索模式: {result.retrieval_mode.value}")
        print(f"   - 置信度: {result.confidence:.2f}")
        print(f"   - 质量分数: {result.quality_score:.2f}")
        
        if result.retrieved_docs:
            print(f"   - 检索文档: {len(result.retrieved_docs)} 个")
        
        if result.graph_entities:
            print(f"   - 匹配实体: {[e['name'] for e in result.graph_entities]}")
        
        if result.reasoning_chain:
            print(f"\n🔍 推理链:")
            for step in result.reasoning_chain:
                print(f"   → {step}")


def demo_fine_tuning():
    """
    演示微调数据生成功能
    """
    print("\n" + "=" * 60)
    print("📚 微调数据生成演示")
    print("=" * 60 + "\n")
    
    # 加载文档
    documents = load_documents()
    if not documents:
        print("❌ 没有找到文档，请在 data/documents 目录下放置文档")
        return
    
    # 分块
    chunks = chunk_documents(documents)[:10]  # 只用前10个块演示
    
    from src.stage_4.fine_tuning import (
        EmbeddingFineTuner,
        LLMFineTuner,
        TrainingDataGenerator,
    )
    
    config = get_stage4_config()
    
    # 1. 生成 LLM 微调数据
    print("\n📝 生成 LLM 微调数据...")
    llm_finetuner = LLMFineTuner(config=config)
    qa_pairs = llm_finetuner.generate_qa_pairs(chunks, pairs_per_doc=3)
    
    print(f"   生成了 {len(qa_pairs)} 个 QA 对")
    
    # 显示示例
    if qa_pairs:
        print("\n   示例 QA 对:")
        for i, pair in enumerate(qa_pairs[:3], 1):
            print(f"\n   [{i}] Q: {pair.question[:100]}...")
            print(f"       A: {pair.answer[:100]}...")
    
    # 保存
    llm_finetuner.export_jsonl()
    llm_finetuner.export_json()
    print(f"\n   数据已保存到: {llm_finetuner.output_dir}")
    
    # 2. 生成 Embedding 训练数据
    print("\n📊 生成 Embedding 训练数据...")
    emb_finetuner = EmbeddingFineTuner(config=config)
    pairs, triplets = emb_finetuner.generate_training_data(chunks)
    
    print(f"   生成了 {len(pairs)} 个训练对, {len(triplets)} 个三元组")
    
    # 保存
    emb_finetuner.save_training_data()
    print(f"   数据已保存到: {emb_finetuner.output_dir}")
    
    # 3. 统计信息
    print("\n📈 统计信息:")
    stats = llm_finetuner.get_statistics()
    print(f"   QA 对总数: {stats['total_pairs']}")
    print(f"   难度分布: {stats['difficulty_distribution']}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🎯 UltimateRAG Stage 4: GraphRAG & Fine-tuning")
    print("=" * 60)
    
    print("\n请选择演示模式:")
    print("1. GraphRAG 演示")
    print("2. 终极 RAG 演示")
    print("3. 微调数据生成演示")
    print("4. 退出")
    
    while True:
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == "1":
            demo_graph_rag()
        elif choice == "2":
            demo_ultimate_rag()
        elif choice == "3":
            demo_fine_tuning()
        elif choice == "4":
            print("👋 再见！")
            break
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    main()

