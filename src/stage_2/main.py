"""
Phase 2: 质量飞跃 (Advanced RAG) - 主入口

演示 Advanced RAG 的完整流程：
1. 语义分块
2. 元数据提取
3. 混合检索
4. 查询重写
5. 重排序
"""

import sys
from pathlib import Path
from loguru import logger

from src.stage_1.config import Config, get_config
from src.stage_1.document_loader import DocumentLoader
from src.stage_1.vectorstore import VectorStoreManager

from .semantic_chunker import SemanticChunker
from .metadata_extractor import MetadataExtractor
from .hybrid_retriever import HybridRetriever
from .query_rewriter import QueryRewriter
from .reranker import Reranker
from .advanced_rag_chain import AdvancedRAGChain


def setup_logger():
    """配置 loguru 日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True
    )


def load_and_index_documents_advanced(
    data_path: str,
    config: Config,
    use_semantic_chunking: bool = True,
    force_resplit: bool = False
) -> tuple:
    """
    加载并索引文档（使用高级功能）
    
    Args:
        data_path: 文档路径
        config: 配置对象
        use_semantic_chunking: 是否使用语义分块
        force_resplit: 是否强制重新分块（忽略缓存）
        
    Returns:
        tuple: (文档列表, 向量存储管理器)
    """
    # 1. 加载文档
    logger.info("📄 开始加载文档...")
    loader = DocumentLoader()
    documents = loader.load(data_path)
    
    if not documents:
        logger.warning("⚠️ 没有找到任何文档")
        return None, None
    
    # 2. 元数据提取
    logger.info("📋 提取元数据...")
    metadata_extractor = MetadataExtractor()
    documents = metadata_extractor.enrich_documents(documents)
    
    # 3. 分块处理
    if use_semantic_chunking:
        logger.info("🧠 使用语义分块...")
        try:
            chunker = SemanticChunker(config)
            chunks = chunker.split_documents(
                documents, 
                use_cache=True, 
                force_resplit=force_resplit
            )
        except Exception as e:
            logger.warning(f"⚠️ 语义分块失败，回退到固定分块: {e}")
            from src.stage_1.chunker import TextChunker
            chunker = TextChunker(config.chunk_size, config.chunk_overlap)
            chunks = chunker.split_documents(
                documents, 
                use_cache=True,
                force_resplit=force_resplit
            )
    else:
        logger.info("✂️ 使用固定分块...")
        from src.stage_1.chunker import TextChunker
        chunker = TextChunker(config.chunk_size, config.chunk_overlap)
        chunks = chunker.split_documents(
            documents, 
            use_cache=True,
            force_resplit=force_resplit
        )
    
    # 4. 存入向量库
    logger.info("🗄️ 向量化存储...")
    vectorstore_manager = VectorStoreManager(config, collection_name="advanced_rag")
    vectorstore_manager.add_documents(chunks)
    
    return chunks, vectorstore_manager


def interactive_qa_advanced(rag_chain: AdvancedRAGChain):
    """
    高级交互式问答
    
    Args:
        rag_chain: 高级 RAG 链实例
    """
    print("\n" + "=" * 60)
    print("🚀 Advanced RAG 问答系统已启动")
    print("=" * 60)
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("输入 'detail' 切换详细信息模式")
    print("-" * 60 + "\n")
    
    show_detail = False
    
    while True:
        try:
            question = input("👤 您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见！")
                break
            
            if question.lower() == "detail":
                show_detail = not show_detail
                status = "开启" if show_detail else "关闭"
                print(f"\n📊 详细信息模式已{status}\n")
                continue
            
            print("\n🤔 正在思考（使用高级检索）...\n")
            
            if show_detail:
                result = rag_chain.ask_with_details(question)
                print(f"🤖 回答:\n{result['answer']}\n")
                print("📊 检索详情:")
                print(f"  - 生成的查询: {len(result['queries'])} 个")
                for i, q in enumerate(result['queries'][:3]):
                    print(f"    [{i+1}] {q[:50]}...")
                print(f"  - 检索文档: {result['retrieved_docs']} 个")
                print(f"  - 重排序后: {result['reranked_docs']} 个\n")
                
                if result['sources']:
                    print("📚 Top-3 来源:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  [{i}] {source['source']} (分数: {source['score']:.4f})")
                        print(f"      {source['content'][:80]}...\n")
            else:
                answer = rag_chain.ask(question)
                print(f"🤖 回答:\n{answer}\n")
            
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            logger.error(f"❌ 发生错误: {e}")
            print(f"\n❌ 发生错误: {e}\n")


def main():
    """主入口函数"""
    setup_logger()
    
    print("\n" + "=" * 60)
    print("🚀 Phase 2: Advanced RAG 质量飞跃")
    print("=" * 60 + "\n")
    
    # 加载配置
    config = get_config()
    
    # 验证配置
    if not config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        return
    
    # 解析参数
    import argparse
    parser = argparse.ArgumentParser(description="Advanced RAG 系统")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/documents",
        help="文档路径（文件或目录）"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="强制重新索引文档"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="禁用语义分块"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="禁用重排序"
    )
    args = parser.parse_args()
    
    # 加载文档
    data_path = Path(args.data)
    chunks = None
    vectorstore_manager = VectorStoreManager(config, collection_name="advanced_rag")
    
    if args.reindex or vectorstore_manager.vectorstore._collection.count() == 0:
        if data_path.exists():
            chunks, vectorstore_manager = load_and_index_documents_advanced(
                str(data_path),
                config,
                use_semantic_chunking=not args.no_semantic,
                force_resplit=args.reindex
            )
            if chunks is None:
                return
        else:
            logger.warning(f"⚠️ 数据路径不存在: {data_path}")
            logger.info("💡 请创建 ./data/documents 目录并放入文档")
            return
    else:
        logger.info(f"📦 使用已有向量库")
        # 尝试从缓存加载已有的语义分块
        if data_path.exists():
            loader = DocumentLoader()
            docs = loader.load(str(data_path))
            # 优先使用语义分块缓存
            if not args.no_semantic:
                chunker = SemanticChunker(config)
                chunks = chunker.split_documents(docs, use_cache=True)
                logger.info(f"✅ 已加载 {len(chunks)} 个语义分块")
            else:
                from src.stage_1.chunker import TextChunker
                chunker = TextChunker(config.chunk_size, config.chunk_overlap)
                chunks = chunker.split_documents(docs)
    
    # 创建高级 RAG 链
    rag_chain = AdvancedRAGChain(
        documents=chunks,
        vectorstore_manager=vectorstore_manager,
        config=config,
        use_query_rewrite=True,
        use_hybrid_search=chunks is not None,
        use_reranking=not args.no_rerank
    )
    
    # 启动交互
    interactive_qa_advanced(rag_chain)


if __name__ == "__main__":
    main()
