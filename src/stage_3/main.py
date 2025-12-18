"""
Phase 3: 架构进化 (Modular & Agentic RAG) - 主入口

演示 Agentic RAG 的完整流程：
1. 智能路由 - 根据问题类型选择处理方式
2. 自反思 RAG - 答案质量自评估和迭代优化
3. 工具调用 - 搜索、计算、代码执行
4. 父子索引 - 精准检索 + 完整上下文
5. 上下文压缩 - 精简无关内容
"""

import sys
from pathlib import Path
from loguru import logger

from src.stage_1.config import Config, get_config
from src.stage_1.document_loader import DocumentLoader
from src.stage_1.vectorstore import VectorStoreManager

from src.stage_2.semantic_chunker import SemanticChunker
from src.stage_2.metadata_extractor import MetadataExtractor

from .config import Stage3Config, get_stage3_config
from .agentic_rag_chain import AgenticRAGChain, AgenticRAGResult
from .router import RouteType


def setup_logger():
    """配置 loguru 日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True
    )


def load_and_index_documents(
    data_path: str,
    config: Config,
    use_semantic_chunking: bool = True,
    force_resplit: bool = False
) -> list:
    """
    加载并索引文档
    
    Args:
        data_path: 文档路径
        config: 配置对象
        use_semantic_chunking: 是否使用语义分块
        force_resplit: 是否强制重新分块
        
    Returns:
        list: 分块后的文档列表
    """
    # 1. 加载文档
    logger.info("📄 开始加载文档...")
    loader = DocumentLoader()
    documents = loader.load(data_path)
    
    if not documents:
        logger.warning("⚠️ 没有找到任何文档")
        return None
    
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
    
    logger.info(f"✅ 加载完成: {len(documents)} 个文档 -> {len(chunks)} 个分块")
    
    return chunks


def print_result(result: AgenticRAGResult, show_details: bool = False):
    """
    打印处理结果
    
    Args:
        result: 处理结果
        show_details: 是否显示详细信息
    """
    print(f"\n🤖 回答:\n{result.answer}\n")
    
    if show_details:
        print("=" * 50)
        print("📊 处理详情:")
        print(f"  - 路由类型: {result.route_type.value}")
        print(f"  - 置信度: {result.confidence:.2f}")
        print(f"  - 迭代次数: {result.iterations}")
        
        if result.tool_used:
            print(f"  - 使用工具: {result.tool_used}")
        
        if result.sources:
            print(f"\n📚 参考来源 ({len(result.sources)} 个):")
            for i, source in enumerate(result.sources[:3], 1):
                print(f"  [{i}] {source['source']}")
                print(f"      {source['content'][:80]}...")
        
        print(f"\n🔄 推理链:")
        for i, step in enumerate(result.reasoning_chain, 1):
            print(f"  {i}. {step}")
        
        print("=" * 50)


def interactive_qa(rag_chain: AgenticRAGChain):
    """
    交互式问答
    
    Args:
        rag_chain: Agentic RAG 链实例
    """
    print("\n" + "=" * 60)
    print("🤖 Agentic RAG 问答系统已启动")
    print("=" * 60)
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("输入 'detail' 切换详细信息模式")
    print("输入 'help' 查看帮助")
    print("-" * 60 + "\n")
    
    show_details = False
    
    while True:
        try:
            question = input("👤 您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见！")
                break
            
            if question.lower() == "detail":
                show_details = not show_details
                status = "开启" if show_details else "关闭"
                print(f"\n📊 详细信息模式已{status}\n")
                continue
            
            if question.lower() == "help":
                print("\n" + "=" * 50)
                print("🔍 系统能力:")
                print("  - 知识库检索: 从文档中查找信息")
                print("  - Web 搜索: 获取实时互联网信息")
                print("  - 数学计算: 执行数学运算")
                print("  - 代码执行: 运行 Python 代码")
                print("  - 直接回答: 通用问题回答")
                print("\n💡 示例问题:")
                print("  - 知识库: 什么是 RAG？")
                print("  - 搜索: 今天的天气怎么样？")
                print("  - 计算: 123 * 456 等于多少？")
                print("  - 代码: 用 Python 生成 10 个随机数")
                print("  - 闲聊: 你好，介绍一下你自己")
                print("=" * 50 + "\n")
                continue
            
            print("\n🤔 正在思考（使用 Agentic RAG）...\n")
            
            result = rag_chain.ask(question)
            print_result(result, show_details)
            
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
    print("🚀 Phase 3: Agentic RAG 架构进化")
    print("=" * 60 + "\n")
    
    # 加载配置
    config = get_config()
    stage3_config = get_stage3_config()
    
    # 验证配置
    if not config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        return
    
    # 解析参数
    import argparse
    parser = argparse.ArgumentParser(description="Agentic RAG 系统")
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
        "--no-routing",
        action="store_true",
        help="禁用智能路由"
    )
    parser.add_argument(
        "--no-self-rag",
        action="store_true",
        help="禁用自反思"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="禁用工具调用"
    )
    parser.add_argument(
        "--no-parent-child",
        action="store_true",
        help="禁用父子索引"
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="禁用上下文压缩"
    )
    args = parser.parse_args()
    
    # 加载文档
    data_path = Path(args.data)
    chunks = None
    vectorstore_manager = VectorStoreManager(config, collection_name="agentic_rag")
    
    if args.reindex or vectorstore_manager.vectorstore._collection.count() == 0:
        if data_path.exists():
            chunks = load_and_index_documents(
                str(data_path),
                config,
                use_semantic_chunking=not args.no_semantic,
                force_resplit=args.reindex
            )
            if chunks is None:
                return
            # 将分块存入向量库
            vectorstore_manager.add_documents(chunks)
        else:
            logger.warning(f"⚠️ 数据路径不存在: {data_path}")
            logger.info("💡 请创建 ./data/documents 目录并放入文档")
            return
    else:
        logger.info(f"📦 使用已有向量库")
        # 尝试从缓存加载分块
        if data_path.exists():
            loader = DocumentLoader()
            docs = loader.load(str(data_path))
            if not args.no_semantic:
                try:
                    chunker = SemanticChunker(config)
                    chunks = chunker.split_documents(docs, use_cache=True)
                except Exception:
                    from src.stage_1.chunker import TextChunker
                    chunker = TextChunker(config.chunk_size, config.chunk_overlap)
                    chunks = chunker.split_documents(docs)
            else:
                from src.stage_1.chunker import TextChunker
                chunker = TextChunker(config.chunk_size, config.chunk_overlap)
                chunks = chunker.split_documents(docs)
    
    # 创建 Agentic RAG 链
    rag_chain = AgenticRAGChain(
        documents=chunks,
        vectorstore_manager=vectorstore_manager,
        config=stage3_config,
        enable_routing=not args.no_routing,
        enable_self_rag=not args.no_self_rag,
        enable_tools=not args.no_tools,
        enable_parent_child=not args.no_parent_child,
        enable_compression=not args.no_compression,
        force_reindex=args.reindex
    )
    
    # 启动交互
    interactive_qa(rag_chain)


if __name__ == "__main__":
    main()

