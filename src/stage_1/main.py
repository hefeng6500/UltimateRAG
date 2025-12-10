"""
Phase 1: 原型验证 (MVP) - 主入口

这是 RAG 系统的演示入口，展示完整的问答流程：
1. 加载文档
2. 分块处理
3. 向量化存储
4. 问答交互
"""

import sys
from pathlib import Path
from loguru import logger

from .config import Config, get_config
from .document_loader import DocumentLoader
from .chunker import TextChunker
from .vectorstore import VectorStoreManager
from .rag_chain import RAGChain


def setup_logger():
    """配置 loguru 日志"""
    logger.remove()  # 移除默认处理器
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True
    )


def load_and_index_documents(
    data_path: str,
    config: Config
) -> VectorStoreManager:
    """
    加载并索引文档
    
    Args:
        data_path: 文档路径（文件或目录）
        config: 配置对象
        
    Returns:
        VectorStoreManager: 已索引的向量存储管理器
    """
    # 1. 加载文档
    logger.info("📄 开始加载文档...")
    loader = DocumentLoader()
    documents = loader.load(data_path)
    
    if not documents:
        logger.warning("⚠️ 没有找到任何文档")
        return None
    
    # 2. 分块处理
    logger.info("✂️ 开始分块处理...")
    chunker = TextChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = chunker.split_documents(documents)
    
    # 3. 存入向量库
    logger.info("🗄️ 开始向量化存储...")
    vectorstore_manager = VectorStoreManager(config)
    vectorstore_manager.add_documents(chunks)
    
    return vectorstore_manager


def interactive_qa(rag_chain: RAGChain):
    """
    交互式问答
    
    Args:
        rag_chain: RAG 问答链实例
    """
    print("\n" + "=" * 60)
    print("🤖 RAG 问答系统已启动")
    print("=" * 60)
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("输入 'sources' 可切换显示/隐藏来源模式")
    print("-" * 60 + "\n")
    
    show_sources = False
    
    while True:
        try:
            question = input("👤 您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见！")
                break
            
            if question.lower() == "sources":
                show_sources = not show_sources
                status = "开启" if show_sources else "关闭"
                print(f"\n📋 来源显示已{status}\n")
                continue
            
            print("\n🤔 正在思考...\n")
            
            if show_sources:
                result = rag_chain.ask_with_sources(question)
                print(f"🤖 回答:\n{result['answer']}\n")
                print("📚 参考来源:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  [{i}] {source['source']}")
                    print(f"      {source['content'][:100]}...\n")
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
    print("🚀 Phase 1: RAG 原型验证 (MVP)")
    print("=" * 60 + "\n")
    
    # 加载配置
    config = get_config()
    
    # 验证配置
    if not config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        return
    
    # 检查是否需要索引文档
    import argparse
    parser = argparse.ArgumentParser(description="RAG 原型验证系统")
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
    args = parser.parse_args()
    
    # 初始化向量存储
    vectorstore_manager = VectorStoreManager(config)
    
    # 检查向量库是否为空或需要重新索引
    data_path = Path(args.data)
    if args.reindex or vectorstore_manager.vectorstore._collection.count() == 0:
        if data_path.exists():
            vectorstore_manager = load_and_index_documents(str(data_path), config)
            if vectorstore_manager is None:
                return
        else:
            logger.warning(f"⚠️ 数据路径不存在: {data_path}")
            logger.info("💡 请创建 ./data/documents 目录并放入文档，或使用 --data 参数指定路径")
            return
    else:
        logger.info(f"📦 使用已有向量库")
    
    # 创建 RAG 链并启动交互
    rag_chain = RAGChain(vectorstore_manager, config)
    interactive_qa(rag_chain)


if __name__ == "__main__":
    main()
