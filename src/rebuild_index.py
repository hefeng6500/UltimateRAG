#!/usr/bin/env python3
"""
文档重建索引脚本

可复用脚本，用于重新对 documents 目录下的文档进行切块和 embedding。
支持在 Stage 1、2、3 中使用。

使用方式:
    # 使用默认参数重建所有 stage 的索引
    python -m src.rebuild_index
    
    # 仅重建 stage 1 的索引
    python -m src.rebuild_index --stage 1
    
    # 使用语义分块重建 stage 2 的索引
    python -m src.rebuild_index --stage 2 --semantic
    
    # 重建所有 stage 且使用语义分块
    python -m src.rebuild_index --stage all --semantic
    
    # 指定文档目录和 chunk 目录
    python -m src.rebuild_index --data ./data/documents --chunks-dir ./data/chunks
"""

import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Optional, Literal
from loguru import logger

from langchain_core.documents import Document

# 导入各个组件
from src.stage_1.config import Config, get_config
from src.stage_1.document_loader import DocumentLoader
from src.stage_1.chunker import TextChunker
from src.stage_1.vectorstore import VectorStoreManager


def setup_logger():
    """配置 loguru 日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True
    )


def clear_chunks_cache(chunks_dir: str = "./data/chunks"):
    """
    清除 chunk 缓存目录
    
    Args:
        chunks_dir: chunk 缓存目录路径
    """
    chunks_path = Path(chunks_dir)
    if chunks_path.exists():
        shutil.rmtree(chunks_path)
        logger.info(f"🗑️ 已清除 chunk 缓存: {chunks_path}")
    chunks_path.mkdir(parents=True, exist_ok=True)


def clear_vectorstore(
    config: Config,
    collection_name: str
):
    """
    清除向量库中的指定集合
    
    Args:
        config: 配置对象
        collection_name: 集合名称
    """
    try:
        manager = VectorStoreManager(config, collection_name=collection_name)
        manager.clear()
        logger.info(f"🗑️ 已清除向量库集合: {collection_name}")
    except Exception as e:
        logger.warning(f"⚠️ 清除向量库集合失败 ({collection_name}): {e}")


def load_documents(data_path: str) -> List[Document]:
    """
    加载文档
    
    Args:
        data_path: 文档路径
        
    Returns:
        List[Document]: 文档列表
    """
    loader = DocumentLoader()
    documents = loader.load(data_path)
    
    if not documents:
        logger.warning("⚠️ 没有找到任何文档")
        return []
    
    logger.info(f"📄 已加载 {len(documents)} 个文档")
    return documents


def chunk_documents_fixed(
    documents: List[Document],
    config: Config,
    chunks_dir: str = "./data/chunks"
) -> List[Document]:
    """
    使用固定大小分块
    
    Args:
        documents: 文档列表
        config: 配置对象
        chunks_dir: chunk 缓存目录
        
    Returns:
        List[Document]: 分块后的文档列表
    """
    chunker = TextChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        chunks_dir=chunks_dir
    )
    
    chunks = chunker.split_documents(
        documents,
        use_cache=False,  # 强制重新分块
        force_resplit=True
    )
    
    logger.info(f"✂️ 固定分块完成: {len(documents)} 个文档 -> {len(chunks)} 个块")
    return chunks


def chunk_documents_semantic(
    documents: List[Document],
    config: Config,
    chunks_dir: str = "./data/chunks"
) -> List[Document]:
    """
    使用语义分块
    
    Args:
        documents: 文档列表
        config: 配置对象
        chunks_dir: chunk 缓存目录
        
    Returns:
        List[Document]: 分块后的文档列表
    """
    try:
        from src.stage_2.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker(
            config=config,
            chunks_dir=chunks_dir
        )
        
        chunks = chunker.split_documents(
            documents,
            use_cache=False,  # 强制重新分块
            force_resplit=True
        )
        
        logger.info(f"🧠 语义分块完成: {len(documents)} 个文档 -> {len(chunks)} 个块")
        return chunks
    except ImportError:
        logger.error("❌ 语义分块需要 Stage 2 的模块支持")
        raise


def enrich_metadata(documents: List[Document]) -> List[Document]:
    """
    为文档添加元数据（Stage 2/3 功能）
    
    Args:
        documents: 文档列表
        
    Returns:
        List[Document]: 增强元数据后的文档列表
    """
    try:
        from src.stage_2.metadata_extractor import MetadataExtractor
        
        extractor = MetadataExtractor()
        enriched = extractor.enrich_documents(documents)
        logger.info("📋 元数据提取完成")
        return enriched
    except ImportError:
        logger.warning("⚠️ 元数据提取模块不可用，跳过")
        return documents


def rebuild_index(
    stage: Literal["1", "2", "3", "all"],
    data_path: str = "./data/documents",
    chunks_dir: str = "./data/chunks",
    use_semantic: bool = False,
    use_metadata: bool = True,
    clear_cache: bool = True
):
    """
    重建文档索引
    
    Args:
        stage: 目标 stage ("1", "2", "3" 或 "all")
        data_path: 文档路径
        chunks_dir: chunk 缓存目录
        use_semantic: 是否使用语义分块
        use_metadata: 是否提取元数据
        clear_cache: 是否清除旧缓存
    """
    # 加载配置
    config = get_config()
    
    if not config.validate():
        logger.error("❌ 配置验证失败，请检查 .env 文件")
        return False
    
    # 确定要处理的 stage
    stages = ["1", "2", "3"] if stage == "all" else [stage]
    
    # Stage 与 collection 名称的映射
    stage_collections = {
        "1": "rag_documents",
        "2": "advanced_rag",
        "3": "agentic_rag"
    }
    
    # 1. 清除缓存
    if clear_cache:
        logger.info("=" * 50)
        logger.info("🧹 清除旧数据...")
        logger.info("=" * 50)
        
        # 清除 chunk 缓存
        clear_chunks_cache(chunks_dir)
        
        # 清除对应 stage 的向量库
        for s in stages:
            collection_name = stage_collections[s]
            clear_vectorstore(config, collection_name)
    
    # 2. 加载文档
    logger.info("=" * 50)
    logger.info("📄 加载文档...")
    logger.info("=" * 50)
    
    documents = load_documents(data_path)
    if not documents:
        return False
    
    # 3. 元数据提取（可选）
    if use_metadata and stage in ["2", "3", "all"]:
        logger.info("=" * 50)
        logger.info("📋 提取元数据...")
        logger.info("=" * 50)
        documents = enrich_metadata(documents)
    
    # 4. 分块
    logger.info("=" * 50)
    if use_semantic:
        logger.info("🧠 语义分块...")
    else:
        logger.info("✂️ 固定分块...")
    logger.info("=" * 50)
    
    if use_semantic:
        chunks = chunk_documents_semantic(documents, config, chunks_dir)
    else:
        chunks = chunk_documents_fixed(documents, config, chunks_dir)
    
    if not chunks:
        logger.error("❌ 分块失败，没有生成任何块")
        return False
    
    # 5. 向量化并存入各 stage 的向量库
    logger.info("=" * 50)
    logger.info("🗄️ 向量化存储...")
    logger.info("=" * 50)
    
    for s in stages:
        collection_name = stage_collections[s]
        logger.info(f"📦 Stage {s} ({collection_name})...")
        
        manager = VectorStoreManager(config, collection_name=collection_name)
        manager.add_documents(chunks)
        
        # 验证
        count = manager.vectorstore._collection.count()
        logger.info(f"✅ Stage {s} 完成: {count} 个向量")
    
    # 6. 完成
    logger.info("=" * 50)
    logger.info("🎉 索引重建完成!")
    logger.info("=" * 50)
    
    # 打印摘要
    print("\n" + "=" * 50)
    print("📊 重建摘要")
    print("=" * 50)
    print(f"  • 文档数量: {len(documents)}")
    print(f"  • 分块数量: {len(chunks)}")
    print(f"  • 分块方式: {'语义分块' if use_semantic else '固定分块'}")
    print(f"  • 目标 Stage: {', '.join(stages)}")
    print(f"  • 元数据提取: {'是' if use_metadata else '否'}")
    print("=" * 50 + "\n")
    
    return True


def main():
    """主入口函数"""
    setup_logger()
    
    parser = argparse.ArgumentParser(
        description="文档重建索引脚本 - 重新对文档进行切块和 embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数重建所有 stage 的索引
  python -m src.rebuild_index
  
  # 仅重建 stage 1 的索引
  python -m src.rebuild_index --stage 1
  
  # 使用语义分块重建 stage 2 的索引
  python -m src.rebuild_index --stage 2 --semantic
  
  # 重建所有 stage 且使用语义分块
  python -m src.rebuild_index --stage all --semantic
"""
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="目标 Stage (1, 2, 3 或 all)，默认: all"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="./data/documents",
        help="文档路径（文件或目录），默认: ./data/documents"
    )
    
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="./data/chunks",
        help="chunk 缓存目录，默认: ./data/chunks"
    )
    
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="使用语义分块（否则使用固定分块）"
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="跳过元数据提取"
    )
    
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="保留旧的缓存数据（不清除）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔄 文档重建索引工具")
    print("=" * 60 + "\n")
    
    # 确认操作
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"❌ 数据路径不存在: {data_path}")
        return
    
    print(f"📁 数据路径: {data_path.absolute()}")
    print(f"📦 目标 Stage: {args.stage}")
    print(f"✂️ 分块方式: {'语义分块' if args.semantic else '固定分块'}")
    print(f"📋 元数据提取: {'否' if args.no_metadata else '是'}")
    print(f"🧹 清除旧缓存: {'否' if args.keep_cache else '是'}")
    print()
    
    # 执行重建
    success = rebuild_index(
        stage=args.stage,
        data_path=str(data_path),
        chunks_dir=args.chunks_dir,
        use_semantic=args.semantic,
        use_metadata=not args.no_metadata,
        clear_cache=not args.keep_cache
    )
    
    if not success:
        logger.error("❌ 索引重建失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

