# Phase 2: 质量飞跃 (Advanced RAG)
"""
阶段二：质量飞跃 (Advanced RAG) - 解决"检索不准"

本模块实现高级 RAG 功能，包括：
- 语义分块 (Semantic Chunking)
- 混合检索 (Hybrid Search: BM25 + 向量)
- 查询重写 (Query Rewrite / HyDE)
- 重排序 (Re-ranking)
"""

from .semantic_chunker import SemanticChunker
from .metadata_extractor import MetadataExtractor
from .hybrid_retriever import HybridRetriever
from .query_rewriter import QueryRewriter
from .reranker import Reranker
from .advanced_rag_chain import AdvancedRAGChain

__all__ = [
    "SemanticChunker",
    "MetadataExtractor",
    "HybridRetriever",
    "QueryRewriter",
    "Reranker",
    "AdvancedRAGChain",
]
