# Phase 1: 原型验证 (MVP)
"""
阶段一：原型验证 (MVP) - "Hello World" 级别

本模块实现最基础的 RAG 问答系统，包括：
- 文档加载与预处理
- 固定大小文本分块
- 向量存储与检索  
- 基础问答链
"""

from .config import Config
from .document_loader import DocumentLoader
from .chunker import TextChunker
from .embedder import EmbeddingModel
from .vectorstore import VectorStoreManager
from .rag_chain import RAGChain

__all__ = [
    "Config",
    "DocumentLoader", 
    "TextChunker",
    "EmbeddingModel",
    "VectorStoreManager",
    "RAGChain",
]
