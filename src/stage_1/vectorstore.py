"""
向量存储模块

使用 ChromaDB 作为向量数据库。
支持文档向量化存储和语义检索。
"""

import os
from pathlib import Path
from typing import List, Optional
from loguru import logger

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

from .config import Config, get_config
from .embedder import EmbeddingModel


class VectorStoreManager:
    """
    向量存储管理器
    
    封装 ChromaDB 的初始化、存储和检索操作。
    支持持久化存储，避免重复向量化。
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        collection_name: str = "rag_documents"
    ):
        """
        初始化向量存储管理器
        
        Args:
            config: 配置对象
            collection_name: ChromaDB 集合名称
        """
        self.config = config or get_config()
        self.collection_name = collection_name
        self.persist_dir = Path(self.config.chroma_persist_dir)
        
        # 创建持久化目录
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化嵌入模型
        self._embedding_model = EmbeddingModel(self.config)
        
        # 向量存储实例
        self._vectorstore: Optional[VectorStore] = None
        
        logger.info(f"🗄️ 向量存储管理器初始化: 持久化目录={self.persist_dir}")
    
    @property
    def vectorstore(self) -> VectorStore:
        """
        获取向量存储实例（懒加载）
        
        Returns:
            VectorStore: ChromaDB 向量存储实例
        """
        if self._vectorstore is None:
            self._vectorstore = self._load_or_create_vectorstore()
        return self._vectorstore
    
    def _load_or_create_vectorstore(self) -> VectorStore:
        """
        加载已有向量库或创建新的
        
        Returns:
            VectorStore: ChromaDB 实例
        """
        try:
            # 尝试加载已有的向量库
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embedding_model.embeddings,
                persist_directory=str(self.persist_dir),
            )
            
            # 检查是否已有数据
            count = vectorstore._collection.count()
            if count > 0:
                logger.info(f"✅ 加载已有向量库: {count} 个向量")
            else:
                logger.info("📦 创建新的向量库")
            
            return vectorstore
        except Exception as e:
            logger.warning(f"⚠️ 加载向量库失败，创建新库: {e}")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embedding_model.embeddings,
                persist_directory=str(self.persist_dir),
            )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        向向量库添加文档
        
        Args:
            documents: 文档列表
            
        Returns:
            List[str]: 添加的文档 ID 列表
        """
        if not documents:
            logger.warning("⚠️ 没有文档需要添加")
            return []
        
        ids = self.vectorstore.add_documents(documents)
        logger.info(f"✅ 已添加 {len(documents)} 个文档到向量库")
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        相似度检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Document]: 相关文档列表
        """
        k = k or self.config.top_k
        
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"🔍 检索完成: 查询='{query[:50]}...'，返回 {len(results)} 个结果")
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        带分数的相似度检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[tuple]: (文档, 分数) 列表
        """
        k = k or self.config.top_k
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(
            f"🔍 检索完成: 查询='{query[:50]}...'，"
            f"返回 {len(results)} 个结果 (带分数)"
        )
        
        return results
    
    def as_retriever(self, **kwargs):
        """
        转换为 LangChain Retriever
        
        Args:
            **kwargs: 传递给 as_retriever 的参数
            
        Returns:
            Retriever: LangChain Retriever 实例
        """
        search_kwargs = kwargs.pop("search_kwargs", {})
        if "k" not in search_kwargs:
            search_kwargs["k"] = self.config.top_k
        
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs,
            **kwargs
        )
    
    def clear(self):
        """清空向量库"""
        try:
            # 删除并重新创建集合
            self.vectorstore._client.delete_collection(self.collection_name)
            self._vectorstore = None
            logger.info("🗑️ 向量库已清空")
        except Exception as e:
            logger.error(f"❌ 清空向量库失败: {e}")
            raise
