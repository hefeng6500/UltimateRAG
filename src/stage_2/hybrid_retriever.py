"""
混合检索模块

实现 BM25 关键词检索 + 向量检索的混合策略。
能够同时处理语义匹配和关键词精确匹配。
"""

from typing import List, Optional, Tuple
from loguru import logger
import numpy as np

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.stage_1.config import Config, get_config
from src.stage_1.vectorstore import VectorStoreManager


class HybridRetriever:
    """
    混合检索器
    
    结合 BM25 关键词检索和向量语义检索的优势。
    使用倒排融合（RRF）算法合并结果。
    """
    
    def __init__(
        self,
        documents: List[Document] = None,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        config: Optional[Config] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ):
        """
        初始化混合检索器
        
        Args:
            documents: 文档列表（用于构建 BM25 索引）
            vectorstore_manager: 向量存储管理器
            config: 配置对象
            bm25_weight: BM25 结果权重
            vector_weight: 向量检索结果权重
        """
        self.config = config or get_config()
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(self.config)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # BM25 索引
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        
        if documents:
            self.build_bm25_index(documents)
        
        logger.info(
            f"🔀 混合检索器初始化: BM25权重={bm25_weight}, 向量权重={vector_weight}"
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单的中英文分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        import re
        # 中文按字分，英文按词分
        tokens = []
        
        # 分割中英文
        segments = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())
        
        for segment in segments:
            # 判断是否是中文
            if re.match(r'[\u4e00-\u9fff]', segment):
                # 中文按字符分割
                tokens.extend(list(segment))
            else:
                # 英文保持原样
                tokens.append(segment)
        
        return tokens
    
    def build_bm25_index(self, documents: List[Document]):
        """
        构建 BM25 索引
        
        Args:
            documents: 文档列表
        """
        self._documents = documents
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) 
            for doc in documents
        ]
        
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"✅ BM25 索引构建完成: {len(documents)} 个文档")
    
    def bm25_search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        BM25 关键词检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 分数) 列表
        """
        if self._bm25 is None:
            logger.warning("⚠️ BM25 索引未构建")
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # 获取 Top-K 索引
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = [
            (self._documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]
        
        return results
    
    def vector_search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        向量语义检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 分数) 列表
        """
        results = self.vectorstore_manager.similarity_search_with_score(query, k=k)
        
        # ChromaDB 返回的是距离，需要转换为相似度分数
        # 距离越小，相似度越高
        processed = []
        for doc, distance in results:
            # 将距离转换为 0-1 的相似度分数
            similarity = 1 / (1 + distance)
            processed.append((doc, similarity))
        
        return processed
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[Document, float]]],
        k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        倒排融合算法 (Reciprocal Rank Fusion)
        
        Args:
            result_lists: 多个检索结果列表
            k: RRF 参数（防止排名过于集中）
            
        Returns:
            List[Tuple[Document, float]]: 融合后的结果列表
        """
        # 使用文档内容作为唯一标识
        doc_scores = {}
        doc_map = {}
        
        for results in result_lists:
            for rank, (doc, _) in enumerate(results):
                doc_key = hash(doc.page_content)
                
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = 0
                    doc_map[doc_key] = doc
                
                # RRF 公式: 1 / (k + rank)
                doc_scores[doc_key] += 1 / (k + rank + 1)
        
        # 按分数排序
        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        return [(doc_map[key], doc_scores[key]) for key in sorted_keys]
    
    def search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        混合检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Document]: 检索结果
        """
        k = k or self.config.top_k
        
        # 检索更多结果用于融合
        search_k = k * 3
        
        # BM25 检索
        bm25_results = self.bm25_search(query, k=search_k)
        
        # 向量检索
        vector_results = self.vector_search(query, k=search_k)
        
        # 融合结果
        if bm25_results and vector_results:
            # 加权 RRF
            weighted_bm25 = [(doc, score * self.bm25_weight) for doc, score in bm25_results]
            weighted_vector = [(doc, score * self.vector_weight) for doc, score in vector_results]
            fused = self._reciprocal_rank_fusion([weighted_bm25, weighted_vector])
        elif bm25_results:
            fused = bm25_results
        elif vector_results:
            fused = vector_results
        else:
            fused = []
        
        # 返回 Top-K
        results = [doc for doc, _ in fused[:k]]
        
        logger.info(
            f"🔍 混合检索完成: BM25={len(bm25_results)}, "
            f"向量={len(vector_results)}, 融合后={len(results)}"
        )
        
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        混合检索（带分数）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 分数) 列表
        """
        k = k or self.config.top_k
        search_k = k * 3
        
        bm25_results = self.bm25_search(query, k=search_k)
        vector_results = self.vector_search(query, k=search_k)
        
        if bm25_results and vector_results:
            weighted_bm25 = [(doc, score * self.bm25_weight) for doc, score in bm25_results]
            weighted_vector = [(doc, score * self.vector_weight) for doc, score in vector_results]
            fused = self._reciprocal_rank_fusion([weighted_bm25, weighted_vector])
        elif bm25_results:
            fused = bm25_results
        elif vector_results:
            fused = vector_results
        else:
            fused = []
        
        return fused[:k]
