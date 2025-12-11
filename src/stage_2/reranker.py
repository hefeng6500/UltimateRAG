"""
重排序模块

使用 Cross-Encoder 模型对检索结果进行精细化重排序。
支持 BGE-Reranker 等多种重排序模型。
"""

from typing import List, Optional, Tuple
from loguru import logger

from langchain_core.documents import Document

from src.stage_1.config import Config, get_config


class Reranker:
    """
    重排序器
    
    对粗检索的结果进行精细化重排序，
    使用 Cross-Encoder 模型计算查询与文档的精确相关性分数。
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    }
    
    def __init__(
        self,
        model_name: str = "bge-reranker-base",
        config: Optional[Config] = None,
        use_gpu: bool = False
    ):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
            config: 配置对象
            use_gpu: 是否使用 GPU
        """
        self.config = config or get_config()
        self.model_name = model_name
        self.use_gpu = use_gpu
        
        self._model = None
        self._is_loaded = False
        
        logger.info(f"🔄 重排序器初始化: 模型={model_name}")
    
    def _load_model(self):
        """懒加载重排序模型"""
        if self._is_loaded:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            # 获取完整模型名称
            if self.model_name in self.SUPPORTED_MODELS:
                full_model_name = self.SUPPORTED_MODELS[self.model_name]
            else:
                full_model_name = self.model_name
            
            device = "cuda" if self.use_gpu else "cpu"
            self._model = CrossEncoder(full_model_name, device=device)
            self._is_loaded = True
            
            logger.info(f"✅ 重排序模型加载完成: {full_model_name}")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的结果数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 分数) 列表，按分数降序
        """
        if not documents:
            return []
        
        top_k = top_k or self.config.top_k
        
        # 加载模型
        self._load_model()
        
        # 准备输入对
        pairs = [(query, doc.page_content) for doc in documents]
        
        # 计算分数
        try:
            scores = self._model.predict(pairs)
            
            # 组合文档和分数
            doc_scores = list(zip(documents, scores))
            
            # 按分数降序排序
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回 Top-K
            result = doc_scores[:top_k]
            
            logger.info(
                f"🔄 重排序完成: {len(documents)} -> {len(result)} 个文档, "
                f"最高分={result[0][1]:.4f}"
            )
            
            return result
        except Exception as e:
            logger.error(f"❌ 重排序失败: {e}")
            # 失败时返回原始顺序
            return [(doc, 0.0) for doc in documents[:top_k]]
    
    def rerank_and_filter(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None,
        threshold: float = 0.0
    ) -> List[Document]:
        """
        重排序并过滤低相关文档
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的结果数量
            threshold: 分数阈值，低于此分数的文档将被过滤
            
        Returns:
            List[Document]: 排序并过滤后的文档列表
        """
        reranked = self.rerank(query, documents, top_k=len(documents))
        
        # 过滤低分文档
        filtered = [
            doc for doc, score in reranked 
            if score >= threshold
        ]
        
        # 限制返回数量
        top_k = top_k or self.config.top_k
        result = filtered[:top_k]
        
        if len(filtered) < len(reranked):
            logger.info(f"🔍 过滤低相关文档: {len(reranked)} -> {len(filtered)} -> {len(result)}")
        
        return result


class SimpleReranker:
    """
    简单重排序器（不依赖外部模型）
    
    使用基于规则的方法进行重排序，适用于无法加载模型的情况。
    """
    
    def __init__(self, config: Optional[Config] = None):
        """初始化简单重排序器"""
        self.config = config or get_config()
        logger.info("🔄 简单重排序器初始化完成")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        基于关键词匹配的简单重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的结果数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 分数) 列表
        """
        if not documents:
            return []
        
        top_k = top_k or self.config.top_k
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # 计算匹配的查询词数量
            matches = sum(1 for term in query_terms if term in content_lower)
            
            # 考虑词频
            frequency = sum(content_lower.count(term) for term in query_terms)
            
            # 综合分数
            score = matches * 2 + frequency * 0.1
            scored_docs.append((doc, score))
        
        # 排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
