"""
嵌入模型模块

封装 Embedding 模型，支持 OpenAI 和本地模型。
用于将文本转换为向量表示。
"""

from typing import List, Optional
from loguru import logger

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .config import Config, get_config


class EmbeddingModel:
    """
    嵌入模型封装类
    
    支持 OpenAI Embeddings 和本地 HuggingFace 模型。
    Phase 1 默认使用 OpenAI text-embedding-3-small。
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化嵌入模型
        
        Args:
            config: 配置对象，如果为 None 则从环境变量加载
        """
        self.config = config or get_config()
        self._embeddings: Optional[Embeddings] = None
        
    @property
    def embeddings(self) -> Embeddings:
        """
        获取嵌入模型实例（懒加载）
        
        Returns:
            Embeddings: LangChain 嵌入模型实例
        """
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def _create_embeddings(self) -> Embeddings:
        """
        创建嵌入模型实例
        
        Returns:
            Embeddings: LangChain 嵌入模型实例
        """
        model_name = self.config.embedding_model
        
        # 使用 OpenAI Embeddings
        kwargs = {
            "model": model_name,
            "api_key": self.config.openai_api_key,
        }
        
        # 如果设置了自定义 base_url（如 DeepSeek），则使用
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        
        embeddings = OpenAIEmbeddings(**kwargs)
        
        logger.info(f"🔢 嵌入模型初始化完成: {model_name}")
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        if not texts:
            return []
        
        vectors = self.embeddings.embed_documents(texts)
        logger.debug(f"✅ 已嵌入 {len(texts)} 个文档，向量维度: {len(vectors[0])}")
        return vectors
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 查询向量
        """
        vector = self.embeddings.embed_query(text)
        logger.debug(f"✅ 已嵌入查询，向量维度: {len(vector)}")
        return vector
