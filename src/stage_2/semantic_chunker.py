"""
语义分块模块

实现基于语义的智能分块，保持语义完整性。
相比固定分块，语义分块能更好地保持上下文连贯性。
"""

from typing import List, Optional
from loguru import logger

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import sys
sys.path.append("..")
from stage_1.config import Config, get_config


class SemanticChunker:
    """
    语义分块器
    
    基于语义边界进行分块，而不是简单的字符数切分。
    使用句子嵌入来判断语义断点。
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        breakpoint_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """
        初始化语义分块器
        
        Args:
            config: 配置对象
            breakpoint_threshold: 语义断点阈值（0-1，越大越容易断开）
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self.config = config or get_config()
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # 初始化 embedding 模型（用于计算语义相似度）
        self._embeddings = self._create_embeddings()
        
        # 使用句子分割器作为预处理
        self._sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";"],
            is_separator_regex=False,
        )
        
        logger.info(
            f"🧠 语义分块器初始化: threshold={breakpoint_threshold}, "
            f"min={min_chunk_size}, max={max_chunk_size}"
        )
    
    def _create_embeddings(self) -> OpenAIEmbeddings:
        """创建嵌入模型"""
        kwargs = {
            "model": self.config.embedding_model,
            "api_key": self.config.openai_api_key,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return OpenAIEmbeddings(**kwargs)
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的块"""
        merged = []
        current = ""
        
        for chunk in chunks:
            if len(current) + len(chunk) <= self.max_chunk_size:
                current += chunk
            else:
                if current:
                    merged.append(current.strip())
                current = chunk
        
        if current:
            merged.append(current.strip())
        
        # 过滤太小的块
        return [c for c in merged if len(c) >= self.min_chunk_size]
    
    def split_text(self, text: str) -> List[str]:
        """
        基于语义边界分割文本
        
        Args:
            text: 原始文本
            
        Returns:
            List[str]: 分块后的文本列表
        """
        if not text or len(text) < self.min_chunk_size:
            return [text] if text else []
        
        # 1. 首先按句子分割
        sentences = self._sentence_splitter.split_text(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # 2. 计算每个句子的嵌入
        try:
            embeddings = self._embeddings.embed_documents(sentences)
        except Exception as e:
            logger.warning(f"⚠️ 嵌入计算失败，使用固定分块: {e}")
            return self._merge_small_chunks(sentences)
        
        # 3. 根据语义相似度找断点
        chunks = []
        current_chunk = sentences[0]
        
        for i in range(1, len(sentences)):
            # 计算当前句子与前一个句子的相似度
            similarity = self._compute_similarity(embeddings[i-1], embeddings[i])
            
            # 如果相似度低于阈值，且当前块足够大，则断开
            if similarity < self.breakpoint_threshold and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentences[i]
            else:
                # 检查是否超过最大大小
                if len(current_chunk) + len(sentences[i]) > self.max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentences[i]
                else:
                    current_chunk += " " + sentences[i]
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 4. 合并过小的块
        result = self._merge_small_chunks(chunks)
        
        logger.info(f"✅ 语义分块完成: {len(sentences)} 个句子 -> {len(result)} 个语义块")
        return result
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行语义分块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 分块后的文档列表
        """
        if not documents:
            return []
        
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "chunking_method": "semantic"
                    }
                )
                result.append(new_doc)
        
        logger.info(f"✅ 文档语义分块完成: {len(documents)} 个文档 -> {len(result)} 个块")
        return result
