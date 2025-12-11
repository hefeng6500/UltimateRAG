"""
语义分块模块

实现基于语义的智能分块，保持语义完整性。
相比固定分块，语义分块能更好地保持上下文连贯性。

优化功能：
- 支持将分块结果保存到本地文件
- 支持从本地文件读取已保存的分块
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional
from loguru import logger

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.stage_1.config import Config, get_config
from src.stage_1.embedder import BatchedEmbeddings, ALIYUN_EMBEDDING_BATCH_SIZE


class SemanticChunker:
    """
    语义分块器
    
    基于语义边界进行分块，而不是简单的字符数切分。
    使用句子嵌入来判断语义断点。
    
    支持将分块结果保存到本地，下次可直接读取。
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        breakpoint_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunks_dir: str = "./data/chunks"
    ):
        """
        初始化语义分块器
        
        Args:
            config: 配置对象
            breakpoint_threshold: 语义断点阈值（0-1，越大越容易断开）
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
            chunks_dir: 分块数据存储目录
        """
        self.config = config or get_config()
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunks_dir = Path(chunks_dir)
        
        # 创建存储目录
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def _create_embeddings(self) -> Embeddings:
        """
        创建嵌入模型（使用 BatchedEmbeddings 包装器支持阿里云批量限制）
        
        Returns:
            Embeddings: 已包装分批处理的嵌入模型
        """
        kwargs = {
            "model": self.config.embedding_model,
            "api_key": self.config.openai_api_key,
            "check_embedding_ctx_length": False,  # 阿里云兼容模式需要关闭
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        
        base_embeddings = OpenAIEmbeddings(**kwargs)
        # 使用 BatchedEmbeddings 包装，每批最多 10 条（阿里云限制）
        return BatchedEmbeddings(inner=base_embeddings, batch_size=ALIYUN_EMBEDDING_BATCH_SIZE)
    
    def _get_cache_path(self, documents: List[Document]) -> Path:
        """
        根据文档内容生成缓存文件路径
        
        Args:
            documents: 文档列表
            
        Returns:
            Path: 缓存文件路径
        """
        # 使用文档内容的哈希值作为缓存标识
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.page_content.encode('utf-8'))
        
        # 加入分块参数作为哈希的一部分（语义分块特有参数）
        params = f"semantic_{self.breakpoint_threshold}_{self.min_chunk_size}_{self.max_chunk_size}"
        content_hash.update(params.encode('utf-8'))
        
        return self.chunks_dir / f"chunks_semantic_{content_hash.hexdigest()[:16]}.json"
    
    def _save_chunks(self, chunks: List[Document], cache_path: Path):
        """
        将分块结果保存到本地文件
        
        Args:
            chunks: 分块后的文档列表
            cache_path: 缓存文件路径
        """
        data = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "index": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "size": len(chunk.page_content)
            }
            data.append(chunk_data)
            
            # 同时保存单独的 chunk 文件以便查看
            chunk_file = self.chunks_dir / f"chunk_{i:04d}.md"
            self._save_single_chunk(chunk, i, chunk_file)
        
        # 保存索引文件
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_chunks": len(chunks),
                "chunking_method": "semantic",
                "breakpoint_threshold": self.breakpoint_threshold,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
                "chunks": data
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 语义分块数据已保存: {cache_path}")
    
    def _save_single_chunk(self, chunk: Document, index: int, file_path: Path):
        """
        保存单个 chunk 为可读的 Markdown 文件
        
        Args:
            chunk: 分块文档
            index: 分块索引
            file_path: 文件路径
        """
        source = chunk.metadata.get("file_name", "未知来源")
        method = chunk.metadata.get("chunking_method", "semantic")
        content = f"""# Chunk {index}

## 元数据
- **来源文件**: {source}
- **字符数**: {len(chunk.page_content)}
- **分块索引**: {index}
- **分块方法**: {method}

## 内容

```
{chunk.page_content}
```
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _load_chunks(self, cache_path: Path) -> Optional[List[Document]]:
        """
        从本地文件读取分块结果
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            Optional[List[Document]]: 分块文档列表，如果不存在则返回 None
        """
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            for chunk_data in data["chunks"]:
                doc = Document(
                    page_content=chunk_data["content"],
                    metadata=chunk_data["metadata"]
                )
                chunks.append(doc)
            
            logger.info(f"📂 从缓存加载语义分块: {len(chunks)} 个块")
            return chunks
        except Exception as e:
            logger.warning(f"⚠️ 加载缓存失败: {e}")
            return None
    
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
    
    def split_documents(
        self, 
        documents: List[Document],
        use_cache: bool = True,
        force_resplit: bool = False
    ) -> List[Document]:
        """
        对文档列表进行语义分块
        
        Args:
            documents: 原始文档列表
            use_cache: 是否使用缓存
            force_resplit: 是否强制重新分块（忽略缓存）
            
        Returns:
            List[Document]: 分块后的文档列表
        """
        if not documents:
            logger.warning("⚠️ 输入文档列表为空")
            return []
        
        cache_path = self._get_cache_path(documents)
        
        # 尝试从缓存加载
        if use_cache and not force_resplit:
            cached_chunks = self._load_chunks(cache_path)
            if cached_chunks:
                return cached_chunks
        
        # 执行语义分块
        logger.info("🔄 开始语义分块处理...")
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
        
        # 打印分块统计信息
        sizes = [len(c.page_content) for c in result]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        logger.info(
            f"📊 分块统计: 平均大小={avg_size:.0f}, "
            f"最小={min(sizes) if sizes else 0}, "
            f"最大={max(sizes) if sizes else 0}"
        )
        
        # 保存到本地
        self._save_chunks(result, cache_path)
        
        return result
    
    def clear_cache(self):
        """清空所有缓存的分块文件"""
        import shutil
        if self.chunks_dir.exists():
            shutil.rmtree(self.chunks_dir)
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info("🗑️ 语义分块缓存已清空")
    
    def list_cached_chunks(self) -> List[Path]:
        """
        列出所有缓存的 chunk 文件
        
        Returns:
            List[Path]: chunk 文件路径列表
        """
        if not self.chunks_dir.exists():
            return []
        
        return sorted(self.chunks_dir.glob("chunk_*.md"))
