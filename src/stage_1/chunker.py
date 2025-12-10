"""
文本分块模块

实现 FixedSizeChunking（固定大小分块）策略。
Phase 1 使用基础的固定分块，后续阶段将引入语义分块。

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
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    文本分块器
    
    将长文档切分为固定大小的小块，便于向量化和检索。
    使用 RecursiveCharacterTextSplitter 实现智能分割。
    
    支持将分块结果保存到本地，下次可直接读取。
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
        chunks_dir: str = "./data/chunks"
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分割符列表，按优先级排序
            chunks_dir: 分块数据存储目录
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks_dir = Path(chunks_dir)
        
        # 创建存储目录
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认分割符：按段落 -> 句子 -> 词语顺序分割
        self.separators = separators or [
            "\n\n",  # 段落
            "\n",    # 换行
            "。",    # 中文句号
            ".",     # 英文句号
            "！",    # 中文感叹号
            "!",     # 英文感叹号
            "？",    # 中文问号
            "?",     # 英文问号
            "；",    # 中文分号
            ";",     # 英文分号
            "，",    # 中文逗号
            ",",     # 英文逗号
            " ",     # 空格
            "",      # 字符级别
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info(
            f"✂️ 文本分块器初始化: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"存储目录={self.chunks_dir}"
        )
    
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
        
        # 加入分块参数作为哈希的一部分
        params = f"{self.chunk_size}_{self.chunk_overlap}"
        content_hash.update(params.encode('utf-8'))
        
        return self.chunks_dir / f"chunks_{content_hash.hexdigest()[:16]}.json"
    
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
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunks": data
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 分块数据已保存: {cache_path}")
    
    def _save_single_chunk(self, chunk: Document, index: int, file_path: Path):
        """
        保存单个 chunk 为可读的 Markdown 文件
        
        Args:
            chunk: 分块文档
            index: 分块索引
            file_path: 文件路径
        """
        source = chunk.metadata.get("file_name", "未知来源")
        content = f"""# Chunk {index}

## 元数据
- **来源文件**: {source}
- **字符数**: {len(chunk.page_content)}
- **分块索引**: {index}

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
            
            logger.info(f"📂 从缓存加载分块: {len(chunks)} 个块")
            return chunks
        except Exception as e:
            logger.warning(f"⚠️ 加载缓存失败: {e}")
            return None
    
    def split_documents(
        self, 
        documents: List[Document],
        use_cache: bool = True,
        force_resplit: bool = False
    ) -> List[Document]:
        """
        对文档列表进行分块
        
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
        
        # 执行分块
        logger.info("🔄 开始分块处理...")
        chunks = self.splitter.split_documents(documents)
        
        # 为每个块添加索引信息
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.info(
            f"✅ 分块完成: {len(documents)} 个文档 -> {len(chunks)} 个块"
        )
        
        # 打印分块统计信息
        sizes = [len(c.page_content) for c in chunks]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        logger.info(
            f"📊 分块统计: 平均大小={avg_size:.0f}, "
            f"最小={min(sizes) if sizes else 0}, "
            f"最大={max(sizes) if sizes else 0}"
        )
        
        # 保存到本地
        self._save_chunks(chunks, cache_path)
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """
        对纯文本进行分块
        
        Args:
            text: 原始文本
            
        Returns:
            List[str]: 分块后的文本列表
        """
        if not text:
            return []
        
        chunks = self.splitter.split_text(text)
        logger.info(f"✅ 文本分块完成: {len(text)} 字符 -> {len(chunks)} 个块")
        return chunks
    
    def clear_cache(self):
        """清空所有缓存的分块文件"""
        import shutil
        if self.chunks_dir.exists():
            shutil.rmtree(self.chunks_dir)
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info("🗑️ 分块缓存已清空")
    
    def list_cached_chunks(self) -> List[Path]:
        """
        列出所有缓存的 chunk 文件
        
        Returns:
            List[Path]: chunk 文件路径列表
        """
        if not self.chunks_dir.exists():
            return []
        
        return sorted(self.chunks_dir.glob("chunk_*.md"))
