"""
文本分块模块

实现 FixedSizeChunking（固定大小分块）策略。
Phase 1 使用基础的固定分块，后续阶段将引入语义分块。
"""

from typing import List
from loguru import logger

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    文本分块器
    
    将长文档切分为固定大小的小块，便于向量化和检索。
    使用 RecursiveCharacterTextSplitter 实现智能分割。
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分割符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行分块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 分块后的文档列表
        """
        if not documents:
            logger.warning("⚠️ 输入文档列表为空")
            return []
        
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
        logger.debug(
            f"📊 分块统计: 平均大小={avg_size:.0f}, "
            f"最小={min(sizes) if sizes else 0}, "
            f"最大={max(sizes) if sizes else 0}"
        )
        
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
