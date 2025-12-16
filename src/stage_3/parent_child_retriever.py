"""
父子索引检索器

实现 Parent-Child Indexing 策略：
- 用小的子块进行精准检索
- 返回大的父块提供完整上下文
"""

from typing import List, Optional, Dict, Tuple
from uuid import uuid4
import hashlib
from loguru import logger

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.stage_1.config import Config, get_config
from src.stage_1.vectorstore import VectorStoreManager
from .config import Stage3Config, get_stage3_config


class ParentChildRetriever:
    """
    父子索引检索器
    
    核心思想：
    - 父块（Parent Chunk）：大的文档块，包含完整上下文
    - 子块（Child Chunk）：小的文档块，用于精准匹配
    - 检索时用子块匹配，返回时用父块提供上下文
    """
    
    def __init__(
        self,
        config: Optional[Stage3Config] = None,
        vectorstore_manager: Optional[VectorStoreManager] = None
    ):
        """
        初始化父子检索器
        
        Args:
            config: Stage3 配置
            vectorstore_manager: 向量存储管理器（用于子块检索）
        """
        self.config = config or get_stage3_config()
        
        # 父块存储（内存映射）
        self._parent_store: Dict[str, Document] = {}
        
        # 子块到父块的映射
        self._child_to_parent: Dict[str, str] = {}
        
        # 向量存储（用于子块）
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(
            get_config(),
            collection_name="parent_child_index"
        )
        
        # 分块器
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.parent_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.child_chunk_size,
            chunk_overlap=self.config.child_chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        logger.info(
            f"👨‍👧 父子索引检索器初始化: "
            f"父块={self.config.parent_chunk_size}, "
            f"子块={self.config.child_chunk_size}"
        )
    
    def _generate_id(self, content: str) -> str:
        """
        为内容生成唯一 ID
        
        Args:
            content: 文档内容
            
        Returns:
            str: 唯一 ID
        """
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        添加文档并建立父子索引
        
        Args:
            documents: 原始文档列表
            
        Returns:
            int: 添加的子块数量
        """
        all_child_chunks = []
        
        for doc in documents:
            # 1. 先切分为父块
            parent_chunks = self._parent_splitter.split_documents([doc])
            
            for parent in parent_chunks:
                # 生成父块 ID
                parent_id = self._generate_id(parent.page_content)
                
                # 存储父块
                self._parent_store[parent_id] = parent
                
                # 2. 将父块切分为子块
                child_chunks = self._child_splitter.split_documents([parent])
                
                for child in child_chunks:
                    # 生成子块 ID
                    child_id = self._generate_id(child.page_content)
                    
                    # 建立映射关系
                    self._child_to_parent[child_id] = parent_id
                    
                    # 在子块元数据中添加 ID
                    child.metadata["child_id"] = child_id
                    child.metadata["parent_id"] = parent_id
                    
                    all_child_chunks.append(child)
        
        # 3. 将子块存入向量库
        if all_child_chunks:
            self.vectorstore_manager.add_documents(all_child_chunks)
        
        logger.info(
            f"✅ 父子索引构建完成: "
            f"{len(documents)} 个文档 -> "
            f"{len(self._parent_store)} 个父块 -> "
            f"{len(all_child_chunks)} 个子块"
        )
        
        return len(all_child_chunks)
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        return_parents: bool = True
    ) -> List[Document]:
        """
        检索文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            return_parents: 是否返回父块（否则返回子块）
            
        Returns:
            List[Document]: 检索结果
        """
        k = k or self.config.top_k
        
        # 用子块进行检索
        child_results = self.vectorstore_manager.similarity_search(
            query, 
            k=k * 2  # 检索更多子块以去重
        )
        
        if not return_parents:
            return child_results[:k]
        
        # 转换为父块（去重）
        seen_parents = set()
        parent_results = []
        
        for child in child_results:
            parent_id = child.metadata.get("parent_id")
            
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                
                # 从存储中获取父块
                parent = self._parent_store.get(parent_id)
                if parent:
                    parent_results.append(parent)
                else:
                    # 如果父块不在内存中，使用子块作为回退
                    parent_results.append(child)
                
                if len(parent_results) >= k:
                    break
        
        logger.info(
            f"🔍 父子检索完成: "
            f"检索 {len(child_results)} 个子块 -> "
            f"返回 {len(parent_results)} 个父块"
        )
        
        return parent_results
    
    def retrieve_with_context(
        self,
        query: str,
        k: int = None,
        context_window: int = 1
    ) -> List[Document]:
        """
        检索并扩展上下文窗口
        
        获取父块及其相邻的父块，提供更完整的上下文。
        
        Args:
            query: 查询文本
            k: 返回结果数量
            context_window: 上下文窗口大小（前后各取多少个父块）
            
        Returns:
            List[Document]: 扩展上下文后的检索结果
        """
        # 先获取父块
        parents = self.retrieve(query, k=k, return_parents=True)
        
        if context_window <= 0:
            return parents
        
        # 获取所有父块 ID 的有序列表
        parent_ids = list(self._parent_store.keys())
        
        expanded_results = []
        seen_ids = set()
        
        for parent in parents:
            parent_id = parent.metadata.get("parent_id")
            
            if parent_id:
                try:
                    idx = parent_ids.index(parent_id)
                    
                    # 获取上下文窗口内的父块
                    start_idx = max(0, idx - context_window)
                    end_idx = min(len(parent_ids), idx + context_window + 1)
                    
                    for i in range(start_idx, end_idx):
                        pid = parent_ids[i]
                        if pid not in seen_ids:
                            seen_ids.add(pid)
                            expanded_results.append(self._parent_store[pid])
                except ValueError:
                    # 父块 ID 不在列表中
                    if parent_id not in seen_ids:
                        seen_ids.add(parent_id)
                        expanded_results.append(parent)
            else:
                expanded_results.append(parent)
        
        logger.info(
            f"🔍 上下文扩展: {len(parents)} -> {len(expanded_results)} 个文档"
        )
        
        return expanded_results
    
    def get_parent_by_id(self, parent_id: str) -> Optional[Document]:
        """
        根据 ID 获取父块
        
        Args:
            parent_id: 父块 ID
            
        Returns:
            Optional[Document]: 父块文档
        """
        return self._parent_store.get(parent_id)
    
    def get_statistics(self) -> Dict:
        """
        获取索引统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "parent_count": len(self._parent_store),
            "child_count": len(self._child_to_parent),
            "parent_chunk_size": self.config.parent_chunk_size,
            "child_chunk_size": self.config.child_chunk_size,
        }

