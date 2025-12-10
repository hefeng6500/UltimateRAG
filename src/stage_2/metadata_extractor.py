"""
元数据提取模块

从文档中提取结构化元数据，用于后续的过滤检索。
支持提取标题、日期、作者等信息。
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from langchain_core.documents import Document


class MetadataExtractor:
    """
    元数据提取器
    
    从文档内容和文件路径中提取有价值的元数据。
    """
    
    # 常见的日期格式
    DATE_PATTERNS = [
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?",  # 2024-01-01, 2024年1月1日
        r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",           # 01-01-2024
        r"\d{4}\d{2}\d{2}",                       # 20240101
    ]
    
    # Markdown 标题模式
    HEADER_PATTERNS = [
        r"^#\s+(.+)$",      # # Title
        r"^##\s+(.+)$",     # ## Subtitle
        r"^###\s+(.+)$",    # ### Section
    ]
    
    def __init__(self):
        """初始化元数据提取器"""
        logger.info("📋 元数据提取器初始化完成")
    
    def extract_from_path(self, file_path: str) -> Dict[str, Any]:
        """
        从文件路径提取元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 提取的元数据
        """
        path = Path(file_path)
        
        metadata = {
            "file_name": path.name,
            "file_stem": path.stem,
            "file_extension": path.suffix.lower(),
            "file_path": str(path),
            "parent_directory": path.parent.name,
        }
        
        # 尝试获取文件修改时间
        try:
            stat = path.stat()
            metadata["modified_time"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            metadata["file_size_bytes"] = stat.st_size
        except Exception:
            pass
        
        return metadata
    
    def extract_from_content(self, content: str) -> Dict[str, Any]:
        """
        从文档内容提取元数据
        
        Args:
            content: 文档内容
            
        Returns:
            Dict: 提取的元数据
        """
        metadata = {}
        
        # 提取日期
        dates = self._extract_dates(content)
        if dates:
            metadata["extracted_dates"] = dates
            metadata["first_date"] = dates[0]
        
        # 提取标题（从 Markdown 格式）
        headers = self._extract_headers(content)
        if headers:
            metadata["headers"] = headers[:5]  # 只保留前 5 个标题
            metadata["title"] = headers[0] if headers else None
        
        # 统计信息
        metadata["char_count"] = len(content)
        metadata["word_count"] = len(content.split())
        metadata["line_count"] = content.count("\n") + 1
        
        return metadata
    
    def _extract_dates(self, content: str) -> List[str]:
        """提取文档中的日期"""
        dates = []
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, content)
            dates.extend(matches)
        return list(set(dates))[:10]  # 去重并限制数量
    
    def _extract_headers(self, content: str) -> List[str]:
        """提取 Markdown 标题"""
        headers = []
        for line in content.split("\n"):
            for pattern in self.HEADER_PATTERNS:
                match = re.match(pattern, line.strip())
                if match:
                    headers.append(match.group(1).strip())
                    break
        return headers
    
    def enrich_documents(self, documents: List[Document]) -> List[Document]:
        """
        为文档列表添加丰富的元数据
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 添加元数据后的文档列表
        """
        enriched = []
        
        for doc in documents:
            # 从路径提取元数据
            path_metadata = {}
            if "source" in doc.metadata:
                path_metadata = self.extract_from_path(doc.metadata["source"])
            elif "source_file" in doc.metadata:
                path_metadata = self.extract_from_path(doc.metadata["source_file"])
            
            # 从内容提取元数据
            content_metadata = self.extract_from_content(doc.page_content)
            
            # 合并元数据
            enriched_metadata = {
                **doc.metadata,
                **path_metadata,
                **content_metadata
            }
            
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=enriched_metadata
            )
            enriched.append(enriched_doc)
        
        logger.info(f"✅ 元数据增强完成: {len(documents)} 个文档")
        return enriched
    
    def filter_by_metadata(
        self,
        documents: List[Document],
        filters: Dict[str, Any]
    ) -> List[Document]:
        """
        根据元数据过滤文档
        
        Args:
            documents: 文档列表
            filters: 过滤条件，例如 {"file_extension": ".md"}
            
        Returns:
            List[Document]: 过滤后的文档列表
        """
        result = []
        
        for doc in documents:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                
                doc_value = doc.metadata[key]
                
                # 支持列表匹配
                if isinstance(value, list):
                    if doc_value not in value:
                        match = False
                        break
                else:
                    if doc_value != value:
                        match = False
                        break
            
            if match:
                result.append(doc)
        
        logger.info(f"🔍 元数据过滤: {len(documents)} -> {len(result)} 个文档")
        return result
