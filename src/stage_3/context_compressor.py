"""
上下文压缩模块

实现 Context Compression：
- 从检索结果中提取最相关的句子
- 删除无关内容，节省 Token
- 保留核心信息
"""

from typing import List, Optional, Tuple
from loguru import logger

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import Stage3Config, get_stage3_config


class ExtractedContent(BaseModel):
    """提取的相关内容"""
    relevant_sentences: List[str] = Field(
        description="与问题最相关的句子列表"
    )
    relevance_score: float = Field(
        description="整体相关性分数 0-1",
        ge=0.0,
        le=1.0
    )
    summary: str = Field(
        description="内容的简短摘要"
    )


# 内容提取提示词
EXTRACTION_PROMPT = """你是一个信息提取专家。
从给定文档中提取与用户问题最相关的句子。

用户问题：{question}

文档内容：
{document}

要求：
1. 只提取直接相关的句子
2. 保持句子原样，不要修改
3. 如果没有相关内容，返回空列表
4. 给出整体相关性评分

请提取相关内容。"""

# 快速压缩提示词
COMPRESSION_PROMPT = """压缩以下文本，只保留与问题相关的核心信息：

问题：{question}

原文：
{document}

压缩后的文本（保留关键信息，删除无关内容）："""


class ContextCompressor:
    """
    上下文压缩器
    
    从检索结果中提取最相关的内容，压缩上下文长度。
    """
    
    def __init__(self, config: Optional[Stage3Config] = None):
        """
        初始化上下文压缩器
        
        Args:
            config: Stage3 配置
        """
        self.config = config or get_stage3_config()
        self._llm = self._create_llm()
        self._extractor = self._llm.with_structured_output(ExtractedContent)
        
        logger.info("📦 上下文压缩器初始化完成")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.1,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def extract_relevant_content(
        self,
        question: str,
        document: Document
    ) -> Tuple[List[str], float]:
        """
        从文档中提取相关内容
        
        Args:
            question: 用户问题
            document: 待处理文档
            
        Returns:
            Tuple[List[str], float]: (相关句子列表, 相关性分数)
        """
        prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
        
        try:
            result = self._extractor.invoke(
                prompt.format(
                    question=question,
                    document=document.page_content[:2000]
                )
            )
            return result.relevant_sentences, result.relevance_score
        except Exception as e:
            logger.warning(f"⚠️ 内容提取失败: {e}")
            return [], 0.5
    
    def compress_document(
        self,
        question: str,
        document: Document
    ) -> Document:
        """
        压缩单个文档
        
        Args:
            question: 用户问题
            document: 待压缩文档
            
        Returns:
            Document: 压缩后的文档
        """
        prompt = ChatPromptTemplate.from_template(COMPRESSION_PROMPT)
        
        try:
            response = self._llm.invoke(
                prompt.format(
                    question=question,
                    document=document.page_content[:2000]
                )
            )
            
            compressed_content = response.content.strip()
            
            # 创建新文档保留元数据
            compressed_doc = Document(
                page_content=compressed_content,
                metadata={
                    **document.metadata,
                    "compressed": True,
                    "original_length": len(document.page_content),
                    "compressed_length": len(compressed_content)
                }
            )
            
            return compressed_doc
        except Exception as e:
            logger.warning(f"⚠️ 文档压缩失败: {e}")
            return document
    
    def compress_documents(
        self,
        question: str,
        documents: List[Document],
        method: str = "extract"
    ) -> List[Document]:
        """
        批量压缩文档
        
        Args:
            question: 用户问题
            documents: 文档列表
            method: 压缩方法 ("extract" 提取相关句子, "compress" LLM 压缩)
            
        Returns:
            List[Document]: 压缩后的文档列表
        """
        compressed_docs = []
        total_original = 0
        total_compressed = 0
        
        for doc in documents:
            total_original += len(doc.page_content)
            
            if method == "extract":
                sentences, score = self.extract_relevant_content(question, doc)
                
                if sentences and score >= self.config.min_relevant_score:
                    compressed_content = "\n".join(sentences)
                    compressed_doc = Document(
                        page_content=compressed_content,
                        metadata={
                            **doc.metadata,
                            "compressed": True,
                            "relevance_score": score,
                            "original_length": len(doc.page_content),
                            "compressed_length": len(compressed_content)
                        }
                    )
                    compressed_docs.append(compressed_doc)
                    total_compressed += len(compressed_content)
                elif score >= self.config.min_relevant_score:
                    # 没有提取到句子但相关性够，保留原文档
                    compressed_docs.append(doc)
                    total_compressed += len(doc.page_content)
            else:
                compressed_doc = self.compress_document(question, doc)
                compressed_docs.append(compressed_doc)
                total_compressed += len(compressed_doc.page_content)
        
        compression_ratio = (
            1 - total_compressed / total_original 
            if total_original > 0 else 0
        )
        
        logger.info(
            f"📦 上下文压缩完成: "
            f"{len(documents)} -> {len(compressed_docs)} 个文档, "
            f"压缩比: {compression_ratio:.1%}"
        )
        
        return compressed_docs


class KeywordBasedCompressor:
    """
    基于关键词的快速压缩器（不依赖 LLM）
    
    使用关键词匹配快速过滤不相关的句子。
    """
    
    def __init__(self, config: Optional[Stage3Config] = None):
        """初始化关键词压缩器"""
        self.config = config or get_stage3_config()
        logger.info("🔑 关键词压缩器初始化完成")
    
    def _extract_keywords(self, text: str) -> set:
        """
        提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            set: 关键词集合
        """
        import re
        
        # 简单分词
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
        
        # 过滤停用词（简化版）
        stopwords = {
            '的', '是', '在', '了', '和', '与', '或', '这', '那', '有', '个', '为',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from'
        }
        
        return {w for w in words if w not in stopwords and len(w) > 1}
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分割句子
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 句子列表
        """
        import re
        
        # 按中英文句号分割
        sentences = re.split(r'[。！？\n.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def compress(
        self,
        question: str,
        document: Document,
        min_score: float = 0.3
    ) -> Document:
        """
        基于关键词压缩文档
        
        Args:
            question: 用户问题
            document: 待压缩文档
            min_score: 最小匹配分数
            
        Returns:
            Document: 压缩后的文档
        """
        # 提取问题关键词
        question_keywords = self._extract_keywords(question)
        
        if not question_keywords:
            return document
        
        # 分割句子
        sentences = self._split_sentences(document.page_content)
        
        # 计算每个句子的相关性
        relevant_sentences = []
        for sentence in sentences:
            sentence_keywords = self._extract_keywords(sentence)
            
            if not sentence_keywords:
                continue
            
            # 计算 Jaccard 相似度
            intersection = len(question_keywords & sentence_keywords)
            union = len(question_keywords | sentence_keywords)
            score = intersection / union if union > 0 else 0
            
            if score >= min_score:
                relevant_sentences.append(sentence)
        
        if not relevant_sentences:
            # 如果没有匹配的句子，保留原文档
            return document
        
        compressed_content = "。".join(relevant_sentences)
        
        return Document(
            page_content=compressed_content,
            metadata={
                **document.metadata,
                "compressed": True,
                "method": "keyword",
                "original_length": len(document.page_content),
                "compressed_length": len(compressed_content)
            }
        )
    
    def compress_documents(
        self,
        question: str,
        documents: List[Document],
        min_score: float = 0.3
    ) -> List[Document]:
        """
        批量压缩文档
        
        Args:
            question: 用户问题
            documents: 文档列表
            min_score: 最小匹配分数
            
        Returns:
            List[Document]: 压缩后的文档列表
        """
        compressed = [
            self.compress(question, doc, min_score)
            for doc in documents
        ]
        
        total_original = sum(len(d.page_content) for d in documents)
        total_compressed = sum(len(d.page_content) for d in compressed)
        ratio = 1 - total_compressed / total_original if total_original > 0 else 0
        
        logger.info(f"🔑 关键词压缩完成: 压缩比 {ratio:.1%}")
        
        return compressed

