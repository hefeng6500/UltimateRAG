"""
高级 RAG 链模块

组装 Phase 2 的所有组件，实现完整的 Advanced RAG 流程：
1. 查询重写 (Query Rewrite)
2. 混合检索 (Hybrid Search)
3. 重排序 (Re-ranking)
4. 上下文压缩与生成
"""

from typing import List, Optional, Dict, Any
from loguru import logger

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.stage_1.config import Config, get_config
from src.stage_1.vectorstore import VectorStoreManager

from .hybrid_retriever import HybridRetriever
from .query_rewriter import QueryRewriter
from .reranker import Reranker, SimpleReranker


# 高级 RAG Prompt 模板
ADVANCED_RAG_PROMPT = """你是一个专业的问答助手，具备以下能力：
1. 精确理解用户问题
2. 综合多个文档来源提供全面回答
3. 当信息不足时，诚实承认并说明已知内容

请根据以下参考文档回答用户的问题。

参考文档：
{context}

用户问题：{question}

回答要求：
- 使用中文回答
- 答案要准确且有条理
- 如果文档中没有相关信息，请明确说明

请回答："""


class AdvancedRAGChain:
    """
    高级 RAG 问答链
    
    整合查询重写、混合检索、重排序等高级技术，
    实现比 Phase 1 更准确的检索和问答。
    """
    
    def __init__(
        self,
        documents: List[Document] = None,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        config: Optional[Config] = None,
        use_query_rewrite: bool = True,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        reranker_model: str = "bge-reranker-base"
    ):
        """
        初始化高级 RAG 链
        
        Args:
            documents: 文档列表（用于构建 BM25 索引）
            vectorstore_manager: 向量存储管理器
            config: 配置对象
            use_query_rewrite: 是否启用查询重写
            use_hybrid_search: 是否启用混合检索
            use_reranking: 是否启用重排序
            reranker_model: 重排序模型名称
        """
        self.config = config or get_config()
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(self.config)
        
        # 功能开关
        self.use_query_rewrite = use_query_rewrite
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        
        # 初始化组件
        self._llm = self._create_llm()
        self._prompt = ChatPromptTemplate.from_template(ADVANCED_RAG_PROMPT)
        
        # 查询重写器
        if use_query_rewrite:
            self._query_rewriter = QueryRewriter(self.config)
        
        # 混合检索器
        if use_hybrid_search and documents:
            self._hybrid_retriever = HybridRetriever(
                documents=documents,
                vectorstore_manager=self.vectorstore_manager,
                config=self.config
            )
        else:
            self._hybrid_retriever = None
        
        # 重排序器
        if use_reranking:
            try:
                self._reranker = Reranker(
                    model_name=reranker_model,
                    config=self.config
                )
            except Exception as e:
                logger.warning(f"⚠️ 无法加载重排序模型，使用简单重排: {e}")
                self._reranker = SimpleReranker(self.config)
        
        logger.info(
            f"🚀 高级 RAG 链初始化完成: "
            f"查询重写={use_query_rewrite}, "
            f"混合检索={use_hybrid_search}, "
            f"重排序={use_reranking}"
        )
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.7,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """格式化文档"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知来源")
            content = doc.page_content.strip()
            formatted.append(f"[文档 {i}] (来源: {source})\n{content}")
        return "\n\n---\n\n".join(formatted)
    
    def _retrieve(self, query: str, expanded_queries: List[str] = None) -> List[Document]:
        """
        执行检索（支持多查询）
        
        Args:
            query: 原始查询
            expanded_queries: 扩展后的查询列表
            
        Returns:
            List[Document]: 去重后的检索结果
        """
        queries = expanded_queries or [query]
        all_docs = []
        seen_contents = set()
        
        for q in queries:
            # 使用混合检索或纯向量检索
            if self._hybrid_retriever:
                docs = self._hybrid_retriever.search(q, k=self.config.top_k * 2)
            else:
                docs = self.vectorstore_manager.similarity_search(q, k=self.config.top_k * 2)
            
            # 去重
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        logger.info(f"🔍 检索完成: {len(queries)} 个查询 -> {len(all_docs)} 个唯一文档")
        return all_docs
    
    def ask(self, question: str) -> str:
        """
        提问并获取回答（使用所有高级特性）
        
        Args:
            question: 用户问题
            
        Returns:
            str: LLM 生成的回答
        """
        logger.info(f"❓ 收到问题: {question}")
        
        # 1. 查询重写
        if self.use_query_rewrite:
            queries = self._query_rewriter.generate_multi_queries(question)
        else:
            queries = [question]
        
        # 2. 检索
        docs = self._retrieve(question, queries)
        
        if not docs:
            return "抱歉，没有找到相关的文档来回答这个问题。"
        
        # 3. 重排序
        if self.use_reranking and len(docs) > self.config.top_k:
            reranked = self._reranker.rerank(question, docs, top_k=self.config.top_k)
            docs = [doc for doc, _ in reranked]
        else:
            docs = docs[:self.config.top_k]
        
        # 4. 生成回答
        context = self._format_docs(docs)
        prompt = self._prompt.format(context=context, question=question)
        
        response = self._llm.invoke(prompt)
        answer = response.content
        
        logger.info("✅ 回答生成完成")
        return answer
    
    def ask_with_details(self, question: str) -> Dict[str, Any]:
        """
        提问并返回详细信息（包括中间步骤）
        
        Args:
            question: 用户问题
            
        Returns:
            Dict: 包含 answer、sources、queries 等详细信息
        """
        logger.info(f"❓ 收到问题 (详细模式): {question}")
        
        result = {
            "question": question,
            "queries": [],
            "retrieved_docs": 0,
            "reranked_docs": 0,
            "sources": [],
            "answer": ""
        }
        
        # 1. 查询重写
        if self.use_query_rewrite:
            queries = self._query_rewriter.generate_multi_queries(question)
        else:
            queries = [question]
        result["queries"] = queries
        
        # 2. 检索
        docs = self._retrieve(question, queries)
        result["retrieved_docs"] = len(docs)
        
        if not docs:
            result["answer"] = "抱歉，没有找到相关的文档来回答这个问题。"
            return result
        
        # 3. 重排序
        if self.use_reranking and len(docs) > self.config.top_k:
            reranked = self._reranker.rerank(question, docs, top_k=self.config.top_k)
            docs_with_scores = reranked
            docs = [doc for doc, _ in reranked]
        else:
            docs_with_scores = [(doc, 0.0) for doc in docs[:self.config.top_k]]
            docs = docs[:self.config.top_k]
        
        result["reranked_docs"] = len(docs)
        
        # 整理来源信息
        result["sources"] = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("file_name", "未知"),
                "score": float(score),
                "metadata": {k: v for k, v in doc.metadata.items() if k != "page_content"}
            }
            for doc, score in docs_with_scores
        ]
        
        # 4. 生成回答
        context = self._format_docs(docs)
        prompt = self._prompt.format(context=context, question=question)
        
        response = self._llm.invoke(prompt)
        result["answer"] = response.content
        
        logger.info("✅ 详细回答生成完成")
        return result
