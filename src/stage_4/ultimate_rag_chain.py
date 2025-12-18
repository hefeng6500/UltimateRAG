"""
终极 RAG 链

整合 Stage 1-4 所有组件，实现最强大的 RAG 系统：
- Stage 1: 基础向量检索
- Stage 2: 混合检索 + 重排序
- Stage 3: Agentic RAG（路由、自反思、工具）
- Stage 4: GraphRAG
"""

from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.stage_1.config import get_config
from src.stage_1.vectorstore import VectorStoreManager
from src.stage_2.hybrid_retriever import HybridRetriever
from src.stage_2.reranker import Reranker, SimpleReranker
from src.stage_3.router import QueryRouter, RouteType
from src.stage_3.self_rag import SelfRAG
from src.stage_3.context_compressor import ContextCompressor, KeywordBasedCompressor

from .config import Stage4Config, get_stage4_config
from .graph_rag import (
    GraphRAGChain,
    KnowledgeGraph,
    GraphRetriever,
    GraphRetrievalResult,
)


class RetrievalMode(str, Enum):
    """检索模式"""
    VECTOR = "vector"       # 纯向量检索
    HYBRID = "hybrid"       # 混合检索（向量 + BM25）
    GRAPH = "graph"         # 纯图检索
    FUSION = "fusion"       # 融合检索（向量 + 图）
    AUTO = "auto"           # 自动选择


@dataclass
class UltimateRAGResult:
    """
    终极 RAG 处理结果
    
    整合所有阶段的结果信息。
    """
    answer: str
    query: str
    retrieval_mode: RetrievalMode
    route_type: Optional[RouteType] = None
    
    # 检索信息
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    graph_entities: List[Dict[str, Any]] = field(default_factory=list)
    graph_relations: List[Dict[str, Any]] = field(default_factory=list)
    
    # 质量信息
    confidence: float = 0.0
    quality_score: float = 0.0
    iterations: int = 1
    
    # 来源追溯
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    
    # 工具使用
    tool_used: Optional[str] = None
    tool_result: Optional[str] = None


# 终极 RAG 提示词
ULTIMATE_RAG_PROMPT = """你是一个强大的智能问答助手，能够综合利用知识图谱和文档内容来回答问题。

【知识图谱信息】
{graph_context}

【相关文档】
{doc_context}

用户问题：{question}

回答要求：
1. 综合利用图谱信息和文档内容
2. 优先使用图谱中的结构化关系信息
3. 如果涉及实体关系，请明确说明
4. 如果信息不足，请诚实说明
5. 回答要准确、完整、有条理

请用中文回答："""

# 查询分析提示词
QUERY_ANALYSIS_PROMPT = """分析以下用户查询，判断最适合的检索模式。

用户查询：{query}

检索模式说明：
- vector: 简单的语义相似检索，适合一般性问题
- hybrid: 混合检索，适合包含专有名词或关键词的问题
- graph: 图检索，适合关系查询、路径查询
- fusion: 融合检索，适合复杂问题

请分析查询特点，选择最合适的检索模式。
输出格式（JSON）：{{"mode": "检索模式", "reason": "选择原因"}}"""


class UltimateRAGChain:
    """
    终极 RAG 链
    
    整合 Stage 1-4 所有能力：
    - 向量检索 (Stage 1)
    - 混合检索 + 重排序 (Stage 2)
    - 智能路由 + 自反思 + 工具 (Stage 3)
    - 知识图谱检索 (Stage 4)
    """
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        config: Optional[Stage4Config] = None,
        graph_name: str = "ultimate_rag",
        # 功能开关
        enable_routing: bool = True,
        enable_self_rag: bool = True,
        enable_graph_rag: bool = True,
        enable_reranking: bool = True,
        enable_compression: bool = True,
        # 重建选项
        force_rebuild_graph: bool = False,
        force_rebuild_index: bool = False,
    ):
        """
        初始化终极 RAG 链
        
        Args:
            documents: 文档列表
            config: 配置
            graph_name: 知识图谱名称
            enable_routing: 启用智能路由
            enable_self_rag: 启用自反思
            enable_graph_rag: 启用图检索
            enable_reranking: 启用重排序
            enable_compression: 启用上下文压缩
            force_rebuild_graph: 强制重建知识图谱
            force_rebuild_index: 强制重建向量索引
        """
        self.config = config or get_stage4_config()
        self.base_config = get_config()
        self.graph_name = graph_name
        
        # 功能开关
        self.enable_routing = enable_routing
        self.enable_self_rag = enable_self_rag
        self.enable_graph_rag = enable_graph_rag
        self.enable_reranking = enable_reranking
        self.enable_compression = enable_compression
        
        # 初始化 LLM
        self._llm = self._create_llm()
        
        # 初始化组件
        self._init_components(
            documents,
            force_rebuild_graph,
            force_rebuild_index,
        )
        
        logger.info(
            f"🎯 终极 RAG 链初始化完成:\n"
            f"   - 路由: {enable_routing}\n"
            f"   - 自反思: {enable_self_rag}\n"
            f"   - 图检索: {enable_graph_rag}\n"
            f"   - 重排序: {enable_reranking}\n"
            f"   - 上下文压缩: {enable_compression}"
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
    
    def _init_components(
        self,
        documents: Optional[List[Document]],
        force_rebuild_graph: bool,
        force_rebuild_index: bool,
    ):
        """初始化所有组件"""
        
        # 1. 向量存储
        self._vectorstore_manager = VectorStoreManager(
            self.base_config,
            collection_name=f"ultimate_rag_{self.graph_name}"
        )
        
        if force_rebuild_index:
            self._vectorstore_manager.clear()
            logger.info("🗑️ 已清空向量存储")
        
        # 2. 混合检索器
        self._hybrid_retriever = None
        if documents:
            self._hybrid_retriever = HybridRetriever(
                documents=documents,
                vectorstore_manager=self._vectorstore_manager,
                config=self.base_config,
            )
        
        # 3. 重排序器
        if self.enable_reranking:
            try:
                self._reranker = Reranker(config=self.base_config)
            except Exception:
                self._reranker = SimpleReranker(self.base_config)
        else:
            self._reranker = None
        
        # 4. 路由器
        if self.enable_routing:
            from src.stage_3.config import get_stage3_config
            stage3_config = get_stage3_config()
            self._router = QueryRouter(stage3_config)
        else:
            self._router = None
        
        # 5. 自反思
        if self.enable_self_rag:
            from src.stage_3.config import get_stage3_config
            stage3_config = get_stage3_config()
            self._self_rag = SelfRAG(stage3_config)
        else:
            self._self_rag = None
        
        # 6. 上下文压缩
        if self.enable_compression:
            try:
                self._compressor = ContextCompressor(self.config)
            except Exception:
                self._compressor = KeywordBasedCompressor(self.config)
        else:
            self._compressor = None
        
        # 7. GraphRAG（核心）
        if self.enable_graph_rag:
            self._graph_rag = GraphRAGChain(
                documents=documents,
                config=self.config,
                graph_name=self.graph_name,
                force_rebuild=force_rebuild_graph,
            )
        else:
            self._graph_rag = None
    
    def _analyze_query(self, query: str) -> RetrievalMode:
        """
        分析查询，确定检索模式
        
        Args:
            query: 用户查询
            
        Returns:
            RetrievalMode: 检索模式
        """
        # 关系相关的关键词
        relation_keywords = [
            "关系", "联系", "之间", "和...的", "与...的",
            "路径", "连接", "关联", "涉及到", "相关的",
        ]
        
        # 全局性问题关键词
        global_keywords = [
            "总结", "概括", "全部", "所有", "整体",
            "趋势", "模式", "分布", "统计",
        ]
        
        query_lower = query.lower()
        
        # 检查是否是关系查询
        for kw in relation_keywords:
            if kw in query_lower:
                return RetrievalMode.GRAPH
        
        # 检查是否是全局性问题
        for kw in global_keywords:
            if kw in query_lower:
                return RetrievalMode.GRAPH
        
        # 默认使用融合检索
        return RetrievalMode.FUSION
    
    def _vector_retrieve(
        self,
        query: str,
        k: int,
    ) -> List[Document]:
        """向量检索"""
        if self._hybrid_retriever:
            docs = self._hybrid_retriever.search(query, k=k * 2)
        else:
            docs = self._vectorstore_manager.similarity_search(query, k=k * 2)
        
        # 重排序
        if self._reranker and len(docs) > k:
            reranked = self._reranker.rerank(query, docs, top_k=k)
            docs = [doc for doc, _ in reranked]
        else:
            docs = docs[:k]
        
        return docs
    
    def _graph_retrieve(
        self,
        query: str,
        k: int,
    ) -> GraphRetrievalResult:
        """图检索"""
        if not self._graph_rag:
            return GraphRetrievalResult(query=query, context="图检索未启用")
        
        return self._graph_rag._graph_retriever.retrieve(query, top_k=k)
    
    def _fusion_retrieve(
        self,
        query: str,
        k: int,
    ) -> tuple[List[Document], GraphRetrievalResult]:
        """融合检索"""
        # 向量检索
        docs = self._vector_retrieve(query, k)
        
        # 图检索
        graph_result = self._graph_retrieve(query, k)
        
        return docs, graph_result
    
    def _format_docs(self, docs: List[Document]) -> str:
        """格式化文档"""
        if not docs:
            return "未找到相关文档。"
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知来源")
            content = doc.page_content.strip()[:500]
            formatted.append(f"[文档 {i}] (来源: {source})\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def ask(
        self,
        question: str,
        retrieval_mode: RetrievalMode = RetrievalMode.AUTO,
        top_k: int = None,
    ) -> UltimateRAGResult:
        """
        处理用户问题
        
        Args:
            question: 用户问题
            retrieval_mode: 检索模式
            top_k: 返回的文档数量
            
        Returns:
            UltimateRAGResult: 处理结果
        """
        top_k = top_k or self.config.top_k
        
        logger.info(f"❓ 终极 RAG 问答: {question}")
        
        reasoning_chain = []
        
        # 1. 路由决策（可选）
        route_type = None
        if self.enable_routing and self._router:
            decision = self._router.route(question)
            route_type = decision.route_type
            reasoning_chain.append(
                f"路由决策: {route_type.value} (置信度: {decision.confidence:.2f})"
            )
            
            # 如果路由到工具，使用 Stage 3 的处理
            if route_type in [RouteType.CALCULATOR, RouteType.CODE_EXECUTION]:
                from src.stage_3.agentic_rag_chain import AgenticRAGChain
                # 委托给 Stage 3 处理
                # 这里简化处理，直接使用知识库
                pass
        
        # 2. 确定检索模式
        if retrieval_mode == RetrievalMode.AUTO:
            retrieval_mode = self._analyze_query(question)
        reasoning_chain.append(f"检索模式: {retrieval_mode.value}")
        
        # 3. 执行检索
        docs = []
        graph_result = None
        
        if retrieval_mode == RetrievalMode.VECTOR:
            docs = self._vector_retrieve(question, top_k)
            reasoning_chain.append(f"向量检索: {len(docs)} 个文档")
            
        elif retrieval_mode == RetrievalMode.HYBRID:
            docs = self._vector_retrieve(question, top_k)
            reasoning_chain.append(f"混合检索: {len(docs)} 个文档")
            
        elif retrieval_mode == RetrievalMode.GRAPH:
            graph_result = self._graph_retrieve(question, top_k)
            reasoning_chain.append(
                f"图检索: {len(graph_result.matched_entities)} 个实体"
            )
            
        elif retrieval_mode == RetrievalMode.FUSION:
            docs, graph_result = self._fusion_retrieve(question, top_k)
            reasoning_chain.append(
                f"融合检索: {len(docs)} 个文档, "
                f"{len(graph_result.matched_entities) if graph_result else 0} 个实体"
            )
        
        # 4. 上下文压缩
        if self._compressor and docs:
            docs = self._compressor.compress_documents(question, docs)
            reasoning_chain.append("执行上下文压缩")
        
        # 5. 生成答案
        doc_context = self._format_docs(docs) if docs else "无相关文档"
        graph_context = graph_result.context if graph_result else "无图谱信息"
        
        prompt = ChatPromptTemplate.from_template(ULTIMATE_RAG_PROMPT)
        
        response = self._llm.invoke(
            prompt.format(
                graph_context=graph_context,
                doc_context=doc_context,
                question=question,
            )
        )
        
        answer = response.content
        
        # 6. 自反思（可选）
        iterations = 1
        quality_score = 0.0
        
        if self.enable_self_rag and self._self_rag:
            combined_context = f"{graph_context}\n\n{doc_context}"
            quality_eval = self._self_rag.evaluate_answer_quality(
                question, combined_context, answer
            )
            quality_score = quality_eval.score
            reasoning_chain.append(
                f"质量评估: {quality_eval.grade.value} (分数: {quality_score:.2f})"
            )
        
        # 7. 构建结果
        retrieved_docs = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("file_name", "未知"),
            }
            for doc in docs
        ]
        
        graph_entities = []
        graph_relations = []
        if graph_result:
            graph_entities = [e.to_dict() for e in graph_result.matched_entities]
            if graph_result.subgraph:
                graph_relations = [
                    r.to_dict() for r in graph_result.subgraph.edges[:10]
                ]
        
        # 来源
        sources = retrieved_docs.copy()
        for entity in (graph_result.matched_entities if graph_result else []):
            sources.append({
                "content": entity.description,
                "source": f"实体: {entity.name}",
                "type": "entity",
            })
        
        reasoning_chain.append("处理完成")
        
        result = UltimateRAGResult(
            answer=answer,
            query=question,
            retrieval_mode=retrieval_mode,
            route_type=route_type,
            retrieved_docs=retrieved_docs,
            graph_entities=graph_entities,
            graph_relations=graph_relations,
            confidence=graph_result.confidence if graph_result else 0.5,
            quality_score=quality_score,
            iterations=iterations,
            sources=sources,
            reasoning_chain=reasoning_chain,
        )
        
        logger.info(f"✅ 终极 RAG 问答完成")
        
        return result
    
    def ask_simple(self, question: str) -> str:
        """简化版问答"""
        result = self.ask(question)
        return result.answer
    
    def build_knowledge_graph(self, documents: List[Document]):
        """构建知识图谱"""
        if self._graph_rag:
            self._graph_rag.build_knowledge_graph(documents)
    
    def generate_global_summary(self) -> str:
        """生成全局摘要"""
        if self._graph_rag:
            return self._graph_rag.generate_global_summary()
        return "GraphRAG 未启用"
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """获取实体信息"""
        if self._graph_rag:
            return self._graph_rag.get_entity_info(entity_name)
        return None
    
    def find_path(
        self,
        source: str,
        target: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """查找实体间路径"""
        if self._graph_rag:
            return self._graph_rag.find_path(source, target)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "components": {
                "routing": self.enable_routing,
                "self_rag": self.enable_self_rag,
                "graph_rag": self.enable_graph_rag,
                "reranking": self.enable_reranking,
                "compression": self.enable_compression,
            }
        }
        
        if self._graph_rag:
            stats["knowledge_graph"] = self._graph_rag.get_statistics()
        
        return stats

