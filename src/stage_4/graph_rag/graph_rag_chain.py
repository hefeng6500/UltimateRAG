"""
GraphRAG 链

整合实体抽取、关系抽取、知识图谱、图检索，
实现知识图谱增强的 RAG。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import os

from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.stage_4.config import Stage4Config, get_stage4_config
from src.stage_1.vectorstore import VectorStoreManager
from src.stage_2.hybrid_retriever import HybridRetriever

from .entity_extractor import Entity, EntityExtractor
from .relation_extractor import Relation, RelationExtractor
from .knowledge_graph import KnowledgeGraph
from .graph_store import GraphStore, create_graph_store
from .graph_retriever import GraphRetriever, GraphRetrievalResult


@dataclass
class GraphRAGResult:
    """
    GraphRAG 处理结果
    
    Attributes:
        answer: 答案
        query: 原始查询
        graph_context: 图检索上下文
        vector_context: 向量检索上下文
        matched_entities: 匹配的实体
        related_relations: 相关关系
        sources: 来源信息
        confidence: 置信度
    """
    answer: str
    query: str
    graph_context: str = ""
    vector_context: str = ""
    matched_entities: List[Dict[str, Any]] = field(default_factory=list)
    related_relations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


# GraphRAG 生成提示词
GRAPH_RAG_PROMPT = """你是一个智能问答助手，能够综合利用知识图谱信息和文档内容来回答问题。

【知识图谱信息】
{graph_context}

【文档内容】
{vector_context}

用户问题：{question}

回答要求：
1. 优先利用知识图谱中的实体关系信息
2. 结合文档内容进行补充说明
3. 如果涉及实体间关系，请明确指出
4. 如果信息不足，请诚实说明

请用中文回答："""

# 全局摘要提示词
GLOBAL_SUMMARY_PROMPT = """你是一个智能分析助手，请基于知识图谱的整体结构生成全局性摘要。

知识图谱统计：
- 实体数量：{num_entities}
- 关系数量：{num_relations}
- 主要实体类型：{entity_types}
- 主要关系类型：{relation_types}

主要实体列表：
{top_entities}

核心关系：
{top_relations}

请生成一个全局性摘要，概括这个知识图谱所描述的主要内容和核心关系："""


class GraphRAGChain:
    """
    GraphRAG 链
    
    实现知识图谱增强的 RAG：
    1. 从文档构建知识图谱
    2. 结合图检索和向量检索
    3. 利用图结构信息增强答案
    """
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        config: Optional[Stage4Config] = None,
        graph_store: Optional[GraphStore] = None,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        graph_name: str = "default",
        force_rebuild: bool = False,
    ):
        """
        初始化 GraphRAG 链
        
        Args:
            documents: 文档列表
            config: 配置
            graph_store: 图存储
            vectorstore_manager: 向量存储管理器
            graph_name: 图谱名称
            force_rebuild: 是否强制重建图谱
        """
        self.config = config or get_stage4_config()
        self.graph_name = graph_name
        
        # 初始化 LLM
        self._llm = self._create_llm()
        
        # 初始化图存储
        self._graph_store = graph_store or create_graph_store(config=self.config)
        
        # 初始化或加载知识图谱
        if force_rebuild or not self._graph_store.exists(graph_name):
            self._knowledge_graph = KnowledgeGraph()
        else:
            self._knowledge_graph = self._graph_store.load(graph_name) or KnowledgeGraph()
        
        # 初始化实体和关系抽取器
        self._entity_extractor = EntityExtractor(config=self.config, llm=self._llm)
        self._relation_extractor = RelationExtractor(config=self.config, llm=self._llm)
        
        # 初始化图检索器
        self._graph_retriever = GraphRetriever(
            self._knowledge_graph,
            config=self.config,
            llm=self._llm,
        )
        
        # 初始化向量存储（用于混合检索）
        self._vectorstore_manager = vectorstore_manager
        self._hybrid_retriever = None
        
        if documents:
            self._init_vector_retriever(documents)
        
        # 如果强制重建且有文档，构建图谱
        if force_rebuild and documents:
            self.build_knowledge_graph(documents)
        
        logger.info(
            f"🎯 GraphRAG 链初始化完成: "
            f"图谱 '{graph_name}' (节点: {self._knowledge_graph.num_nodes}, "
            f"边: {self._knowledge_graph.num_edges})"
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
    
    def _init_vector_retriever(self, documents: List[Document]):
        """初始化向量检索器"""
        from src.stage_1.config import get_config
        base_config = get_config()
        
        if self._vectorstore_manager is None:
            self._vectorstore_manager = VectorStoreManager(
                base_config,
                collection_name=f"graph_rag_{self.graph_name}"
            )
        
        self._hybrid_retriever = HybridRetriever(
            documents=documents,
            vectorstore_manager=self._vectorstore_manager,
            config=base_config,
        )
    
    def build_knowledge_graph(
        self,
        documents: Optional[List[Document]] = None,
        save: bool = True,
    ):
        """
        从文档构建知识图谱
        
        Args:
            documents: 文档列表（如果为 None，使用初始化时的文档）
            save: 是否保存图谱
        """
        if documents is None:
            logger.warning("没有提供文档，无法构建图谱")
            return
        
        logger.info(f"🔨 开始构建知识图谱: {len(documents)} 个文档")
        
        all_entities = []
        all_relations = []
        
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}")
            
            text = doc.page_content
            source_doc = doc.metadata.get("file_name", f"doc_{i}")
            
            # 1. 实体抽取
            entities = self._entity_extractor.extract(text, source_doc=source_doc)
            all_entities.extend(entities)
            
            # 2. 关系抽取
            if entities:
                relations = self._relation_extractor.extract(text, entities)
                all_relations.extend(relations)
        
        # 3. 合并实体
        merged_entities = self._entity_extractor.merge_entities(all_entities)
        
        # 4. 合并关系
        merged_relations = self._relation_extractor.merge_relations(all_relations)
        
        # 5. 构建图谱
        for entity in merged_entities:
            self._knowledge_graph.add_entity(entity)
        
        for relation in merged_relations:
            self._knowledge_graph.add_relation(relation)
        
        # 6. 保存图谱
        if save:
            self._graph_store.save(self._knowledge_graph, self.graph_name)
        
        # 7. 更新检索器
        self._graph_retriever = GraphRetriever(
            self._knowledge_graph,
            config=self.config,
            llm=self._llm,
        )
        
        logger.info(
            f"✅ 知识图谱构建完成: "
            f"{self._knowledge_graph.num_nodes} 个实体, "
            f"{self._knowledge_graph.num_edges} 条关系"
        )
    
    def _format_vector_docs(self, docs: List[Document]) -> str:
        """格式化向量检索结果"""
        if not docs:
            return "未找到相关文档。"
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知来源")
            content = doc.page_content.strip()[:500]  # 限制长度
            formatted.append(f"[文档 {i}] (来源: {source})\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def ask(
        self,
        question: str,
        use_vector_retrieval: bool = True,
        top_k: int = None,
    ) -> GraphRAGResult:
        """
        处理用户问题
        
        Args:
            question: 用户问题
            use_vector_retrieval: 是否同时使用向量检索
            top_k: 返回的文档数量
            
        Returns:
            GraphRAGResult: 处理结果
        """
        top_k = top_k or self.config.top_k
        
        logger.info(f"❓ GraphRAG 问答: {question}")
        
        # 1. 图检索
        graph_result = self._graph_retriever.retrieve(question, top_k=top_k)
        graph_context = graph_result.context
        
        # 2. 向量检索（如果启用）
        vector_context = ""
        vector_docs = []
        if use_vector_retrieval and self._hybrid_retriever:
            vector_docs = self._hybrid_retriever.search(question, k=top_k)
            vector_context = self._format_vector_docs(vector_docs)
        
        # 3. 生成答案
        prompt = ChatPromptTemplate.from_template(GRAPH_RAG_PROMPT)
        
        response = self._llm.invoke(
            prompt.format(
                graph_context=graph_context or "无相关图谱信息",
                vector_context=vector_context or "无相关文档",
                question=question,
            )
        )
        
        answer = response.content
        
        # 4. 构建结果
        matched_entities = [e.to_dict() for e in graph_result.matched_entities]
        
        related_relations = []
        if graph_result.subgraph:
            related_relations = [r.to_dict() for r in graph_result.subgraph.edges[:10]]
        
        sources = []
        for doc in vector_docs:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("file_name", "未知"),
                "type": "document",
            })
        for entity in graph_result.matched_entities:
            sources.append({
                "content": entity.description,
                "source": entity.name,
                "type": "entity",
            })
        
        result = GraphRAGResult(
            answer=answer,
            query=question,
            graph_context=graph_context,
            vector_context=vector_context,
            matched_entities=matched_entities,
            related_relations=related_relations,
            sources=sources,
            confidence=graph_result.confidence,
        )
        
        logger.info(f"✅ GraphRAG 问答完成")
        return result
    
    def ask_simple(self, question: str) -> str:
        """简化版问答，只返回答案"""
        result = self.ask(question)
        return result.answer
    
    def generate_global_summary(self) -> str:
        """
        生成知识图谱的全局摘要
        
        Returns:
            str: 全局摘要
        """
        stats = self._knowledge_graph.get_statistics()
        
        # 获取主要实体
        all_entities = self._knowledge_graph.get_all_entities()
        top_entities = "\n".join([
            f"- {e.name} ({e.type.value}): {e.description}"
            for e in all_entities[:20]
        ])
        
        # 获取核心关系
        all_relations = self._knowledge_graph.get_all_relations()
        top_relations = "\n".join([
            f"- {r.source} --[{r.relation_type.value}]--> {r.target}"
            for r in all_relations[:20]
        ])
        
        prompt = ChatPromptTemplate.from_template(GLOBAL_SUMMARY_PROMPT)
        
        response = self._llm.invoke(
            prompt.format(
                num_entities=stats["num_nodes"],
                num_relations=stats["num_edges"],
                entity_types=", ".join(stats["entity_type_counts"].keys()),
                relation_types=", ".join(stats["relation_type_counts"].keys()),
                top_entities=top_entities or "无",
                top_relations=top_relations or "无",
            )
        )
        
        return response.content
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        获取实体详细信息
        
        Args:
            entity_name: 实体名称
            
        Returns:
            Optional[Dict]: 实体信息
        """
        entity = self._knowledge_graph.get_entity_by_name(entity_name)
        if not entity:
            return None
        
        subgraph = self._knowledge_graph.get_neighbors(entity_name, hops=1)
        
        return {
            "entity": entity.to_dict(),
            "neighbors": [e.to_dict() for e in subgraph.nodes if e.id != entity.id],
            "relations": [r.to_dict() for r in subgraph.edges],
        }
    
    def find_path(
        self,
        source_name: str,
        target_name: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        查找两个实体之间的路径
        
        Args:
            source_name: 源实体名称
            target_name: 目标实体名称
            
        Returns:
            Optional[List]: 路径信息
        """
        path = self._knowledge_graph.find_path(source_name, target_name)
        
        if not path:
            return None
        
        return [
            {
                "entity": entity.to_dict(),
                "relation": relation.to_dict() if relation else None,
            }
            for entity, relation in path
        ]
    
    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """获取知识图谱"""
        return self._knowledge_graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return self._knowledge_graph.get_statistics()

