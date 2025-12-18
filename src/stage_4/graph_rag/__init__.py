"""
GraphRAG 模块

提供知识图谱增强的 RAG 能力：
- 实体抽取
- 关系抽取
- 知识图谱构建与管理
- 图检索
"""

from .entity_extractor import (
    Entity,
    EntityType,
    EntityExtractor,
)
from .relation_extractor import (
    Relation,
    RelationType,
    RelationExtractor,
)
from .knowledge_graph import KnowledgeGraph
from .graph_store import (
    GraphStore,
    MemoryGraphStore,
    Neo4jGraphStore,
    create_graph_store,
)
from .graph_retriever import GraphRetriever, GraphRetrievalResult
from .graph_rag_chain import GraphRAGChain, GraphRAGResult

__all__ = [
    # 实体相关
    "Entity",
    "EntityType",
    "EntityExtractor",
    # 关系相关
    "Relation",
    "RelationType",
    "RelationExtractor",
    # 知识图谱
    "KnowledgeGraph",
    # 图存储
    "GraphStore",
    "MemoryGraphStore",
    "Neo4jGraphStore",
    "create_graph_store",
    # 图检索
    "GraphRetriever",
    "GraphRetrievalResult",
    # GraphRAG 链
    "GraphRAGChain",
    "GraphRAGResult",
]

