"""
Phase 3: 架构进化 (Modular & Agentic RAG)

本阶段实现让 RAG 系统具备"思考"能力的核心组件：
- 智能路由：根据问题类型分发到不同处理器
- 自反思 RAG：答案质量自评估和迭代优化
- 工具调用：搜索、计算、代码执行等能力
- 父子索引：精准检索 + 完整上下文
- 上下文压缩：精简无关内容，节省 Token
"""

from .config import Stage3Config, get_stage3_config
from .router import QueryRouter, RouteType
from .self_rag import SelfRAG
from .parent_child_retriever import ParentChildRetriever
from .context_compressor import ContextCompressor
from .agentic_rag_chain import AgenticRAGChain

__all__ = [
    "Stage3Config",
    "get_stage3_config",
    "QueryRouter",
    "RouteType",
    "SelfRAG",
    "ParentChildRetriever",
    "ContextCompressor",
    "AgenticRAGChain",
]

