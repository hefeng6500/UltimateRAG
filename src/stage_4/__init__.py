"""
Stage 4: GraphRAG & Fine-tuning

最高阶段的 RAG 系统，包含：
- 知识图谱增强的 RAG (GraphRAG)
- Embedding 模型微调
- LLM 微调数据准备
"""

from .config import Stage4Config, get_stage4_config

__all__ = [
    "Stage4Config",
    "get_stage4_config",
]

