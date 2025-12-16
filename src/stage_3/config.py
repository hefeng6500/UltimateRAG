"""
Stage 3 配置模块

扩展 Stage 1 的基础配置，增加 Agentic RAG 相关参数。
"""

from dataclasses import dataclass, field
from typing import Optional, List
from src.stage_1.config import Config, get_config


@dataclass
class Stage3Config(Config):
    """
    Stage 3 扩展配置
    
    继承基础配置，增加 Agentic RAG 特有参数。
    """
    
    # Self-RAG 配置
    self_rag_max_iterations: int = 3  # 最大自反思迭代次数
    self_rag_quality_threshold: float = 0.7  # 答案质量阈值
    self_rag_relevance_threshold: float = 0.5  # 检索相关性阈值
    
    # 父子索引配置
    parent_chunk_size: int = 2000  # 父块大小
    child_chunk_size: int = 400   # 子块大小
    child_chunk_overlap: int = 50  # 子块重叠
    
    # 上下文压缩配置
    compression_ratio: float = 0.5  # 目标压缩比例
    min_relevant_score: float = 0.3  # 最小相关性分数
    
    # 工具配置
    enable_web_search: bool = True
    enable_calculator: bool = True
    enable_code_executor: bool = True
    code_executor_timeout: int = 10  # 代码执行超时（秒）
    
    # 路由配置
    router_confidence_threshold: float = 0.6  # 路由置信度阈值
    
    @classmethod
    def from_base_config(cls, base_config: Config = None) -> "Stage3Config":
        """
        从基础配置创建 Stage3 配置
        
        Args:
            base_config: 基础配置对象，若为 None 则自动加载
            
        Returns:
            Stage3Config: Stage3 配置实例
        """
        base = base_config or get_config()
        
        return cls(
            # 继承基础配置
            openai_api_key=base.openai_api_key,
            openai_base_url=base.openai_base_url,
            model_name=base.model_name,
            embedding_model=base.embedding_model,
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            top_k=base.top_k,
            chroma_persist_dir=base.chroma_persist_dir,
        )


# 全局配置实例
_stage3_config: Optional[Stage3Config] = None


def get_stage3_config() -> Stage3Config:
    """
    获取 Stage3 配置实例（单例模式）
    
    Returns:
        Stage3Config: 配置实例
    """
    global _stage3_config
    if _stage3_config is None:
        _stage3_config = Stage3Config.from_base_config()
    return _stage3_config

