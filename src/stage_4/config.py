"""
Stage 4 配置模块

扩展 Stage 3 的配置，增加 GraphRAG 和微调相关参数。
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv
from loguru import logger
from src.stage_3.config import Stage3Config, get_stage3_config


@dataclass
class Stage4Config(Stage3Config):
    """
    Stage 4 扩展配置
    
    继承 Stage 3 配置，增加 GraphRAG 和微调特有参数。
    """
    
    # ===================
    # GraphRAG 配置
    # ===================
    
    # 图存储类型: memory / neo4j
    graph_store_type: str = "memory"
    
    # 支持的实体类型
    entity_types: List[str] = field(default_factory=lambda: [
        "Person",       # 人物
        "Organization", # 组织/公司
        "Location",     # 地点
        "Event",        # 事件
        "Concept",      # 概念/术语
        "Product",      # 产品
        "Time",         # 时间
    ])
    
    # 支持的关系类型
    relation_types: List[str] = field(default_factory=lambda: [
        "belongs_to",      # 隶属关系
        "cooperates_with", # 合作关系
        "competes_with",   # 竞争关系
        "invests_in",      # 投资关系
        "manages",         # 管理关系
        "located_in",      # 位于
        "participates_in", # 参与
        "produces",        # 生产
        "founded",         # 创立
        "works_for",       # 就职于
        "related_to",      # 相关
    ])
    
    # 每个文档块最大实体数
    max_entities_per_chunk: int = 20
    
    # 每个文档块最大关系数
    max_relations_per_chunk: int = 30
    
    # 图遍历深度（跳数）
    graph_traversal_depth: int = 2
    
    # 实体匹配相似度阈值
    entity_similarity_threshold: float = 0.8
    
    # 关系置信度阈值
    relation_confidence_threshold: float = 0.6
    
    # 社区检测分辨率
    community_resolution: float = 1.0
    
    # ===================
    # Neo4j 配置
    # ===================
    
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    
    # ===================
    # Embedding 微调配置
    # ===================
    
    # 基础模型
    embedding_finetune_model: str = "BAAI/bge-base-zh-v1.5"
    
    # 微调输出目录
    embedding_finetune_output_dir: str = "./models/finetuned_embedding"
    
    # 训练轮数
    embedding_finetune_epochs: int = 3
    
    # 批次大小
    embedding_finetune_batch_size: int = 32
    
    # 学习率
    embedding_finetune_lr: float = 2e-5
    
    # 最大序列长度
    embedding_finetune_max_length: int = 512
    
    # ===================
    # LLM 微调数据配置
    # ===================
    
    # 每个文档生成的 QA 对数量
    qa_pairs_per_doc: int = 5
    
    # QA 难度级别
    qa_difficulty_levels: List[str] = field(default_factory=lambda: [
        "easy",   # 简单：直接从文本中找答案
        "medium", # 中等：需要简单推理
        "hard",   # 困难：需要综合多段信息
    ])
    
    # 微调数据输出目录
    finetune_data_output_dir: str = "./data/finetune"
    
    # ===================
    # 图持久化配置
    # ===================
    
    # 图数据存储目录
    graph_persist_dir: str = "./data/graph_db"
    
    @classmethod
    def from_stage3_config(cls, stage3_config: Stage3Config = None) -> "Stage4Config":
        """
        从 Stage3 配置创建 Stage4 配置
        
        Args:
            stage3_config: Stage3 配置对象，若为 None 则自动加载
            
        Returns:
            Stage4Config: Stage4 配置实例
        """
        # 加载 .env 文件（确保环境变量可用）
        load_dotenv()
        
        base = stage3_config or get_stage3_config()
        
        # 从环境变量加载 Stage4 特有配置
        config = cls(
            # 继承 Stage 1 配置
            openai_api_key=base.openai_api_key,
            openai_base_url=base.openai_base_url,
            model_name=base.model_name,
            embedding_model=base.embedding_model,
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            top_k=base.top_k,
            chroma_persist_dir=base.chroma_persist_dir,
            
            # 继承 Stage 3 配置
            self_rag_max_iterations=base.self_rag_max_iterations,
            self_rag_quality_threshold=base.self_rag_quality_threshold,
            self_rag_relevance_threshold=base.self_rag_relevance_threshold,
            parent_chunk_size=base.parent_chunk_size,
            child_chunk_size=base.child_chunk_size,
            child_chunk_overlap=base.child_chunk_overlap,
            compression_ratio=base.compression_ratio,
            min_relevant_score=base.min_relevant_score,
            enable_web_search=base.enable_web_search,
            enable_calculator=base.enable_calculator,
            enable_code_executor=base.enable_code_executor,
            code_executor_timeout=base.code_executor_timeout,
            router_confidence_threshold=base.router_confidence_threshold,
            
            # Stage 4 特有配置（从环境变量加载）
            graph_store_type=os.getenv("GRAPH_STORE_TYPE", "memory"),
            neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            embedding_finetune_model=os.getenv(
                "EMBEDDING_FINETUNE_MODEL", 
                "BAAI/bge-base-zh-v1.5"
            ),
            graph_persist_dir=os.getenv("GRAPH_PERSIST_DIR", "./data/graph_db"),
        )
        
        # 打印 Stage 4 特有配置
        logger.info(f"📊 Stage4 配置加载完成:")
        logger.info(f"   - 图存储类型: {config.graph_store_type}")
        if config.graph_store_type == "neo4j":
            logger.info(f"   - Neo4j URI: {config.neo4j_uri}")
            logger.info(f"   - Neo4j 用户: {config.neo4j_username}")
            logger.info(f"   - Neo4j 数据库: {config.neo4j_database}")
        
        return config


# 全局配置实例
_stage4_config: Optional[Stage4Config] = None


def get_stage4_config() -> Stage4Config:
    """
    获取 Stage4 配置实例（单例模式）
    
    Returns:
        Stage4Config: 配置实例
    """
    global _stage4_config
    if _stage4_config is None:
        _stage4_config = Stage4Config.from_stage3_config()
    return _stage4_config

