"""
配置管理模块

负责加载环境变量和管理配置参数。
支持 OpenAI、DeepSeek 等多种 LLM 提供商。
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger


@dataclass
class Config:
    """
    RAG 系统配置类
    
    Attributes:
        openai_api_key: OpenAI API 密钥
        openai_base_url: OpenAI API 基础 URL（用于兼容 DeepSeek 等）
        model_name: LLM 模型名称
        embedding_model: Embedding 模型名称
        chunk_size: 文档分块大小（token 数）
        chunk_overlap: 分块重叠大小（token 数）
        top_k: 向量检索返回的文档数量
        chroma_persist_dir: ChromaDB 持久化目录
    """
    
    # API 配置
    openai_api_key: str = ""
    openai_base_url: Optional[str] = None
    
    # 模型配置
    model_name: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    
    # 分块配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # 检索配置
    top_k: int = 3
    
    # 存储配置
    chroma_persist_dir: str = "./data/chroma_db"
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量加载配置
        
        Returns:
            Config: 配置实例
        """
        # 加载 .env 文件
        load_dotenv()
        
        config = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            top_k=int(os.getenv("TOP_K", "3")),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        )
        
        # 验证必要配置
        if not config.openai_api_key:
            logger.warning("未设置 OPENAI_API_KEY，请在 .env 文件中配置")
        
        logger.info(f"✅ 配置加载完成: 模型={config.model_name}, 分块大小={config.chunk_size}")
        return config
    
    def validate(self) -> bool:
        """
        验证配置是否有效
        
        Returns:
            bool: 配置是否有效
        """
        if not self.openai_api_key:
            logger.error("❌ OPENAI_API_KEY 未设置")
            return False
        return True


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """
    获取全局配置实例（单例模式）
    
    Returns:
        Config: 配置实例
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
