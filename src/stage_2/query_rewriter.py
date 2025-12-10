"""
查询重写模块

实现多种查询重写策略来提升检索召回率：
- 多路查询生成
- HyDE (假设文档嵌入)
- 查询扩展
"""

from typing import List, Optional
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import sys
sys.path.append("..")
from stage_1.config import Config, get_config


class QueryRewriter:
    """
    查询重写器
    
    通过多种策略改写用户查询，提升检索效果。
    """
    
    # 多路查询生成提示词
    MULTI_QUERY_PROMPT = """你是一个专业的搜索查询优化专家。
请根据用户的原始问题，生成 3 个不同角度的搜索查询。
这些查询应该：
1. 保持原始问题的核心意图
2. 使用不同的表达方式
3. 可能包含同义词或相关概念

原始问题：{question}

请输出 3 个改写后的查询，每行一个，不要编号："""

    # HyDE 提示词：生成假设答案
    HYDE_PROMPT = """请针对以下问题，写一段简短的假设性回答（约 50-100 字）。
这个回答应该像是从一份专业文档中摘录的内容。

问题：{question}

假设回答："""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化查询重写器
        
        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        self._llm = self._create_llm()
        
        logger.info("✏️ 查询重写器初始化完成")
    
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
    
    def generate_multi_queries(self, query: str) -> List[str]:
        """
        生成多路查询
        
        Args:
            query: 原始查询
            
        Returns:
            List[str]: 改写后的查询列表（包含原始查询）
        """
        prompt = ChatPromptTemplate.from_template(self.MULTI_QUERY_PROMPT)
        
        try:
            response = self._llm.invoke(prompt.format(question=query))
            # 解析响应
            generated_queries = [
                q.strip() 
                for q in response.content.strip().split("\n") 
                if q.strip()
            ]
            
            # 加上原始查询
            all_queries = [query] + generated_queries[:3]
            
            logger.info(f"🔄 多路查询生成: 1 -> {len(all_queries)} 个查询")
            for i, q in enumerate(all_queries):
                logger.debug(f"  [{i}] {q[:50]}...")
            
            return all_queries
        except Exception as e:
            logger.warning(f"⚠️ 多路查询生成失败: {e}")
            return [query]
    
    def generate_hyde_query(self, query: str) -> str:
        """
        生成 HyDE 假设文档
        
        HyDE (Hypothetical Document Embeddings):
        先让 LLM 生成一个假设的答案，用这个假设答案去检索真实文档。
        
        Args:
            query: 原始查询
            
        Returns:
            str: 假设文档内容
        """
        prompt = ChatPromptTemplate.from_template(self.HYDE_PROMPT)
        
        try:
            response = self._llm.invoke(prompt.format(question=query))
            hyde_doc = response.content.strip()
            
            logger.info(f"📝 HyDE 假设文档生成完成: {len(hyde_doc)} 字符")
            logger.debug(f"  假设文档: {hyde_doc[:100]}...")
            
            return hyde_doc
        except Exception as e:
            logger.warning(f"⚠️ HyDE 生成失败: {e}")
            return query
    
    def expand_query(self, query: str) -> str:
        """
        查询扩展：添加同义词和相关概念
        
        Args:
            query: 原始查询
            
        Returns:
            str: 扩展后的查询
        """
        # 简单的中英文同义词扩展
        expansions = {
            "RAG": "检索增强生成 Retrieval-Augmented Generation",
            "LLM": "大语言模型 Large Language Model",
            "向量": "vector embedding 嵌入",
            "检索": "retrieval search 搜索",
            "分块": "chunking segmentation 切分",
        }
        
        expanded = query
        for key, value in expansions.items():
            if key.lower() in query.lower():
                expanded += f" {value}"
        
        if expanded != query:
            logger.info(f"🔍 查询扩展: {query[:30]}... -> {expanded[:50]}...")
        
        return expanded
    
    def rewrite(
        self,
        query: str,
        strategy: str = "multi_query"
    ) -> List[str]:
        """
        执行查询重写
        
        Args:
            query: 原始查询
            strategy: 重写策略，可选 "multi_query", "hyde", "expand", "all"
            
        Returns:
            List[str]: 重写后的查询列表
        """
        if strategy == "multi_query":
            return self.generate_multi_queries(query)
        
        elif strategy == "hyde":
            hyde_doc = self.generate_hyde_query(query)
            return [query, hyde_doc]
        
        elif strategy == "expand":
            expanded = self.expand_query(query)
            return [query, expanded] if expanded != query else [query]
        
        elif strategy == "all":
            results = set([query])
            
            # 多路查询
            results.update(self.generate_multi_queries(query))
            
            # HyDE
            hyde_doc = self.generate_hyde_query(query)
            results.add(hyde_doc)
            
            # 扩展
            expanded = self.expand_query(query)
            results.add(expanded)
            
            return list(results)
        
        else:
            logger.warning(f"⚠️ 未知的重写策略: {strategy}")
            return [query]
