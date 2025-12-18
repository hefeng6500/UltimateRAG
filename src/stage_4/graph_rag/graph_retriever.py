"""
图检索器

基于知识图谱进行智能检索。
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import re

from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.stage_4.config import Stage4Config, get_stage4_config
from .knowledge_graph import KnowledgeGraph, SubGraph
from .entity_extractor import Entity, EntityExtractor
from .relation_extractor import Relation


@dataclass
class GraphRetrievalResult:
    """
    图检索结果
    
    Attributes:
        query: 原始查询
        matched_entities: 匹配到的实体
        subgraph: 相关子图
        context: 生成的上下文文本
        paths: 找到的路径（如果查询涉及两个实体）
        confidence: 置信度
    """
    query: str
    matched_entities: List[Entity] = field(default_factory=list)
    subgraph: Optional[SubGraph] = None
    context: str = ""
    paths: List[List[Tuple[Entity, Optional[Relation]]]] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_documents(self) -> List[Document]:
        """转换为 Document 列表"""
        docs = []
        
        if self.context:
            docs.append(Document(
                page_content=self.context,
                metadata={
                    "source": "knowledge_graph",
                    "type": "graph_context",
                    "matched_entities": [e.name for e in self.matched_entities],
                }
            ))
        
        return docs


# 实体识别提示词
ENTITY_RECOGNITION_PROMPT = """从用户的问题中识别出可能存在于知识图谱中的实体名称。

用户问题：{query}

请列出问题中提到的所有可能是实体的词语（人名、公司名、地名、产品名等），每行一个。
如果没有明确的实体，输出"无"。

实体列表："""


class GraphRetriever:
    """
    图检索器
    
    基于知识图谱进行智能检索，支持：
    - 实体匹配
    - 邻域探索
    - 路径查找
    - 子图提取
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        config: Optional[Stage4Config] = None,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        初始化图检索器
        
        Args:
            knowledge_graph: 知识图谱
            config: 配置
            llm: LLM 实例
        """
        self.config = config or get_stage4_config()
        self.graph = knowledge_graph
        self._llm = llm or self._create_llm()
        self._entity_prompt = ChatPromptTemplate.from_template(ENTITY_RECOGNITION_PROMPT)
        
        logger.info("🔍 图检索器初始化完成")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        从查询中提取实体名称
        
        Args:
            query: 用户查询
            
        Returns:
            List[str]: 实体名称列表
        """
        try:
            # 使用 LLM 识别实体
            prompt = self._entity_prompt.format(query=query)
            response = self._llm.invoke(prompt)
            
            # 解析响应
            lines = response.content.strip().split('\n')
            entities = []
            
            for line in lines:
                line = line.strip().lstrip('-').lstrip('•').strip()
                if line and line != "无":
                    entities.append(line)
            
            return entities
            
        except Exception as e:
            logger.warning(f"LLM 实体提取失败: {e}")
            # 降级：简单的关键词提取
            return self._simple_entity_extraction(query)
    
    def _simple_entity_extraction(self, query: str) -> List[str]:
        """简单的实体提取（作为降级方案）"""
        # 移除常见停用词
        stopwords = {'的', '是', '在', '有', '和', '与', '了', '吗', '呢', '什么', '怎么', '如何', '为什么'}
        
        # 分词（简单按空格和标点分割）
        words = re.split(r'[\s,，。？！、]+', query)
        
        # 过滤
        entities = [w for w in words if w and len(w) > 1 and w not in stopwords]
        
        return entities
    
    def _match_entities(self, query_entities: List[str]) -> List[Entity]:
        """
        将查询中的实体与图谱中的实体匹配
        
        Args:
            query_entities: 查询中提取的实体名称
            
        Returns:
            List[Entity]: 匹配到的实体
        """
        matched = []
        
        for name in query_entities:
            # 精确匹配
            entity = self.graph.get_entity_by_name(name)
            if entity:
                matched.append(entity)
                continue
            
            # 模糊匹配
            candidates = self.graph.search_entities(name, top_k=3)
            for candidate in candidates:
                # 计算相似度（简单的包含关系）
                if name.lower() in candidate.name.lower() or candidate.name.lower() in name.lower():
                    if candidate not in matched:
                        matched.append(candidate)
                        break
        
        return matched
    
    def _generate_context(
        self,
        matched_entities: List[Entity],
        subgraph: SubGraph,
        paths: List[List[Tuple[Entity, Optional[Relation]]]],
    ) -> str:
        """
        生成上下文文本
        
        Args:
            matched_entities: 匹配的实体
            subgraph: 相关子图
            paths: 路径信息
            
        Returns:
            str: 上下文文本
        """
        context_parts = []
        
        # 1. 实体信息
        if matched_entities:
            context_parts.append("【相关实体】")
            for entity in matched_entities:
                info = f"- {entity.name} ({entity.type.value})"
                if entity.description:
                    info += f": {entity.description}"
                context_parts.append(info)
        
        # 2. 关系信息
        if subgraph and subgraph.edges:
            context_parts.append("\n【实体关系】")
            for relation in subgraph.edges[:10]:  # 限制数量
                info = f"- {relation.source} --[{relation.relation_type.value}]--> {relation.target}"
                if relation.description:
                    info += f" ({relation.description})"
                context_parts.append(info)
        
        # 3. 路径信息
        if paths:
            context_parts.append("\n【关系路径】")
            for i, path in enumerate(paths[:3], 1):  # 限制路径数量
                path_str = " -> ".join(
                    f"{e.name}[{r.relation_type.value if r else '起点'}]"
                    for e, r in path
                )
                context_parts.append(f"路径 {i}: {path_str}")
        
        return "\n".join(context_parts)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hops: int = None,
    ) -> GraphRetrievalResult:
        """
        执行图检索
        
        Args:
            query: 用户查询
            top_k: 返回的最大实体数
            hops: 探索的跳数
            
        Returns:
            GraphRetrievalResult: 检索结果
        """
        hops = hops or self.config.graph_traversal_depth
        
        logger.info(f"🔍 图检索: {query}")
        
        # 1. 从查询中提取实体
        query_entities = self._extract_entities_from_query(query)
        logger.debug(f"提取的查询实体: {query_entities}")
        
        # 2. 匹配图谱中的实体
        matched_entities = self._match_entities(query_entities)
        logger.debug(f"匹配到的实体: {[e.name for e in matched_entities]}")
        
        if not matched_entities:
            logger.warning("未匹配到任何实体")
            return GraphRetrievalResult(
                query=query,
                matched_entities=[],
                context="知识图谱中未找到相关实体。",
                confidence=0.0,
            )
        
        # 3. 获取相关子图
        all_nodes = []
        all_edges = []
        
        for entity in matched_entities[:top_k]:
            sub = self.graph.get_neighbors(entity.name, hops=hops)
            all_nodes.extend(sub.nodes)
            all_edges.extend(sub.edges)
        
        # 去重
        unique_nodes = list({e.id: e for e in all_nodes}.values())
        unique_edges = list({r.id: r for r in all_edges}.values())
        
        subgraph = SubGraph(nodes=unique_nodes, edges=unique_edges)
        
        # 4. 如果有多个实体，尝试查找路径
        paths = []
        if len(matched_entities) >= 2:
            for i in range(len(matched_entities)):
                for j in range(i + 1, len(matched_entities)):
                    path = self.graph.find_path(
                        matched_entities[i].name,
                        matched_entities[j].name,
                        max_depth=hops + 2,
                    )
                    if path:
                        paths.append(path)
        
        # 5. 生成上下文
        context = self._generate_context(matched_entities, subgraph, paths)
        
        # 6. 计算置信度
        confidence = min(1.0, len(matched_entities) / max(len(query_entities), 1))
        
        result = GraphRetrievalResult(
            query=query,
            matched_entities=matched_entities,
            subgraph=subgraph,
            context=context,
            paths=paths,
            confidence=confidence,
        )
        
        logger.info(
            f"✅ 图检索完成: {len(matched_entities)} 个实体, "
            f"{len(subgraph.edges)} 条关系, "
            f"{len(paths)} 条路径"
        )
        
        return result
    
    def retrieve_by_entity(
        self,
        entity_name: str,
        hops: int = None,
    ) -> GraphRetrievalResult:
        """
        按实体名称检索
        
        Args:
            entity_name: 实体名称
            hops: 探索跳数
            
        Returns:
            GraphRetrievalResult: 检索结果
        """
        hops = hops or self.config.graph_traversal_depth
        
        entity = self.graph.get_entity_by_name(entity_name)
        if not entity:
            return GraphRetrievalResult(
                query=entity_name,
                context=f"未找到实体: {entity_name}",
                confidence=0.0,
            )
        
        subgraph = self.graph.get_neighbors(entity_name, hops=hops)
        context = self._generate_context([entity], subgraph, [])
        
        return GraphRetrievalResult(
            query=entity_name,
            matched_entities=[entity],
            subgraph=subgraph,
            context=context,
            confidence=1.0,
        )
    
    def retrieve_path(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 5,
    ) -> GraphRetrievalResult:
        """
        检索两个实体之间的路径
        
        Args:
            source_name: 源实体名称
            target_name: 目标实体名称
            max_depth: 最大搜索深度
            
        Returns:
            GraphRetrievalResult: 检索结果
        """
        source = self.graph.get_entity_by_name(source_name)
        target = self.graph.get_entity_by_name(target_name)
        
        if not source or not target:
            return GraphRetrievalResult(
                query=f"{source_name} -> {target_name}",
                context="源或目标实体不存在",
                confidence=0.0,
            )
        
        path = self.graph.find_path(source_name, target_name, max_depth)
        
        if not path:
            return GraphRetrievalResult(
                query=f"{source_name} -> {target_name}",
                matched_entities=[source, target],
                context=f"未找到 {source_name} 到 {target_name} 的路径",
                confidence=0.5,
            )
        
        # 提取路径中的所有实体和关系
        entities = [e for e, _ in path]
        relations = [r for _, r in path if r]
        
        subgraph = SubGraph(nodes=entities, edges=relations)
        context = self._generate_context([source, target], subgraph, [path])
        
        return GraphRetrievalResult(
            query=f"{source_name} -> {target_name}",
            matched_entities=[source, target],
            subgraph=subgraph,
            context=context,
            paths=[path],
            confidence=1.0,
        )

