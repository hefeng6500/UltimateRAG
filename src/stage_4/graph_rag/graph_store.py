"""
图存储模块

提供知识图谱的持久化存储，支持：
- 内存存储（开发/测试用）
- Neo4j 存储（生产环境）
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os
import json

from loguru import logger

from src.stage_4.config import Stage4Config, get_stage4_config
from .knowledge_graph import KnowledgeGraph
from .entity_extractor import Entity, EntityType
from .relation_extractor import Relation, RelationType


class GraphStore(ABC):
    """
    图存储抽象基类
    
    定义图存储的通用接口。
    """
    
    @abstractmethod
    def save(self, graph: KnowledgeGraph, name: str = "default"):
        """保存知识图谱"""
        pass
    
    @abstractmethod
    def load(self, name: str = "default") -> Optional[KnowledgeGraph]:
        """加载知识图谱"""
        pass
    
    @abstractmethod
    def exists(self, name: str = "default") -> bool:
        """检查图谱是否存在"""
        pass
    
    @abstractmethod
    def delete(self, name: str = "default"):
        """删除知识图谱"""
        pass
    
    @abstractmethod
    def list_graphs(self) -> List[str]:
        """列出所有图谱"""
        pass


class MemoryGraphStore(GraphStore):
    """
    内存图存储
    
    将知识图谱存储在内存中，支持持久化到文件。
    适合开发和测试环境。
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        config: Optional[Stage4Config] = None,
    ):
        """
        初始化内存图存储
        
        Args:
            persist_dir: 持久化目录
            config: 配置
        """
        self.config = config or get_stage4_config()
        self.persist_dir = persist_dir or self.config.graph_persist_dir
        
        # 内存存储
        self._graphs: Dict[str, KnowledgeGraph] = {}
        
        # 确保目录存在
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # 尝试从磁盘加载
        self._load_from_disk()
        
        logger.info(f"📁 内存图存储初始化完成: {self.persist_dir}")
    
    def _get_filepath(self, name: str) -> str:
        """获取图谱文件路径"""
        return os.path.join(self.persist_dir, f"{name}.json")
    
    def _load_from_disk(self):
        """从磁盘加载所有图谱"""
        if not os.path.exists(self.persist_dir):
            return
        
        for filename in os.listdir(self.persist_dir):
            if filename.endswith('.json'):
                name = filename[:-5]
                filepath = os.path.join(self.persist_dir, filename)
                try:
                    self._graphs[name] = KnowledgeGraph.load(filepath)
                    logger.debug(f"已加载图谱: {name}")
                except Exception as e:
                    logger.warning(f"加载图谱失败: {name}, 错误: {e}")
    
    def save(self, graph: KnowledgeGraph, name: str = "default"):
        """保存知识图谱"""
        self._graphs[name] = graph
        
        # 持久化到文件
        filepath = self._get_filepath(name)
        graph.save(filepath)
        
        logger.info(f"图谱已保存: {name} (节点: {graph.num_nodes}, 边: {graph.num_edges})")
    
    def load(self, name: str = "default") -> Optional[KnowledgeGraph]:
        """加载知识图谱"""
        if name in self._graphs:
            return self._graphs[name]
        
        # 尝试从文件加载
        filepath = self._get_filepath(name)
        if os.path.exists(filepath):
            self._graphs[name] = KnowledgeGraph.load(filepath)
            return self._graphs[name]
        
        return None
    
    def exists(self, name: str = "default") -> bool:
        """检查图谱是否存在"""
        return name in self._graphs or os.path.exists(self._get_filepath(name))
    
    def delete(self, name: str = "default"):
        """删除知识图谱"""
        if name in self._graphs:
            del self._graphs[name]
        
        filepath = self._get_filepath(name)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"图谱已删除: {name}")
    
    def list_graphs(self) -> List[str]:
        """列出所有图谱"""
        names = set(self._graphs.keys())
        
        # 添加磁盘上的图谱
        if os.path.exists(self.persist_dir):
            for filename in os.listdir(self.persist_dir):
                if filename.endswith('.json'):
                    names.add(filename[:-5])
        
        return list(names)
    
    def get_or_create(self, name: str = "default") -> KnowledgeGraph:
        """获取或创建图谱"""
        graph = self.load(name)
        if graph is None:
            graph = KnowledgeGraph()
            self._graphs[name] = graph
        return graph


class Neo4jGraphStore(GraphStore):
    """
    Neo4j 图存储
    
    将知识图谱存储在 Neo4j 图数据库中。
    适合生产环境和大规模数据。
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        config: Optional[Stage4Config] = None,
    ):
        """
        初始化 Neo4j 图存储
        
        Args:
            uri: Neo4j URI
            username: 用户名
            password: 密码
            database: 数据库名称
            config: 配置
        """
        self.config = config or get_stage4_config()
        
        self.uri = uri or self.config.neo4j_uri
        self.username = username or self.config.neo4j_username
        self.password = password or self.config.neo4j_password
        self.database = database or self.config.neo4j_database
        
        self._driver = None
        self._connect()
        
        logger.info(f"🔗 Neo4j 图存储初始化完成: {self.uri}, 数据库: {self.database}")
    
    def _connect(self):
        """连接到 Neo4j"""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # 测试连接（指定数据库）
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j 连接成功，数据库: {self.database}")
        except ImportError:
            logger.error("请安装 neo4j 包: pip install neo4j")
            raise
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            raise
    
    def _get_session(self):
        """获取 Neo4j session"""
        if self._driver is None:
            self._connect()
        return self._driver.session(database=self.database)
    
    def save(self, graph: KnowledgeGraph, name: str = "default"):
        """保存知识图谱到 Neo4j"""
        with self._get_session() as session:
            # 清除现有图谱（按名称标记）
            session.run(
                "MATCH (n {graph_name: $name}) DETACH DELETE n",
                name=name
            )
            
            # 添加实体（节点）- 使用实体类型作为额外标签
            for entity in graph.get_all_entities():
                # 动态添加实体类型标签 (Entity:Person, Entity:Organization 等)
                entity_label = entity.type.value
                query = f"""
                CREATE (n:Entity:{entity_label} {{
                    id: $id,
                    name: $name,
                    type: $type,
                    description: $description,
                    aliases: $aliases,
                    graph_name: $graph_name
                }})
                """
                session.run(
                    query,
                    id=entity.id,
                    name=entity.name,
                    type=entity.type.value,
                    description=entity.description,
                    aliases=entity.aliases,
                    graph_name=name,
                )
            
            # 添加关系（边）- 使用关系类型作为 Neo4j 关系类型
            for relation in graph.get_all_relations():
                # 动态创建关系类型 (FOUNDED, MANAGES, LOCATED_IN 等)
                rel_type = relation.relation_type.value.upper()
                query = f"""
                MATCH (s:Entity {{name: $source, graph_name: $graph_name}})
                MATCH (t:Entity {{name: $target, graph_name: $graph_name}})
                CREATE (s)-[r:{rel_type} {{
                    id: $id,
                    description: $description,
                    confidence: $confidence
                }}]->(t)
                """
                session.run(
                    query,
                    id=relation.id,
                    source=relation.source,
                    target=relation.target,
                    description=relation.description,
                    confidence=relation.confidence,
                    graph_name=name,
                )
            
            logger.info(
                f"✅ 图谱已保存到 Neo4j: {name} "
                f"(节点: {graph.num_nodes}, 边: {graph.num_edges})"
            )
    
    def load(self, name: str = "default") -> Optional[KnowledgeGraph]:
        """从 Neo4j 加载知识图谱"""
        graph = KnowledgeGraph()
        
        with self._get_session() as session:
            # 加载实体
            result = session.run(
                "MATCH (n:Entity {graph_name: $name}) RETURN n",
                name=name
            )
            
            entities = []
            for record in result:
                node = record["n"]
                entity = Entity(
                    name=node["name"],
                    type=EntityType(node["type"]),
                    description=node.get("description", ""),
                    aliases=node.get("aliases", []),
                )
                entities.append(entity)
                graph.add_entity(entity)
            
            if not entities:
                return None
            
            # 加载关系 - 获取所有类型的关系
            result = session.run(
                """
                MATCH (s:Entity {graph_name: $name})-[r]->(t:Entity {graph_name: $name})
                RETURN s.name as source, t.name as target, type(r) as rel_type, 
                       r.description as description, r.confidence as confidence
                """,
                name=name
            )
            
            for record in result:
                rel_type_str = record["rel_type"].lower()
                # 尝试匹配关系类型枚举
                try:
                    rel_type = RelationType(rel_type_str)
                except ValueError:
                    rel_type = RelationType.RELATED_TO
                
                relation = Relation(
                    source=record["source"],
                    target=record["target"],
                    relation_type=rel_type,
                    description=record.get("description", "") or "",
                    confidence=record.get("confidence", 1.0) or 1.0,
                )
                graph.add_relation(relation)
            
            logger.info(
                f"✅ 从 Neo4j 加载图谱: {name} "
                f"(节点: {graph.num_nodes}, 边: {graph.num_edges})"
            )
            return graph
    
    def exists(self, name: str = "default") -> bool:
        """检查图谱是否存在"""
        with self._get_session() as session:
            result = session.run(
                "MATCH (n:Entity {graph_name: $name}) RETURN count(n) as count",
                name=name
            )
            count = result.single()["count"]
            return count > 0
    
    def delete(self, name: str = "default"):
        """删除知识图谱"""
        with self._get_session() as session:
            session.run(
                "MATCH (n {graph_name: $name}) DETACH DELETE n",
                name=name
            )
            logger.info(f"Neo4j 图谱已删除: {name}")
    
    def list_graphs(self) -> List[str]:
        """列出所有图谱"""
        with self._get_session() as session:
            result = session.run(
                "MATCH (n:Entity) RETURN DISTINCT n.graph_name as name"
            )
            return [record["name"] for record in result if record["name"]]
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def __del__(self):
        self.close()


def create_graph_store(
    store_type: Optional[str] = None,
    config: Optional[Stage4Config] = None,
    **kwargs,
) -> GraphStore:
    """
    创建图存储实例
    
    Args:
        store_type: 存储类型 (memory / neo4j)
        config: 配置
        **kwargs: 额外参数
        
    Returns:
        GraphStore: 图存储实例
    """
    config = config or get_stage4_config()
    store_type = store_type or config.graph_store_type
    
    if store_type == "memory":
        return MemoryGraphStore(config=config, **kwargs)
    elif store_type == "neo4j":
        return Neo4jGraphStore(config=config, **kwargs)
    else:
        logger.warning(f"未知存储类型: {store_type}，使用内存存储")
        return MemoryGraphStore(config=config, **kwargs)

