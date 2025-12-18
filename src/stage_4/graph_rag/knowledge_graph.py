"""
知识图谱核心类

管理实体和关系的图结构，提供图操作接口。
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

from loguru import logger

from .entity_extractor import Entity, EntityType
from .relation_extractor import Relation, RelationType


@dataclass
class GraphNode:
    """图节点（实体的图表示）"""
    entity: Entity
    in_edges: Set[str] = field(default_factory=set)   # 入边ID集合
    out_edges: Set[str] = field(default_factory=set)  # 出边ID集合


@dataclass
class GraphEdge:
    """图边（关系的图表示）"""
    relation: Relation
    source_id: str  # 源节点ID
    target_id: str  # 目标节点ID


@dataclass
class SubGraph:
    """子图"""
    nodes: List[Entity]
    edges: List[Relation]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


class KnowledgeGraph:
    """
    知识图谱
    
    管理实体（节点）和关系（边）的图结构。
    """
    
    def __init__(self):
        """初始化知识图谱"""
        # 节点存储: entity_id -> GraphNode
        self._nodes: Dict[str, GraphNode] = {}
        
        # 边存储: relation_id -> GraphEdge
        self._edges: Dict[str, GraphEdge] = {}
        
        # 名称索引: entity_name.lower() -> entity_id
        self._name_index: Dict[str, str] = {}
        
        # 类型索引: entity_type -> set of entity_ids
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        
        # 邻接表: entity_id -> set of (relation_id, neighbor_id)
        self._adjacency: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        
        logger.info("📊 知识图谱初始化完成")
    
    @property
    def num_nodes(self) -> int:
        """节点数量"""
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        """边数量"""
        return len(self._edges)
    
    def add_entity(self, entity: Entity) -> str:
        """
        添加实体到图中
        
        Args:
            entity: 实体对象
            
        Returns:
            str: 实体ID
        """
        entity_id = entity.id
        
        if entity_id in self._nodes:
            logger.debug(f"实体已存在: {entity.name}")
            return entity_id
        
        # 创建节点
        node = GraphNode(entity=entity)
        self._nodes[entity_id] = node
        
        # 更新名称索引
        self._name_index[entity.name.lower()] = entity_id
        for alias in entity.aliases:
            self._name_index[alias.lower()] = entity_id
        
        # 更新类型索引
        self._type_index[entity.type].add(entity_id)
        
        logger.debug(f"添加实体: {entity.name} ({entity.type.value})")
        return entity_id
    
    def add_relation(self, relation: Relation) -> Optional[str]:
        """
        添加关系到图中
        
        Args:
            relation: 关系对象
            
        Returns:
            Optional[str]: 关系ID，如果添加失败则返回 None
        """
        # 查找源和目标节点
        source_id = self._name_index.get(relation.source.lower())
        target_id = self._name_index.get(relation.target.lower())
        
        if not source_id or not target_id:
            logger.warning(
                f"关系添加失败：源或目标实体不存在: "
                f"{relation.source} -> {relation.target}"
            )
            return None
        
        relation_id = relation.id
        
        if relation_id in self._edges:
            logger.debug(f"关系已存在: {relation.source} -> {relation.target}")
            return relation_id
        
        # 创建边
        edge = GraphEdge(
            relation=relation,
            source_id=source_id,
            target_id=target_id,
        )
        self._edges[relation_id] = edge
        
        # 更新节点的边集合
        self._nodes[source_id].out_edges.add(relation_id)
        self._nodes[target_id].in_edges.add(relation_id)
        
        # 更新邻接表
        self._adjacency[source_id].add((relation_id, target_id))
        self._adjacency[target_id].add((relation_id, source_id))
        
        logger.debug(
            f"添加关系: {relation.source} --[{relation.relation_type.value}]--> "
            f"{relation.target}"
        )
        return relation_id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        node = self._nodes.get(entity_id)
        return node.entity if node else None
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """通过名称获取实体"""
        entity_id = self._name_index.get(name.lower())
        return self.get_entity(entity_id) if entity_id else None
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """获取关系"""
        edge = self._edges.get(relation_id)
        return edge.relation if edge else None
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """获取指定类型的所有实体"""
        entity_ids = self._type_index.get(entity_type, set())
        return [self._nodes[eid].entity for eid in entity_ids]
    
    def get_all_entities(self) -> List[Entity]:
        """获取所有实体"""
        return [node.entity for node in self._nodes.values()]
    
    def get_all_relations(self) -> List[Relation]:
        """获取所有关系"""
        return [edge.relation for edge in self._edges.values()]
    
    def get_neighbors(
        self,
        entity_name: str,
        hops: int = 1,
        relation_types: Optional[List[RelationType]] = None,
    ) -> SubGraph:
        """
        获取实体的邻居（多跳）
        
        Args:
            entity_name: 实体名称
            hops: 跳数
            relation_types: 限制的关系类型
            
        Returns:
            SubGraph: 包含邻居节点和边的子图
        """
        entity_id = self._name_index.get(entity_name.lower())
        if not entity_id:
            logger.warning(f"实体不存在: {entity_name}")
            return SubGraph(nodes=[], edges=[])
        
        visited_nodes: Set[str] = {entity_id}
        visited_edges: Set[str] = set()
        current_layer = {entity_id}
        
        for _ in range(hops):
            next_layer = set()
            for node_id in current_layer:
                for rel_id, neighbor_id in self._adjacency.get(node_id, set()):
                    # 检查关系类型
                    if relation_types:
                        edge = self._edges.get(rel_id)
                        if edge and edge.relation.relation_type not in relation_types:
                            continue
                    
                    if neighbor_id not in visited_nodes:
                        next_layer.add(neighbor_id)
                        visited_nodes.add(neighbor_id)
                    
                    if rel_id not in visited_edges:
                        visited_edges.add(rel_id)
            
            current_layer = next_layer
            if not current_layer:
                break
        
        # 构建子图
        nodes = [self._nodes[nid].entity for nid in visited_nodes]
        edges = [self._edges[eid].relation for eid in visited_edges]
        
        return SubGraph(nodes=nodes, edges=edges)
    
    def find_path(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 5,
    ) -> Optional[List[Tuple[Entity, Optional[Relation]]]]:
        """
        查找两个实体之间的最短路径
        
        Args:
            source_name: 源实体名称
            target_name: 目标实体名称
            max_depth: 最大搜索深度
            
        Returns:
            Optional[List]: 路径列表，每个元素是 (实体, 关系) 元组
        """
        source_id = self._name_index.get(source_name.lower())
        target_id = self._name_index.get(target_name.lower())
        
        if not source_id or not target_id:
            logger.warning(f"源或目标实体不存在")
            return None
        
        if source_id == target_id:
            return [(self._nodes[source_id].entity, None)]
        
        # BFS 查找最短路径
        visited = {source_id}
        queue = [(source_id, [(source_id, None)])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for rel_id, neighbor_id in self._adjacency.get(current_id, set()):
                if neighbor_id == target_id:
                    # 找到目标
                    result = []
                    for node_id, prev_rel_id in path:
                        entity = self._nodes[node_id].entity
                        relation = self._edges[prev_rel_id].relation if prev_rel_id else None
                        result.append((entity, relation))
                    
                    # 添加最后一个节点和边
                    result.append((
                        self._nodes[target_id].entity,
                        self._edges[rel_id].relation,
                    ))
                    return result
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [(neighbor_id, rel_id)]
                    queue.append((neighbor_id, new_path))
        
        return None
    
    def get_subgraph(self, entity_names: List[str]) -> SubGraph:
        """
        获取包含指定实体的子图
        
        Args:
            entity_names: 实体名称列表
            
        Returns:
            SubGraph: 子图
        """
        entity_ids = set()
        for name in entity_names:
            eid = self._name_index.get(name.lower())
            if eid:
                entity_ids.add(eid)
        
        nodes = [self._nodes[eid].entity for eid in entity_ids]
        
        # 找出这些节点之间的所有边
        edges = []
        for rel_id, edge in self._edges.items():
            if edge.source_id in entity_ids and edge.target_id in entity_ids:
                edges.append(edge.relation)
        
        return SubGraph(nodes=nodes, edges=edges)
    
    def search_entities(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Entity]:
        """
        搜索实体（简单的名称匹配）
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            
        Returns:
            List[Entity]: 匹配的实体列表
        """
        query_lower = query.lower()
        results = []
        
        for name, entity_id in self._name_index.items():
            if query_lower in name:
                entity = self._nodes[entity_id].entity
                if entity not in results:
                    results.append(entity)
                    if len(results) >= top_k:
                        break
        
        return results
    
    def merge(self, other: "KnowledgeGraph"):
        """
        合并另一个知识图谱
        
        Args:
            other: 另一个知识图谱
        """
        # 添加所有实体
        for entity in other.get_all_entities():
            self.add_entity(entity)
        
        # 添加所有关系
        for relation in other.get_all_relations():
            self.add_relation(relation)
        
        logger.info(
            f"合并知识图谱: 当前节点数 {self.num_nodes}, 边数 {self.num_edges}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "nodes": [node.entity.to_dict() for node in self._nodes.values()],
            "edges": [edge.relation.to_dict() for edge in self._edges.values()],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """从字典创建知识图谱"""
        kg = cls()
        
        # 添加实体
        for node_data in data.get("nodes", []):
            entity = Entity.from_dict(node_data)
            kg.add_entity(entity)
        
        # 添加关系
        for edge_data in data.get("edges", []):
            relation = Relation.from_dict(edge_data)
            kg.add_relation(relation)
        
        return kg
    
    def save(self, filepath: str):
        """保存知识图谱到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"知识图谱已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """从文件加载知识图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        kg = cls.from_dict(data)
        logger.info(f"知识图谱已加载: {filepath}, 节点数: {kg.num_nodes}, 边数: {kg.num_edges}")
        return kg
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        # 计算节点度数
        degrees = []
        for node_id in self._nodes:
            degree = len(self._adjacency.get(node_id, set()))
            degrees.append(degree)
        
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        
        # 按类型统计
        type_counts = {
            t.value: len(ids) for t, ids in self._type_index.items() if ids
        }
        
        # 关系类型统计
        relation_counts = defaultdict(int)
        for edge in self._edges.values():
            relation_counts[edge.relation.relation_type.value] += 1
        
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "avg_degree": avg_degree,
            "entity_type_counts": type_counts,
            "relation_type_counts": dict(relation_counts),
        }
    
    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={self.num_nodes}, edges={self.num_edges})"

