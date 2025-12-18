"""
Stage 4 单元测试

测试 GraphRAG 和微调模块。
"""

import pytest
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

from src.stage_4.config import Stage4Config, get_stage4_config
from src.stage_4.graph_rag.entity_extractor import (
    Entity,
    EntityType,
    EntityExtractor,
)
from src.stage_4.graph_rag.relation_extractor import (
    Relation,
    RelationType,
    RelationExtractor,
)
from src.stage_4.graph_rag.knowledge_graph import KnowledgeGraph, SubGraph
from src.stage_4.graph_rag.graph_store import MemoryGraphStore
from src.stage_4.fine_tuning.llm_finetuner import QAPair, FineTuneDataFormat


class TestEntity:
    """实体测试"""
    
    def test_entity_creation(self):
        """测试实体创建"""
        entity = Entity(
            name="华为",
            type=EntityType.ORGANIZATION,
            description="中国科技公司",
        )
        
        assert entity.name == "华为"
        assert entity.type == EntityType.ORGANIZATION
        assert entity.id is not None
    
    def test_entity_to_dict(self):
        """测试实体序列化"""
        entity = Entity(
            name="任正非",
            type=EntityType.PERSON,
            description="华为创始人",
            aliases=["任老"],
        )
        
        data = entity.to_dict()
        
        assert data["name"] == "任正非"
        assert data["type"] == "Person"
        assert "任老" in data["aliases"]
    
    def test_entity_from_dict(self):
        """测试实体反序列化"""
        data = {
            "name": "深圳",
            "type": "Location",
            "description": "中国城市",
        }
        
        entity = Entity.from_dict(data)
        
        assert entity.name == "深圳"
        assert entity.type == EntityType.LOCATION


class TestRelation:
    """关系测试"""
    
    def test_relation_creation(self):
        """测试关系创建"""
        relation = Relation(
            source="任正非",
            target="华为",
            relation_type=RelationType.FOUNDED,
            description="创立了华为公司",
        )
        
        assert relation.source == "任正非"
        assert relation.target == "华为"
        assert relation.id is not None
    
    def test_relation_to_dict(self):
        """测试关系序列化"""
        relation = Relation(
            source="华为",
            target="深圳",
            relation_type=RelationType.LOCATED_IN,
        )
        
        data = relation.to_dict()
        
        assert data["source"] == "华为"
        assert data["target"] == "深圳"
        assert data["relation_type"] == "located_in"


class TestKnowledgeGraph:
    """知识图谱测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.kg = KnowledgeGraph()
    
    def test_add_entity(self):
        """测试添加实体"""
        entity = Entity(
            name="华为",
            type=EntityType.ORGANIZATION,
        )
        
        entity_id = self.kg.add_entity(entity)
        
        assert entity_id is not None
        assert self.kg.num_nodes == 1
    
    def test_add_relation(self):
        """测试添加关系"""
        # 先添加实体
        e1 = Entity(name="任正非", type=EntityType.PERSON)
        e2 = Entity(name="华为", type=EntityType.ORGANIZATION)
        
        self.kg.add_entity(e1)
        self.kg.add_entity(e2)
        
        # 添加关系
        relation = Relation(
            source="任正非",
            target="华为",
            relation_type=RelationType.FOUNDED,
        )
        
        rel_id = self.kg.add_relation(relation)
        
        assert rel_id is not None
        assert self.kg.num_edges == 1
    
    def test_get_entity_by_name(self):
        """测试按名称获取实体"""
        entity = Entity(name="华为", type=EntityType.ORGANIZATION)
        self.kg.add_entity(entity)
        
        found = self.kg.get_entity_by_name("华为")
        
        assert found is not None
        assert found.name == "华为"
    
    def test_get_neighbors(self):
        """测试获取邻居"""
        # 构建简单图谱
        e1 = Entity(name="A", type=EntityType.CONCEPT)
        e2 = Entity(name="B", type=EntityType.CONCEPT)
        e3 = Entity(name="C", type=EntityType.CONCEPT)
        
        self.kg.add_entity(e1)
        self.kg.add_entity(e2)
        self.kg.add_entity(e3)
        
        self.kg.add_relation(Relation(
            source="A", target="B", relation_type=RelationType.RELATED_TO
        ))
        self.kg.add_relation(Relation(
            source="B", target="C", relation_type=RelationType.RELATED_TO
        ))
        
        # 获取 A 的一跳邻居
        subgraph = self.kg.get_neighbors("A", hops=1)
        
        assert len(subgraph.nodes) == 2  # A 和 B
        assert len(subgraph.edges) == 1
    
    def test_find_path(self):
        """测试路径查找"""
        # 构建图谱
        for name in ["A", "B", "C"]:
            self.kg.add_entity(Entity(name=name, type=EntityType.CONCEPT))
        
        self.kg.add_relation(Relation(
            source="A", target="B", relation_type=RelationType.RELATED_TO
        ))
        self.kg.add_relation(Relation(
            source="B", target="C", relation_type=RelationType.RELATED_TO
        ))
        
        # 查找路径
        path = self.kg.find_path("A", "C")
        
        assert path is not None
        assert len(path) == 3  # A -> B -> C
    
    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        # 创建图谱
        self.kg.add_entity(Entity(name="测试", type=EntityType.CONCEPT))
        
        # 保存
        filepath = str(tmp_path / "test_kg.json")
        self.kg.save(filepath)
        
        # 加载
        loaded_kg = KnowledgeGraph.load(filepath)
        
        assert loaded_kg.num_nodes == 1
        assert loaded_kg.get_entity_by_name("测试") is not None


class TestGraphStore:
    """图存储测试"""
    
    def test_memory_store(self, tmp_path):
        """测试内存存储"""
        store = MemoryGraphStore(persist_dir=str(tmp_path))
        
        # 创建并保存图谱
        kg = KnowledgeGraph()
        kg.add_entity(Entity(name="测试", type=EntityType.CONCEPT))
        
        store.save(kg, "test")
        
        # 检查存在性
        assert store.exists("test")
        
        # 加载
        loaded = store.load("test")
        assert loaded is not None
        assert loaded.num_nodes == 1
        
        # 列出所有图谱
        graphs = store.list_graphs()
        assert "test" in graphs
        
        # 删除
        store.delete("test")
        assert not store.exists("test")


class TestQAPair:
    """QA 对测试"""
    
    def test_qa_pair_creation(self):
        """测试 QA 对创建"""
        pair = QAPair(
            question="什么是 RAG？",
            answer="RAG 是检索增强生成...",
            difficulty="medium",
        )
        
        assert pair.question == "什么是 RAG？"
        assert pair.difficulty == "medium"
    
    def test_openai_format(self):
        """测试 OpenAI 格式转换"""
        pair = QAPair(
            question="什么是 RAG？",
            answer="RAG 是检索增强生成...",
        )
        
        data = pair.to_openai_format("你是助手")
        
        assert "messages" in data
        assert len(data["messages"]) == 3  # system, user, assistant
        assert data["messages"][0]["role"] == "system"
    
    def test_alpaca_format(self):
        """测试 Alpaca 格式转换"""
        pair = QAPair(
            question="什么是 RAG？",
            answer="RAG 是检索增强生成...",
            context="一些背景信息",
        )
        
        data = pair.to_alpaca_format()
        
        assert data["instruction"] == "什么是 RAG？"
        assert data["input"] == "一些背景信息"
        assert data["output"] == "RAG 是检索增强生成..."


class TestConfig:
    """配置测试"""
    
    def test_config_loading(self):
        """测试配置加载"""
        config = get_stage4_config()
        
        assert config is not None
        assert config.graph_store_type in ["memory", "neo4j"]
        assert len(config.entity_types) > 0
        assert len(config.relation_types) > 0
    
    def test_config_inheritance(self):
        """测试配置继承"""
        config = get_stage4_config()
        
        # 应该继承 Stage 3 配置
        assert hasattr(config, "self_rag_max_iterations")
        assert hasattr(config, "parent_chunk_size")
        
        # 应该有 Stage 4 特有配置
        assert hasattr(config, "graph_store_type")
        assert hasattr(config, "embedding_finetune_model")


# 集成测试（需要 API Key）
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("DASHSCOPE_API_KEY"),
    reason="需要 API Key"
)
class TestIntegration:
    """集成测试"""
    
    def test_entity_extraction(self):
        """测试实体抽取"""
        extractor = EntityExtractor()
        
        text = "华为公司在深圳成立，任正非是创始人。"
        entities = extractor.extract(text)
        
        assert len(entities) > 0
        # 应该抽取到华为、深圳、任正非
        names = [e.name for e in entities]
        assert any("华为" in n for n in names)
    
    def test_relation_extraction(self):
        """测试关系抽取"""
        # 先抽取实体
        entity_extractor = EntityExtractor()
        relation_extractor = RelationExtractor()
        
        text = "华为公司在深圳成立，任正非是创始人。"
        entities = entity_extractor.extract(text)
        
        if entities:
            relations = relation_extractor.extract(text, entities)
            # 应该能抽取到一些关系
            assert isinstance(relations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

