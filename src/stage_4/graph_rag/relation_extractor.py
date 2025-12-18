"""
关系抽取器

使用 LLM 从文本中抽取实体间的关系。
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.stage_4.config import Stage4Config, get_stage4_config
from .entity_extractor import Entity


class RelationType(str, Enum):
    """关系类型枚举"""
    BELONGS_TO = "belongs_to"        # 隶属关系
    COOPERATES_WITH = "cooperates_with"  # 合作关系
    COMPETES_WITH = "competes_with"  # 竞争关系
    INVESTS_IN = "invests_in"        # 投资关系
    MANAGES = "manages"              # 管理关系
    LOCATED_IN = "located_in"        # 位于
    PARTICIPATES_IN = "participates_in"  # 参与
    PRODUCES = "produces"            # 生产
    FOUNDED = "founded"              # 创立
    WORKS_FOR = "works_for"          # 就职于
    RELATED_TO = "related_to"        # 相关（通用关系）
    OWNS = "owns"                    # 拥有
    SUBSIDIARY_OF = "subsidiary_of"  # 子公司
    ACQUIRED = "acquired"            # 收购
    PARTNERED_WITH = "partnered_with"  # 合作伙伴


@dataclass
class Relation:
    """
    关系数据类
    
    Attributes:
        source: 源实体名称
        target: 目标实体名称
        relation_type: 关系类型
        description: 关系描述
        source_text: 来源文本
        confidence: 置信度
        bidirectional: 是否双向关系
        metadata: 额外元数据
    """
    source: str
    target: str
    relation_type: RelationType
    description: str = ""
    source_text: str = ""
    confidence: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """生成关系唯一ID"""
        content = f"{self.source}:{self.relation_type.value}:{self.target}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "description": self.description,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """从字典创建关系"""
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=RelationType(data["relation_type"]),
            description=data.get("description", ""),
            source_text=data.get("source_text", ""),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
            metadata=data.get("metadata", {}),
        )


# Pydantic 模型用于结构化输出
class ExtractedRelation(BaseModel):
    """LLM 输出的关系结构"""
    source: str = Field(description="源实体名称")
    target: str = Field(description="目标实体名称")
    relation_type: str = Field(description="关系类型")
    description: str = Field(description="关系的描述")


class RelationExtractionResult(BaseModel):
    """关系抽取结果"""
    relations: List[ExtractedRelation] = Field(description="抽取的关系列表")


# 关系抽取提示词
RELATION_EXTRACTION_PROMPT = """你是一个专业的关系抽取专家。请从给定的文本中抽取实体之间的关系。

已识别的实体：
{entities}

支持的关系类型：
- belongs_to: 隶属关系（如：部门属于公司）
- cooperates_with: 合作关系（如：两家公司合作）
- competes_with: 竞争关系（如：两家公司竞争）
- invests_in: 投资关系（如：A 投资 B）
- manages: 管理关系（如：某人管理某部门）
- located_in: 位于（如：公司位于某城市）
- participates_in: 参与（如：某人参与某事件）
- produces: 生产（如：公司生产某产品）
- founded: 创立（如：某人创立某公司）
- works_for: 就职于（如：某人在某公司工作）
- owns: 拥有（如：某人拥有某公司）
- subsidiary_of: 子公司（如：A 是 B 的子公司）
- acquired: 收购（如：A 收购了 B）
- partnered_with: 合作伙伴
- related_to: 通用相关关系

抽取要求：
1. 只抽取实体列表中的实体之间的关系
2. 关系必须在文本中有明确的依据
3. 为每个关系提供简短描述
4. 关系方向很重要：source -> target

文本内容：
{text}

请严格按以下 JSON 格式输出（不要添加任何其他内容）：
```json
{{
    "relations": [
        {{"source": "实体A", "target": "实体B", "relation_type": "关系类型", "description": "关系描述"}},
        {{"source": "实体C", "target": "实体D", "relation_type": "关系类型", "description": "关系描述"}}
    ]
}}
```"""


class RelationExtractor:
    """
    关系抽取器
    
    使用 LLM 从文本中抽取实体间的关系。
    """
    
    def __init__(
        self,
        config: Optional[Stage4Config] = None,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        初始化关系抽取器
        
        Args:
            config: Stage4 配置
            llm: LLM 实例
        """
        self.config = config or get_stage4_config()
        self._llm = llm or self._create_llm()
        self._prompt = ChatPromptTemplate.from_template(RELATION_EXTRACTION_PROMPT)
        
        # 关系类型映射
        self._type_mapping = {
            "belongs_to": RelationType.BELONGS_TO,
            "cooperates_with": RelationType.COOPERATES_WITH,
            "competes_with": RelationType.COMPETES_WITH,
            "invests_in": RelationType.INVESTS_IN,
            "manages": RelationType.MANAGES,
            "located_in": RelationType.LOCATED_IN,
            "participates_in": RelationType.PARTICIPATES_IN,
            "produces": RelationType.PRODUCES,
            "founded": RelationType.FOUNDED,
            "works_for": RelationType.WORKS_FOR,
            "owns": RelationType.OWNS,
            "subsidiary_of": RelationType.SUBSIDIARY_OF,
            "acquired": RelationType.ACQUIRED,
            "partnered_with": RelationType.PARTNERED_WITH,
            "related_to": RelationType.RELATED_TO,
        }
        
        logger.info("🔗 关系抽取器初始化完成")
    
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
    
    def _format_entities(self, entities: List[Entity]) -> str:
        """格式化实体列表"""
        lines = []
        for e in entities:
            lines.append(f"- {e.name} ({e.type.value}): {e.description}")
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> List[ExtractedRelation]:
        """解析 LLM 响应"""
        try:
            content = response.strip()
            
            logger.debug(f"关系抽取 LLM 原始响应: {content[:500]}...")
            
            # 处理 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
            
            logger.debug(f"处理后的 JSON: {content[:300]}...")
            
            data = json.loads(content)
            
            # 处理不同的返回格式
            if isinstance(data, list):
                relations_data = data
            elif isinstance(data, dict) and "relations" in data:
                relations_data = data["relations"]
            else:
                relations_data = [data]
            
            relations = []
            for item in relations_data:
                try:
                    relation = ExtractedRelation(
                        source=item.get("source", ""),
                        target=item.get("target", ""),
                        relation_type=item.get("relation_type", "related_to"),
                        description=item.get("description", ""),
                    )
                    if relation.source and relation.target:
                        relations.append(relation)
                        logger.debug(f"解析到关系: {relation.source} -> {relation.target}")
                except Exception as e:
                    logger.warning(f"解析关系失败: {item}, 错误: {e}")
            
            logger.info(f"📦 成功解析 {len(relations)} 条关系")
            return relations
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 解析失败: {e}")
            logger.error(f"❌ 原始内容: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"❌ 解析响应时发生异常: {e}")
            return []
    
    def extract(
        self,
        text: str,
        entities: List[Entity],
        max_relations: Optional[int] = None,
    ) -> List[Relation]:
        """
        从文本中抽取关系
        
        Args:
            text: 输入文本
            entities: 已识别的实体列表
            max_relations: 最大关系数量
            
        Returns:
            List[Relation]: 抽取的关系列表
        """
        if not text.strip() or not entities:
            return []
        
        max_relations = max_relations or self.config.max_relations_per_chunk
        
        try:
            # 构建实体列表字符串
            entities_str = self._format_entities(entities)
            
            # 构建提示
            prompt = self._prompt.format(text=text, entities=entities_str)
            
            # 调用 LLM
            response = self._llm.invoke(prompt)
            
            # 解析响应
            extracted = self._parse_response(response.content)
            
            # 创建实体名称集合（用于验证）
            entity_names = {e.name.lower() for e in entities}
            entity_names.update(alias.lower() for e in entities for alias in e.aliases)
            
            # 转换为 Relation 对象
            relations = []
            for item in extracted[:max_relations]:
                # 验证源和目标实体存在
                source_valid = item.source.lower() in entity_names
                target_valid = item.target.lower() in entity_names
                
                if not source_valid or not target_valid:
                    logger.debug(f"跳过无效关系: {item.source} -> {item.target}")
                    continue
                
                # 获取关系类型
                relation_type = self._type_mapping.get(
                    item.relation_type.lower(), 
                    RelationType.RELATED_TO
                )
                
                # 计算置信度（基于关系类型）
                confidence = 0.9 if relation_type != RelationType.RELATED_TO else 0.7
                
                relation = Relation(
                    source=item.source,
                    target=item.target,
                    relation_type=relation_type,
                    description=item.description,
                    source_text=text[:200],
                    confidence=confidence,
                )
                relations.append(relation)
            
            logger.debug(f"从文本中抽取了 {len(relations)} 个关系")
            return relations
            
        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []
    
    def extract_batch(
        self,
        texts: List[str],
        entities_list: List[List[Entity]],
    ) -> List[List[Relation]]:
        """
        批量抽取关系
        
        Args:
            texts: 文本列表
            entities_list: 每个文本对应的实体列表
            
        Returns:
            List[List[Relation]]: 每个文本对应的关系列表
        """
        results = []
        
        for i, (text, entities) in enumerate(zip(texts, entities_list)):
            logger.info(f"关系抽取进度: {i+1}/{len(texts)}")
            relations = self.extract(text, entities)
            results.append(relations)
        
        total = sum(len(r) for r in results)
        logger.info(f"✅ 批量关系抽取完成: {len(texts)} 个文本, 共 {total} 个关系")
        
        return results
    
    def merge_relations(self, relations: List[Relation]) -> List[Relation]:
        """
        合并重复的关系
        
        Args:
            relations: 关系列表
            
        Returns:
            List[Relation]: 合并后的关系列表
        """
        if not relations:
            return []
        
        # 按 (source, target, type) 分组
        relation_groups: Dict[Tuple[str, str, str], List[Relation]] = {}
        
        for rel in relations:
            key = (rel.source.lower(), rel.target.lower(), rel.relation_type.value)
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(rel)
        
        # 合并每组关系
        merged = []
        for group in relation_groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 合并：取最高置信度，合并描述
                main = max(group, key=lambda r: r.confidence)
                all_descriptions = set(r.description for r in group if r.description)
                
                merged_relation = Relation(
                    source=main.source,
                    target=main.target,
                    relation_type=main.relation_type,
                    description=" | ".join(all_descriptions) if all_descriptions else main.description,
                    confidence=max(r.confidence for r in group),
                    bidirectional=main.bidirectional,
                )
                merged.append(merged_relation)
        
        logger.info(f"合并关系: {len(relations)} -> {len(merged)}")
        return merged

