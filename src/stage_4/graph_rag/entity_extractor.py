"""
实体抽取器

使用 LLM 从文本中抽取命名实体。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.stage_4.config import Stage4Config, get_stage4_config


class EntityType(str, Enum):
    """实体类型枚举"""
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    EVENT = "Event"
    CONCEPT = "Concept"
    PRODUCT = "Product"
    TIME = "Time"
    OTHER = "Other"


@dataclass
class Entity:
    """
    实体数据类
    
    Attributes:
        name: 实体名称
        type: 实体类型
        description: 实体描述
        aliases: 实体别名列表
        source_text: 来源文本片段
        source_doc: 来源文档ID
        confidence: 置信度分数
        metadata: 额外元数据
    """
    name: str
    type: EntityType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    source_text: str = ""
    source_doc: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """生成实体唯一ID"""
        content = f"{self.name}:{self.type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "aliases": self.aliases,
            "source_text": self.source_text,
            "source_doc": self.source_doc,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """从字典创建实体"""
        return cls(
            name=data["name"],
            type=EntityType(data["type"]),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            source_text=data.get("source_text", ""),
            source_doc=data.get("source_doc", ""),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


# Pydantic 模型用于结构化输出
class ExtractedEntity(BaseModel):
    """LLM 输出的实体结构"""
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型")
    description: str = Field(description="实体的简短描述")
    aliases: List[str] = Field(default_factory=list, description="实体的其他名称或别名")


class EntityExtractionResult(BaseModel):
    """实体抽取结果"""
    entities: List[ExtractedEntity] = Field(description="抽取的实体列表")


# 实体抽取提示词
ENTITY_EXTRACTION_PROMPT = """你是一个专业的实体抽取专家。请从给定的文本中抽取所有重要的命名实体。

支持的实体类型：
- Person: 人物（如：张三、马云、任正非）
- Organization: 组织/公司（如：华为、阿里巴巴、中国银行）
- Location: 地点（如：北京、深圳、硅谷）
- Event: 事件（如：世界杯、双十一、年度大会）
- Concept: 概念/术语（如：人工智能、区块链、量子计算）
- Product: 产品（如：iPhone、ChatGPT、微信）
- Time: 时间（如：2024年、第三季度、去年）

抽取要求：
1. 只抽取文本中明确提到的实体
2. 为每个实体提供简短描述
3. 如果实体有别名，请一并列出
4. 忽略过于通用的词汇

文本内容：
{text}

请严格按以下 JSON 格式输出（不要添加任何其他内容）：
```json
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "description": "简短描述", "aliases": ["别名1", "别名2"]}},
        {{"name": "实体名称2", "type": "实体类型", "description": "简短描述", "aliases": []}}
    ]
}}
```"""


class EntityExtractor:
    """
    实体抽取器
    
    使用 LLM 从文本中抽取命名实体。
    """
    
    def __init__(
        self,
        config: Optional[Stage4Config] = None,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        初始化实体抽取器
        
        Args:
            config: Stage4 配置
            llm: LLM 实例（可选，不提供则自动创建）
        """
        self.config = config or get_stage4_config()
        self._llm = llm or self._create_llm()
        self._prompt = ChatPromptTemplate.from_template(ENTITY_EXTRACTION_PROMPT)
        
        # 实体类型映射
        self._type_mapping = {
            "Person": EntityType.PERSON,
            "Organization": EntityType.ORGANIZATION,
            "Location": EntityType.LOCATION,
            "Event": EntityType.EVENT,
            "Concept": EntityType.CONCEPT,
            "Product": EntityType.PRODUCT,
            "Time": EntityType.TIME,
        }
        
        logger.info("🔍 实体抽取器初始化完成")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0,  # 使用确定性输出
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def _parse_response(self, response: str) -> List[ExtractedEntity]:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON
            content = response.strip()
            
            logger.debug(f"LLM 原始响应: {content[:500]}...")
            
            # 处理可能的 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()
                    # 移除可能的语言标识（如 json）
                    if content.startswith("json"):
                        content = content[4:].strip()
            
            logger.debug(f"处理后的 JSON: {content[:300]}...")
            
            data = json.loads(content)
            
            # 处理不同的返回格式
            if isinstance(data, list):
                entities_data = data
            elif isinstance(data, dict) and "entities" in data:
                entities_data = data["entities"]
            else:
                entities_data = [data]
            
            entities = []
            for item in entities_data:
                try:
                    entity = ExtractedEntity(
                        name=item.get("name", ""),
                        type=item.get("type", "Other"),
                        description=item.get("description", ""),
                        aliases=item.get("aliases", []),
                    )
                    if entity.name:  # 忽略空名称
                        entities.append(entity)
                        logger.debug(f"解析到实体: {entity.name} ({entity.type})")
                except Exception as e:
                    logger.warning(f"解析实体失败: {item}, 错误: {e}")
            
            logger.info(f"📦 成功解析 {len(entities)} 个实体")
            return entities
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 解析失败: {e}")
            logger.error(f"❌ 原始内容: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"❌ 解析响应时发生异常: {e}")
            logger.error(f"❌ 原始内容: {response[:500]}...")
            return []
    
    def extract(
        self,
        text: str,
        source_doc: str = "",
        max_entities: Optional[int] = None,
    ) -> List[Entity]:
        """
        从文本中抽取实体
        
        Args:
            text: 输入文本
            source_doc: 来源文档ID
            max_entities: 最大实体数量（默认使用配置值）
            
        Returns:
            List[Entity]: 抽取的实体列表
        """
        if not text.strip():
            logger.warning("输入文本为空，跳过实体抽取")
            return []
        
        max_entities = max_entities or self.config.max_entities_per_chunk
        
        logger.info(f"🔍 开始实体抽取，文本长度: {len(text)} 字符")
        
        try:
            # 构建提示
            prompt = self._prompt.format(text=text)
            
            # 调用 LLM
            logger.debug(f"调用 LLM 进行实体抽取...")
            response = self._llm.invoke(prompt)
            
            logger.debug(f"LLM 响应长度: {len(response.content)} 字符")
            
            # 解析响应
            extracted = self._parse_response(response.content)
            
            if not extracted:
                logger.warning(f"⚠️ 未从文本中抽取到任何实体")
                logger.warning(f"⚠️ 文本预览: {text[:200]}...")
            
            # 转换为 Entity 对象
            entities = []
            for item in extracted[:max_entities]:
                entity_type = self._type_mapping.get(item.type, EntityType.OTHER)
                entity = Entity(
                    name=item.name,
                    type=entity_type,
                    description=item.description,
                    aliases=item.aliases,
                    source_text=text[:200],  # 保留部分源文本
                    source_doc=source_doc,
                )
                entities.append(entity)
            
            logger.info(f"✅ 实体抽取完成: {len(entities)} 个实体")
            for e in entities:
                logger.debug(f"   - {e.name} ({e.type.value})")
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ 实体抽取失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def extract_batch(
        self,
        texts: List[str],
        source_docs: Optional[List[str]] = None,
    ) -> List[List[Entity]]:
        """
        批量抽取实体
        
        Args:
            texts: 文本列表
            source_docs: 来源文档ID列表
            
        Returns:
            List[List[Entity]]: 每个文本对应的实体列表
        """
        source_docs = source_docs or [""] * len(texts)
        results = []
        
        for i, (text, source_doc) in enumerate(zip(texts, source_docs)):
            logger.info(f"抽取进度: {i+1}/{len(texts)}")
            entities = self.extract(text, source_doc)
            results.append(entities)
        
        total = sum(len(e) for e in results)
        logger.info(f"✅ 批量抽取完成: {len(texts)} 个文本, 共 {total} 个实体")
        
        return results
    
    def merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        合并相同或相似的实体
        
        Args:
            entities: 实体列表
            
        Returns:
            List[Entity]: 合并后的实体列表
        """
        if not entities:
            return []
        
        # 按名称分组
        entity_groups: Dict[str, List[Entity]] = {}
        
        for entity in entities:
            # 标准化名称（小写，去空格）
            key = entity.name.lower().strip()
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # 合并每组实体
        merged = []
        for group in entity_groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 合并：取最常见的类型，合并描述和别名
                main = group[0]
                all_aliases = set(main.aliases)
                all_descriptions = [main.description]
                
                for e in group[1:]:
                    all_aliases.update(e.aliases)
                    all_aliases.add(e.name)  # 将其他名称作为别名
                    if e.description:
                        all_descriptions.append(e.description)
                
                # 去除主名称本身
                all_aliases.discard(main.name)
                all_aliases.discard(main.name.lower())
                
                merged_entity = Entity(
                    name=main.name,
                    type=main.type,
                    description=" | ".join(set(all_descriptions)),
                    aliases=list(all_aliases),
                    source_doc=main.source_doc,
                    confidence=sum(e.confidence for e in group) / len(group),
                )
                merged.append(merged_entity)
        
        logger.info(f"合并实体: {len(entities)} -> {len(merged)}")
        return merged

