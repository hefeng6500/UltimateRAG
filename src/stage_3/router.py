"""
智能路由模块

根据用户问题类型，智能分发到不同的处理器：
- KNOWLEDGE_BASE: 知识库检索（RAG）
- WEB_SEARCH: Web 实时搜索
- CALCULATOR: 数学计算
- CODE_EXECUTION: 代码执行
- DIRECT_ANSWER: 直接 LLM 回答（闲聊）
"""

from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import Stage3Config, get_stage3_config


class RouteType(str, Enum):
    """路由类型枚举"""
    KNOWLEDGE_BASE = "knowledge_base"  # 知识库检索
    WEB_SEARCH = "web_search"          # Web 搜索
    CALCULATOR = "calculator"          # 计算器
    CODE_EXECUTION = "code_execution"  # 代码执行
    DIRECT_ANSWER = "direct_answer"    # 直接回答


class RouteDecision(BaseModel):
    """路由决策的结构化输出"""
    route_type: RouteType = Field(
        description="选择的路由类型"
    )
    confidence: float = Field(
        description="决策置信度，0-1 之间",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="选择该路由的原因"
    )
    transformed_query: str = Field(
        description="针对该路由优化后的查询"
    )


# 路由决策提示词
ROUTER_PROMPT = """你是一个智能问题路由器。分析用户的问题，决定应该如何处理。

可选的路由类型：
1. knowledge_base - 需要从知识库/文档中检索信息的问题（如：文档内容、技术细节、历史记录等）
2. web_search - 需要实时互联网信息的问题（如：最新新闻、天气、股价、当前事件等）
3. calculator - 需要数学计算的问题（如：加减乘除、百分比、统计等）
4. code_execution - 需要执行代码来解答的问题（如：数据分析、算法演示等）
5. direct_answer - 不需要外部信息的通用问题（如：闲聊、常识问答、概念解释等）

判断规则：
- 如果问题涉及特定文档、内部知识、专业术语 → knowledge_base
- 如果问题需要最新信息、实时数据 → web_search
- 如果问题是数学计算、统计问题 → calculator
- 如果问题需要运行代码才能解答 → code_execution
- 如果是日常闲聊或简单的常识问题 → direct_answer

用户问题：{question}

请分析并给出你的路由决策。"""


class QueryRouter:
    """
    智能查询路由器
    
    使用 LLM 分析问题类型，决定最佳处理路径。
    支持多路由策略和置信度评估。
    """
    
    def __init__(self, config: Optional[Stage3Config] = None):
        """
        初始化路由器
        
        Args:
            config: Stage3 配置对象
        """
        self.config = config or get_stage3_config()
        self._llm = self._create_llm()
        self._structured_llm = self._llm.with_structured_output(RouteDecision)
        
        logger.info("🚦 智能路由器初始化完成")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.1,  # 低温度以获得更一致的分类结果
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def route(self, question: str) -> RouteDecision:
        """
        对问题进行路由决策
        
        Args:
            question: 用户问题
            
        Returns:
            RouteDecision: 路由决策结果
        """
        prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
        
        try:
            decision: RouteDecision = self._structured_llm.invoke(
                prompt.format(question=question)
            )
            
            logger.info(
                f"🚦 路由决策: {decision.route_type.value} "
                f"(置信度: {decision.confidence:.2f})"
            )
            logger.debug(f"   原因: {decision.reasoning}")
            
            return decision
            
        except Exception as e:
            logger.warning(f"⚠️ 路由决策失败，回退到知识库检索: {e}")
            return RouteDecision(
                route_type=RouteType.KNOWLEDGE_BASE,
                confidence=0.5,
                reasoning="路由决策失败，使用默认知识库检索",
                transformed_query=question
            )
    
    def should_use_fallback(self, decision: RouteDecision) -> bool:
        """
        判断是否需要使用回退策略
        
        当置信度低于阈值时，可能需要尝试多种策略。
        
        Args:
            decision: 路由决策
            
        Returns:
            bool: 是否需要回退
        """
        return decision.confidence < self.config.router_confidence_threshold
    
    def get_fallback_routes(self, primary: RouteType) -> list[RouteType]:
        """
        获取回退路由列表
        
        Args:
            primary: 主路由类型
            
        Returns:
            list[RouteType]: 回退路由列表
        """
        fallback_map = {
            RouteType.KNOWLEDGE_BASE: [RouteType.WEB_SEARCH, RouteType.DIRECT_ANSWER],
            RouteType.WEB_SEARCH: [RouteType.KNOWLEDGE_BASE, RouteType.DIRECT_ANSWER],
            RouteType.CALCULATOR: [RouteType.DIRECT_ANSWER],
            RouteType.CODE_EXECUTION: [RouteType.DIRECT_ANSWER],
            RouteType.DIRECT_ANSWER: [RouteType.KNOWLEDGE_BASE],
        }
        return fallback_map.get(primary, [RouteType.DIRECT_ANSWER])


class KeywordRouter:
    """
    基于关键词的快速路由器（备选方案）
    
    不依赖 LLM，使用规则匹配进行快速路由。
    """
    
    # 关键词规则
    ROUTE_KEYWORDS = {
        RouteType.CALCULATOR: [
            "计算", "算一下", "加", "减", "乘", "除", "等于", 
            "百分比", "%", "平均", "总和", "最大", "最小",
            "calculate", "sum", "average", "percent"
        ],
        RouteType.CODE_EXECUTION: [
            "运行代码", "执行", "python", "代码", "程序",
            "run code", "execute", "script"
        ],
        RouteType.WEB_SEARCH: [
            "最新", "今天", "现在", "实时", "新闻", "天气",
            "股价", "汇率", "latest", "current", "today", "news"
        ],
        RouteType.DIRECT_ANSWER: [
            "你好", "hello", "hi", "谢谢", "再见", "是什么意思",
            "什么是", "解释一下", "tell me about"
        ],
    }
    
    def route(self, question: str) -> Tuple[RouteType, float]:
        """
        基于关键词的快速路由
        
        Args:
            question: 用户问题
            
        Returns:
            Tuple[RouteType, float]: (路由类型, 置信度)
        """
        question_lower = question.lower()
        
        # 检查各类关键词
        for route_type, keywords in self.ROUTE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    return route_type, 0.8
        
        # 默认走知识库检索
        return RouteType.KNOWLEDGE_BASE, 0.6

