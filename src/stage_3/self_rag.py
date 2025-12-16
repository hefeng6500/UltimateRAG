"""
Self-RAG (自反思 RAG) 模块

实现答案质量自评估和迭代优化：
1. 检索相关性评估 - 判断检索到的文档是否相关
2. 答案质量评估 - 判断生成的答案是否完整准确
3. 迭代检索 - 质量不足时重新检索
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import Stage3Config, get_stage3_config


class RelevanceGrade(str, Enum):
    """检索相关性等级"""
    HIGHLY_RELEVANT = "highly_relevant"  # 高度相关
    PARTIALLY_RELEVANT = "partially_relevant"  # 部分相关
    NOT_RELEVANT = "not_relevant"  # 不相关


class QualityGrade(str, Enum):
    """答案质量等级"""
    EXCELLENT = "excellent"  # 优秀
    ACCEPTABLE = "acceptable"  # 可接受
    NEEDS_IMPROVEMENT = "needs_improvement"  # 需要改进
    INSUFFICIENT = "insufficient"  # 不足


class RelevanceEvaluation(BaseModel):
    """检索相关性评估结果"""
    grade: RelevanceGrade = Field(description="相关性等级")
    score: float = Field(description="相关性分数 0-1", ge=0.0, le=1.0)
    reasoning: str = Field(description="评估原因")


class QualityEvaluation(BaseModel):
    """答案质量评估结果"""
    grade: QualityGrade = Field(description="质量等级")
    score: float = Field(description="质量分数 0-1", ge=0.0, le=1.0)
    is_complete: bool = Field(description="答案是否完整")
    is_accurate: bool = Field(description="答案是否准确")
    missing_info: str = Field(description="缺失的信息描述")
    suggestions: str = Field(description="改进建议")


@dataclass
class SelfRAGResult:
    """Self-RAG 处理结果"""
    answer: str
    documents: List[Document]
    iterations: int
    relevance_scores: List[float]
    quality_score: float
    quality_grade: QualityGrade
    reasoning_chain: List[str]


# 相关性评估提示词
RELEVANCE_PROMPT = """你是一个文档相关性评估专家。
评估给定文档与用户问题的相关程度。

用户问题：{question}

待评估文档：
{document}

评估标准：
- highly_relevant: 文档直接回答了问题的核心内容
- partially_relevant: 文档包含一些相关信息，但不完整
- not_relevant: 文档与问题基本无关

请给出你的评估。"""

# 质量评估提示词
QUALITY_PROMPT = """你是一个答案质量评估专家。
评估给定答案对用户问题的回答质量。

用户问题：{question}

参考文档：
{context}

生成的答案：
{answer}

评估标准：
- excellent: 答案完整、准确、有条理，完全解答了问题
- acceptable: 答案基本正确，但可能遗漏一些细节
- needs_improvement: 答案部分正确，但有明显遗漏或不准确
- insufficient: 答案不能有效回答问题

请给出详细的质量评估。"""

# 查询优化提示词
QUERY_REFINEMENT_PROMPT = """基于当前的检索结果不够理想，请帮我优化搜索查询。

原始问题：{question}

当前检索到的信息：
{current_info}

缺失的信息：
{missing_info}

请生成一个更精确的搜索查询，用于找到缺失的信息。只输出新的查询语句，不要其他内容。"""


class SelfRAG:
    """
    自反思 RAG
    
    实现 RAG 答案的自我评估和迭代优化。
    核心流程：检索 -> 评估相关性 -> 生成答案 -> 评估质量 -> 必要时迭代
    """
    
    def __init__(self, config: Optional[Stage3Config] = None):
        """
        初始化 Self-RAG
        
        Args:
            config: Stage3 配置
        """
        self.config = config or get_stage3_config()
        self._llm = self._create_llm()
        self._relevance_evaluator = self._llm.with_structured_output(RelevanceEvaluation)
        self._quality_evaluator = self._llm.with_structured_output(QualityEvaluation)
        
        logger.info("🔄 Self-RAG 初始化完成")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.1,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def evaluate_relevance(
        self, 
        question: str, 
        document: Document
    ) -> RelevanceEvaluation:
        """
        评估单个文档的相关性
        
        Args:
            question: 用户问题
            document: 待评估文档
            
        Returns:
            RelevanceEvaluation: 相关性评估结果
        """
        prompt = ChatPromptTemplate.from_template(RELEVANCE_PROMPT)
        
        try:
            result = self._relevance_evaluator.invoke(
                prompt.format(
                    question=question,
                    document=document.page_content[:1000]
                )
            )
            return result
        except Exception as e:
            logger.warning(f"⚠️ 相关性评估失败: {e}")
            return RelevanceEvaluation(
                grade=RelevanceGrade.PARTIALLY_RELEVANT,
                score=0.5,
                reasoning="评估失败，使用默认值"
            )
    
    def filter_relevant_documents(
        self,
        question: str,
        documents: List[Document],
        threshold: float = None
    ) -> Tuple[List[Document], List[float]]:
        """
        过滤相关文档
        
        Args:
            question: 用户问题
            documents: 文档列表
            threshold: 相关性阈值
            
        Returns:
            Tuple[List[Document], List[float]]: (过滤后的文档, 相关性分数)
        """
        threshold = threshold or self.config.self_rag_relevance_threshold
        
        filtered_docs = []
        scores = []
        
        for doc in documents:
            evaluation = self.evaluate_relevance(question, doc)
            
            if evaluation.score >= threshold:
                filtered_docs.append(doc)
                scores.append(evaluation.score)
                logger.debug(
                    f"  ✅ 保留文档 (分数: {evaluation.score:.2f}): "
                    f"{doc.page_content[:50]}..."
                )
            else:
                logger.debug(
                    f"  ❌ 过滤文档 (分数: {evaluation.score:.2f}): "
                    f"{doc.page_content[:50]}..."
                )
        
        logger.info(
            f"🔍 相关性过滤: {len(documents)} -> {len(filtered_docs)} 个文档"
        )
        
        return filtered_docs, scores
    
    def evaluate_answer_quality(
        self,
        question: str,
        context: str,
        answer: str
    ) -> QualityEvaluation:
        """
        评估答案质量
        
        Args:
            question: 用户问题
            context: 参考上下文
            answer: 生成的答案
            
        Returns:
            QualityEvaluation: 质量评估结果
        """
        prompt = ChatPromptTemplate.from_template(QUALITY_PROMPT)
        
        try:
            result = self._quality_evaluator.invoke(
                prompt.format(
                    question=question,
                    context=context[:2000],
                    answer=answer
                )
            )
            
            logger.info(
                f"📊 答案质量评估: {result.grade.value} "
                f"(分数: {result.score:.2f})"
            )
            
            return result
        except Exception as e:
            logger.warning(f"⚠️ 质量评估失败: {e}")
            return QualityEvaluation(
                grade=QualityGrade.ACCEPTABLE,
                score=0.6,
                is_complete=True,
                is_accurate=True,
                missing_info="",
                suggestions=""
            )
    
    def refine_query(
        self,
        question: str,
        current_info: str,
        missing_info: str
    ) -> str:
        """
        优化搜索查询
        
        Args:
            question: 原始问题
            current_info: 当前检索到的信息
            missing_info: 缺失的信息描述
            
        Returns:
            str: 优化后的查询
        """
        prompt = ChatPromptTemplate.from_template(QUERY_REFINEMENT_PROMPT)
        
        try:
            response = self._llm.invoke(
                prompt.format(
                    question=question,
                    current_info=current_info[:500],
                    missing_info=missing_info
                )
            )
            refined_query = response.content.strip()
            logger.info(f"🔄 查询优化: {question[:30]}... -> {refined_query[:30]}...")
            return refined_query
        except Exception as e:
            logger.warning(f"⚠️ 查询优化失败: {e}")
            return question
    
    def should_iterate(self, quality_eval: QualityEvaluation) -> bool:
        """
        判断是否需要迭代
        
        Args:
            quality_eval: 质量评估结果
            
        Returns:
            bool: 是否需要迭代
        """
        # 质量不足且有缺失信息时需要迭代
        return (
            quality_eval.score < self.config.self_rag_quality_threshold
            and quality_eval.missing_info
            and quality_eval.grade in [
                QualityGrade.NEEDS_IMPROVEMENT,
                QualityGrade.INSUFFICIENT
            ]
        )
    
    def generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 参考上下文
            
        Returns:
            str: 生成的答案
        """
        prompt = ChatPromptTemplate.from_template(
            """基于以下参考文档回答用户问题。
如果文档中没有足够信息，请诚实说明。

参考文档：
{context}

用户问题：{question}

请用中文回答："""
        )
        
        response = self._llm.invoke(
            prompt.format(context=context, question=question)
        )
        return response.content

