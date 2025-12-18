"""
Agentic RAG Chain

整合所有 Stage 3 组件，实现代理式 RAG：
- 智能路由
- 自反思检索
- 工具调用
- 父子索引
- 上下文压缩
"""

from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.stage_1.config import Config, get_config
from src.stage_1.vectorstore import VectorStoreManager
from src.stage_2.hybrid_retriever import HybridRetriever
from src.stage_2.query_rewriter import QueryRewriter
from src.stage_2.reranker import Reranker, SimpleReranker

from .config import Stage3Config, get_stage3_config
from .router import QueryRouter, RouteType, RouteDecision
from .self_rag import SelfRAG, QualityGrade
from .tools import WebSearchTool, CalculatorTool, CodeExecutorTool, ToolResult
from .parent_child_retriever import ParentChildRetriever
from .context_compressor import ContextCompressor, KeywordBasedCompressor


@dataclass
class AgenticRAGResult:
    """Agentic RAG 处理结果"""
    answer: str
    route_type: RouteType
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_used: Optional[str] = None
    tool_result: Optional[str] = None
    iterations: int = 1
    quality_score: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)


# RAG 生成提示词
AGENTIC_RAG_PROMPT = """你是一个智能问答助手，具备深度分析和综合能力。

请基于以下参考信息回答用户问题：

{context}

用户问题：{question}

回答要求：
1. 准确性：只基于提供的信息回答，不要编造
2. 完整性：尽可能全面地回答问题
3. 条理性：使用清晰的结构组织答案
4. 诚实性：如果信息不足，请明确说明

同时，允许你：
- 归纳总结
- 风格分析
- 观点选择
- 句子评价
- 合理推断（必须基于文档给出的内容）

请用中文回答："""

# 直接回答提示词（不需要检索）
DIRECT_ANSWER_PROMPT = """你是一个智能助手，请直接回答用户问题。

用户问题：{question}

请用中文回答："""


class AgenticRAGChain:
    """
    代理式 RAG 链
    
    整合智能路由、自反思、工具调用、父子索引、上下文压缩，
    实现更智能的问答系统。
    """
    
    def __init__(
        self,
        documents: List[Document] = None,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        config: Optional[Stage3Config] = None,
        # 功能开关
        enable_routing: bool = True,
        enable_self_rag: bool = True,
        enable_tools: bool = True,
        enable_parent_child: bool = True,
        enable_compression: bool = True,
        enable_reranking: bool = True,
        force_reindex: bool = False
    ):
        """
        初始化 Agentic RAG 链
        
        Args:
            documents: 文档列表
            vectorstore_manager: 向量存储管理器
            config: Stage3 配置
            enable_routing: 启用智能路由
            enable_self_rag: 启用自反思
            enable_tools: 启用工具调用
            enable_parent_child: 启用父子索引
            enable_compression: 启用上下文压缩
            enable_reranking: 启用重排序
        """
        self.config = config or get_stage3_config()
        self.base_config = get_config()
        
        # 功能开关
        self.enable_routing = enable_routing
        self.enable_self_rag = enable_self_rag
        self.enable_tools = enable_tools
        self.enable_parent_child = enable_parent_child
        self.enable_compression = enable_compression
        self.enable_reranking = enable_reranking
        self.force_reindex = force_reindex
        
        # 初始化 LLM
        self._llm = self._create_llm()
        
        # 初始化组件
        self._init_components(documents, vectorstore_manager)
        
        logger.info(
            f"🤖 Agentic RAG 链初始化完成:\n"
            f"   - 路由: {enable_routing}\n"
            f"   - 自反思: {enable_self_rag}\n"
            f"   - 工具: {enable_tools}\n"
            f"   - 父子索引: {enable_parent_child}\n"
            f"   - 上下文压缩: {enable_compression}\n"
            f"   - 重排序: {enable_reranking}"
        )
    
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
    
    def _init_components(
        self,
        documents: List[Document],
        vectorstore_manager: Optional[VectorStoreManager]
    ):
        """初始化各个组件"""
        
        # 1. 向量存储
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(
            self.base_config,
            collection_name="agentic_rag"
        )
        
        # 2. 路由器
        if self.enable_routing:
            self._router = QueryRouter(self.config)
        
        # 3. 自反思 RAG
        if self.enable_self_rag:
            self._self_rag = SelfRAG(self.config)
        
        # 4. 工具
        if self.enable_tools:
            self._tools = {
                "web_search": WebSearchTool(),
                "calculator": CalculatorTool(),
                "code_executor": CodeExecutorTool(timeout=self.config.code_executor_timeout)
            }
        
        # 5. 父子索引检索器
        if self.enable_parent_child and documents:
            pc_vectorstore_manager = VectorStoreManager(
                self.base_config,
                collection_name="parent_child_index"
            )
            
            # 如果强制重建索引，先清空
            if self.force_reindex:
                pc_vectorstore_manager.clear()
                logger.info("🗑️ 已清空父子索引向量库")
            
            self._parent_child_retriever = ParentChildRetriever(
                config=self.config,
                vectorstore_manager=pc_vectorstore_manager
            )
            # 只有当向量库为空或强制重建时才添加文档，避免重复添加
            if self.force_reindex or pc_vectorstore_manager.vectorstore._collection.count() == 0:
                self._parent_child_retriever.add_documents(documents)
            else:
                logger.info(
                    f"📦 使用已有父子索引: "
                    f"{pc_vectorstore_manager.vectorstore._collection.count()} 个子块"
                )
        else:
            self._parent_child_retriever = None
        
        # 6. 混合检索器
        if documents:
            self._hybrid_retriever = HybridRetriever(
                documents=documents,
                vectorstore_manager=self.vectorstore_manager,
                config=self.base_config
            )
        else:
            self._hybrid_retriever = None
        
        # 7. 查询重写器
        self._query_rewriter = QueryRewriter(self.base_config)
        
        # 8. 重排序器
        if self.enable_reranking:
            try:
                self._reranker = Reranker(config=self.base_config)
            except Exception:
                self._reranker = SimpleReranker(self.base_config)
        
        # 9. 上下文压缩器
        if self.enable_compression:
            try:
                self._compressor = ContextCompressor(self.config)
            except Exception:
                self._compressor = KeywordBasedCompressor(self.config)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """格式化文档"""
        if not docs:
            return "未找到相关文档。"
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知来源")
            content = doc.page_content.strip()
            formatted.append(f"[文档 {i}] (来源: {source})\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def _retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        执行检索
        
        根据配置选择使用父子索引或普通检索。
        """
        k = k or self.config.top_k
        
        if self.enable_parent_child and self._parent_child_retriever:
            docs = self._parent_child_retriever.retrieve(query, k=k * 2)
        elif self._hybrid_retriever:
            docs = self._hybrid_retriever.search(query, k=k * 2)
        else:
            docs = self.vectorstore_manager.similarity_search(query, k=k * 2)
        
        # 重排序
        if self.enable_reranking and len(docs) > k:
            reranked = self._reranker.rerank(query, docs, top_k=k)
            docs = [doc for doc, _ in reranked]
        else:
            docs = docs[:k]
        
        return docs
    
    def _handle_knowledge_base(
        self,
        question: str,
        reasoning_chain: List[str]
    ) -> tuple[str, List[Document], int]:
        """
        处理知识库检索
        
        Returns:
            Tuple[str, List[Document], int]: (答案, 文档列表, 迭代次数)
        """
        reasoning_chain.append("执行知识库检索...")
        
        # 查询重写
        queries = self._query_rewriter.generate_multi_queries(question)
        reasoning_chain.append(f"生成 {len(queries)} 个查询变体")
        
        # 执行检索
        all_docs = []
        seen_contents = set()
        
        for q in queries:
            docs = self._retrieve(q)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        reasoning_chain.append(f"检索到 {len(all_docs)} 个唯一文档")
        
        # 上下文压缩
        if self.enable_compression and all_docs:
            all_docs = self._compressor.compress_documents(question, all_docs)
            reasoning_chain.append("执行上下文压缩")
        
        # 截取 top-k
        docs = all_docs[:self.config.top_k]
        
        if not docs:
            return "抱歉，在知识库中没有找到相关信息。", [], 1
        
        # 生成答案
        context = self._format_docs(docs)
        prompt = ChatPromptTemplate.from_template(AGENTIC_RAG_PROMPT)
        
        response = self._llm.invoke(
            prompt.format(context=context, question=question)
        )
        answer = response.content
        
        # 自反思（如果启用）
        iterations = 1
        if self.enable_self_rag:
            quality_eval = self._self_rag.evaluate_answer_quality(
                question, context, answer
            )
            reasoning_chain.append(
                f"答案质量评估: {quality_eval.grade.value} "
                f"(分数: {quality_eval.score:.2f})"
            )
            
            # 如果质量不足，尝试迭代
            while (
                self._self_rag.should_iterate(quality_eval) and 
                iterations < self.config.self_rag_max_iterations
            ):
                iterations += 1
                reasoning_chain.append(f"开始第 {iterations} 轮迭代...")
                
                # 优化查询
                refined_query = self._self_rag.refine_query(
                    question,
                    context[:500],
                    quality_eval.missing_info
                )
                
                # 重新检索
                new_docs = self._retrieve(refined_query)
                
                # 合并文档
                for doc in new_docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        docs.append(doc)
                
                # 重新生成答案
                context = self._format_docs(docs[:self.config.top_k + iterations])
                response = self._llm.invoke(
                    prompt.format(context=context, question=question)
                )
                answer = response.content
                
                # 重新评估
                quality_eval = self._self_rag.evaluate_answer_quality(
                    question, context, answer
                )
                reasoning_chain.append(
                    f"第 {iterations} 轮质量: {quality_eval.grade.value}"
                )
        
        return answer, docs, iterations
    
    def _handle_web_search(
        self,
        question: str,
        reasoning_chain: List[str]
    ) -> tuple[str, Optional[ToolResult]]:
        """处理 Web 搜索"""
        reasoning_chain.append("执行 Web 搜索...")
        
        tool = self._tools.get("web_search")
        if not tool:
            return "Web 搜索功能未启用。", None
        
        result = tool.run(question)
        
        if not result.is_success:
            reasoning_chain.append(f"搜索失败: {result.error}")
            return f"搜索失败: {result.error}", result
        
        reasoning_chain.append(f"搜索成功，找到 {result.metadata.get('count', 0)} 个结果")
        
        # 基于搜索结果生成答案
        prompt = ChatPromptTemplate.from_template(
            """基于以下搜索结果回答用户问题：

搜索结果：
{search_results}

用户问题：{question}

请综合搜索结果，用中文给出准确的回答："""
        )
        
        response = self._llm.invoke(
            prompt.format(search_results=result.output, question=question)
        )
        
        return response.content, result
    
    def _handle_calculator(
        self,
        question: str,
        reasoning_chain: List[str]
    ) -> tuple[str, Optional[ToolResult]]:
        """处理计算请求"""
        reasoning_chain.append("执行数学计算...")
        
        tool = self._tools.get("calculator")
        if not tool:
            return "计算器功能未启用。", None
        
        # 尝试从问题中提取表达式
        # 简单方法：让 LLM 提取
        extract_prompt = ChatPromptTemplate.from_template(
            """从以下问题中提取需要计算的数学表达式。
只输出一个可以直接计算的数学表达式，不要其他内容。

问题：{question}

数学表达式："""
        )
        
        response = self._llm.invoke(extract_prompt.format(question=question))
        expression = response.content.strip()
        
        reasoning_chain.append(f"提取表达式: {expression}")
        
        result = tool.run(expression)
        
        if result.is_success:
            reasoning_chain.append(f"计算成功")
            return result.output, result
        else:
            reasoning_chain.append(f"计算失败: {result.error}")
            return f"计算失败: {result.error}", result
    
    def _handle_code_execution(
        self,
        question: str,
        reasoning_chain: List[str]
    ) -> tuple[str, Optional[ToolResult]]:
        """处理代码执行"""
        reasoning_chain.append("生成并执行代码...")
        
        tool = self._tools.get("code_executor")
        if not tool:
            return "代码执行功能未启用。", None
        
        # 让 LLM 生成代码
        code_prompt = ChatPromptTemplate.from_template(
            """根据用户需求编写 Python 代码。
代码应该简洁、安全，并通过 print 输出结果。
只输出代码，不要其他解释。

可用的模块：math, random, datetime, collections, itertools, statistics, json, re

用户需求：{question}

Python 代码："""
        )
        
        response = self._llm.invoke(code_prompt.format(question=question))
        code = response.content.strip()
        
        # 清理代码（移除可能的 markdown 代码块标记）
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        reasoning_chain.append(f"生成代码: {code[:100]}...")
        
        result = tool.run(code)
        
        if result.is_success:
            reasoning_chain.append("代码执行成功")
            
            # 整合答案
            answer_prompt = ChatPromptTemplate.from_template(
                """用户问题：{question}

执行的代码：
```python
{code}
```

执行结果：
{result}

请基于代码执行结果，用自然语言回答用户问题："""
            )
            
            final_response = self._llm.invoke(
                answer_prompt.format(
                    question=question,
                    code=code,
                    result=result.output
                )
            )
            
            return final_response.content, result
        else:
            reasoning_chain.append(f"代码执行失败: {result.error}")
            return f"代码执行失败: {result.error}", result
    
    def _handle_direct_answer(
        self,
        question: str,
        reasoning_chain: List[str]
    ) -> str:
        """处理直接回答"""
        reasoning_chain.append("生成直接回答...")
        
        prompt = ChatPromptTemplate.from_template(DIRECT_ANSWER_PROMPT)
        response = self._llm.invoke(prompt.format(question=question))
        
        return response.content
    
    def ask(self, question: str) -> AgenticRAGResult:
        """
        处理用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            AgenticRAGResult: 处理结果
        """
        logger.info(f"❓ 收到问题: {question}")
        
        reasoning_chain = []
        
        # 1. 路由决策
        if self.enable_routing:
            decision = self._router.route(question)
            route_type = decision.route_type
            confidence = decision.confidence
            reasoning_chain.append(f"路由决策: {route_type.value} (置信度: {confidence:.2f})")
            reasoning_chain.append(f"原因: {decision.reasoning}")
        else:
            # 默认使用知识库
            route_type = RouteType.KNOWLEDGE_BASE
            confidence = 1.0
            reasoning_chain.append("路由已禁用，使用知识库检索")
        
        # 2. 根据路由执行相应处理
        answer = ""
        sources = []
        tool_used = None
        tool_result = None
        iterations = 1
        quality_score = 0.0
        
        if route_type == RouteType.KNOWLEDGE_BASE:
            answer, docs, iterations = self._handle_knowledge_base(
                question, reasoning_chain
            )
            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("file_name", "未知"),
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            
        elif route_type == RouteType.WEB_SEARCH:
            if self.enable_tools:
                answer, tool_res = self._handle_web_search(question, reasoning_chain)
                tool_used = "web_search"
                tool_result = tool_res.output if tool_res else None
            else:
                answer = "Web 搜索功能未启用。"
                reasoning_chain.append("工具已禁用")
                
        elif route_type == RouteType.CALCULATOR:
            if self.enable_tools:
                answer, tool_res = self._handle_calculator(question, reasoning_chain)
                tool_used = "calculator"
                tool_result = tool_res.output if tool_res else None
            else:
                answer = "计算器功能未启用。"
                reasoning_chain.append("工具已禁用")
                
        elif route_type == RouteType.CODE_EXECUTION:
            if self.enable_tools:
                answer, tool_res = self._handle_code_execution(question, reasoning_chain)
                tool_used = "code_executor"
                tool_result = tool_res.output if tool_res else None
            else:
                answer = "代码执行功能未启用。"
                reasoning_chain.append("工具已禁用")
                
        elif route_type == RouteType.DIRECT_ANSWER:
            answer = self._handle_direct_answer(question, reasoning_chain)
        
        reasoning_chain.append("处理完成")
        
        result = AgenticRAGResult(
            answer=answer,
            route_type=route_type,
            confidence=confidence,
            sources=sources,
            tool_used=tool_used,
            tool_result=tool_result,
            iterations=iterations,
            quality_score=quality_score,
            reasoning_chain=reasoning_chain
        )
        
        logger.info(f"✅ 问答完成: {route_type.value}")
        
        return result
    
    def ask_simple(self, question: str) -> str:
        """
        简化版问答（只返回答案）
        
        Args:
            question: 用户问题
            
        Returns:
            str: 答案
        """
        result = self.ask(question)
        return result.answer

