"""
RAG 问答链模块

实现完整的 RAG 问答流程：
用户提问 -> 向量检索 -> 构建 Prompt -> LLM 生成回答
"""

from typing import List, Optional, Dict, Any
from loguru import logger

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import Config, get_config
from .vectorstore import VectorStoreManager


# 默认的 RAG Prompt 模板
DEFAULT_RAG_PROMPT = """你是一个专业的问答助手。请根据以下参考文档回答用户的问题。

如果参考文档中没有相关信息，请诚实地说"根据提供的文档，我无法回答这个问题"。
不要编造信息，只基于参考文档的内容进行回答。

参考文档：
{context}

用户问题：{question}

请用中文回答："""

# DEFAULT_RAG_PROMPT = """
# 你是一个能够进行深度阅读理解、总结和分析的中文助手。

# 请基于参考文档回答用户问题，但允许你进行：
# - 归纳总结
# - 风格分析
# - 观点选择
# - 句子评价
# - 合理推断（必须基于文档给出的内容）

# 禁止编造超出文档不存在的事实信息，但允许在文中信息基础上做解释性、分析性和判断性的扩展。

# 如果文档完全无关，才回答："根据提供的文档，我无法回答这个问题"。

# 【参考文档】
# {context}

# 【用户问题】
# {question}

# 请用中文回答：
# """


class RAGChain:
    """
    RAG 问答链
    
    组装检索器、Prompt 和 LLM，实现端到端的问答功能。
    """
    
    def __init__(
        self,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        config: Optional[Config] = None,
        prompt_template: Optional[str] = None
    ):
        """
        初始化 RAG 问答链
        
        Args:
            vectorstore_manager: 向量存储管理器
            config: 配置对象
            prompt_template: 自定义 Prompt 模板
        """
        self.config = config or get_config()
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager(self.config)
        
        # 初始化 LLM
        self._llm = self._create_llm()
        
        # 创建 Prompt 模板
        self._prompt = ChatPromptTemplate.from_template(
            prompt_template or DEFAULT_RAG_PROMPT
        )
        
        # 构建 RAG 链
        self._chain = self._build_chain()
        
        logger.info("🔗 RAG 问答链初始化完成")
    
    def _create_llm(self) -> ChatOpenAI:
        """
        创建 LLM 实例
        
        Returns:
            ChatOpenAI: LangChain ChatOpenAI 实例
        """
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.7,
        }
        
        # 如果设置了自定义 base_url
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        
        llm = ChatOpenAI(**kwargs)
        logger.info(f"🤖 LLM 初始化完成: {self.config.model_name}")
        
        return llm
    
    def _format_docs(self, docs: List[Document]) -> str:
        """
        格式化检索到的文档
        
        Args:
            docs: 文档列表
            
        Returns:
            str: 格式化后的文档文本
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知来源")
            content = doc.page_content.strip()
            formatted.append(f"[文档 {i}] (来源: {source})\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def _build_chain(self):
        """
        构建 RAG 链
        
        Returns:
            Runnable: LangChain Runnable 链
        """
        retriever = self.vectorstore_manager.as_retriever()
        
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask(self, question: str) -> str:
        """
        提问并获取回答
        
        Args:
            question: 用户问题
            
        Returns:
            str: LLM 生成的回答
        """
        logger.info(f"❓ 收到问题: {question}")
        
        try:
            answer = self._chain.invoke(question)
            logger.info(f"✅ 回答生成完成")
            return answer
        except Exception as e:
            logger.error(f"❌ 生成回答失败: {e}")
            raise
    
    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """
        提问并返回回答及来源文档
        
        Args:
            question: 用户问题
            
        Returns:
            Dict: 包含 answer 和 sources 的字典
        """
        logger.info(f"❓ 收到问题 (带来源): {question}")
        
        # 先检索文档
        docs = self.vectorstore_manager.similarity_search(question)
        
        # 构建上下文
        context = self._format_docs(docs)
        
        # 生成回答
        prompt = self._prompt.format(context=context, question=question)
        answer = self._llm.invoke(prompt).content
        
        # 整理来源信息
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("file_name", "未知"),
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question
        }
    
    def stream(self, question: str):
        """
        流式生成回答
        
        Args:
            question: 用户问题
            
        Yields:
            str: 回答的文本块
        """
        logger.info(f"❓ 收到问题 (流式): {question}")
        
        for chunk in self._chain.stream(question):
            yield chunk
