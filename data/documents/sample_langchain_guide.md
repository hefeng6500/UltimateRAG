# LangChain 入门指南

## 什么是 LangChain？

LangChain 是一个用于构建 LLM（大语言模型）应用的开源框架。它提供了一套完整的工具和抽象，帮助开发者快速构建基于 LLM 的应用程序。

## 核心概念

### 1. Chat Models

Chat Models 是 LangChain 中最基础的组件，用于与 LLM 进行对话。

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("Hello, how are you?")
```

### 2. Prompts

Prompt 模板用于构建发送给 LLM 的提示词。

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "请用中文回答以下问题：{question}"
)
```

### 3. Chains

Chain 是将多个组件连接起来形成工作流的方式。

```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "什么是人工智能？"})
```

### 4. Retrievers

Retriever 用于从向量数据库中检索相关文档。

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("查询问题")
```

## 安装

```bash
pip install langchain langchain-openai langchain-community
```

## 环境配置

设置 OpenAI API Key：

```bash
export OPENAI_API_KEY="your-api-key"
```

## 最佳实践

1. **使用最新版本**：LangChain 更新频繁，建议使用最新版本
2. **模块化设计**：将不同功能拆分成独立模块
3. **错误处理**：为 LLM 调用添加适当的错误处理
4. **日志记录**：使用 LangSmith 进行调试和监控

## 常见问题

### Q: 如何处理长文档？
A: 使用 Text Splitter 将长文档切分成小块。

### Q: 如何提高检索质量？
A: 使用混合检索（向量 + 关键词）和 Re-ranking。

### Q: 如何降低成本？
A: 使用缓存、选择合适的模型、优化 Prompt 长度。

## 版本更新

- v1.1.3: 最新稳定版本
- 新增 Deep Agents 支持
- 改进的 Streaming API
- 更好的 MCP 协议支持
