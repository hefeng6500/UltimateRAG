# Phase 1: 原型验证 (MVP) - 学习总结

## 📚 本阶段学习收获

> [!TIP]
> Phase 1 完成了 RAG 系统的基础架构搭建，实现了从文档加载到问答的完整流程。

---

## 🎯 完成的功能

### 1. 配置管理 (`config.py`)
- 使用 `dataclass` 定义配置结构
- 支持从环境变量加载配置
- 实现单例模式的配置管理

### 2. 文档加载 (`document_loader.py`)
- 支持 PDF、Markdown、TXT、DOCX 四种格式
- 自动识别文件类型并选择加载器
- 支持递归加载整个目录

### 3. 文本分块 (`chunker.py`)
- 使用 `RecursiveCharacterTextSplitter` 智能分割
- 支持中英文混合切分
- 可配置块大小和重叠

### 4. 嵌入模型 (`embedder.py`)
- 封装 OpenAI Embeddings
- 支持自定义 base_url（兼容 DeepSeek 等）
- 懒加载机制

### 5. 向量存储 (`vectorstore.py`)
- 使用 ChromaDB 持久化存储
- 支持相似度检索和带分数检索
- 可转换为 LangChain Retriever

### 6. RAG 问答链 (`rag_chain.py`)
- 组装检索器、Prompt 和 LLM
- 支持普通问答和带来源问答
- 支持流式输出

---

## 💡 技术要点

### LangChain 1.1.3 新特性
```python
# 使用 langchain_core 的基础类
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 使用 LCEL (LangChain Expression Language) 构建链
chain = prompt | llm | StrOutputParser()
```

### ChromaDB 集成
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)
```

### 分块最佳实践
- **chunk_size**: 512 字符（约 100-200 token）
- **chunk_overlap**: 50 字符（约 10%）
- **分隔符优先级**: 段落 → 句子 → 词语

---

## ⚠️ 遇到的问题与解决方案

### 问题 1: LangChain 版本兼容性
**现象**: 旧代码使用 `from langchain.xxx` 导入报错
**解决**: 使用新的模块结构
- `langchain_core`: 核心基础类
- `langchain_openai`: OpenAI 集成
- `langchain_community`: 社区集成

### 问题 2: ChromaDB 持久化
**现象**: 每次运行都需要重新索引
**解决**: 指定 `persist_directory` 参数，自动保存和加载

---

## 📊 测试结果

```
✅ 8 个单元测试全部通过
- TestConfig: 配置加载测试
- TestDocumentLoader: 文档加载测试
- TestTextChunker: 分块测试
```

---

## 📈 Phase 1 局限性

1. **分块太碎**: 固定大小分块可能切断完整的语义
2. **检索不准**: 纯向量检索对关键词匹配不友好
3. **无排序优化**: 检索结果未经过重排序

> [!NOTE]
> 这些问题将在 Phase 2 (Advanced RAG) 中解决。

---

## 🔗 文件结构

```
src/stage_1/
├── __init__.py           # 包初始化
├── config.py             # 配置管理
├── document_loader.py    # 文档加载器
├── chunker.py            # 文本分块器
├── embedder.py           # 嵌入模型
├── vectorstore.py        # 向量存储
├── rag_chain.py          # RAG 问答链
├── main.py               # 主入口
└── tests/
    └── test_rag.py       # 单元测试
```

---

## 🚀 使用方法

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 2. 运行 RAG 系统
cd /path/to/UltimateRAG
source venv/bin/activate
python -m stage_1.main --data ./data/documents

# 3. 开始问答
# 输入问题即可获得回答
```

---

## 📖 参考资料

- [LangChain 官方文档](https://docs.langchain.com/)
- [ChromaDB 官方文档](https://docs.trychroma.com/)
- [LangChain 1.1.3 Release Notes](https://github.com/langchain-ai/langchain/releases)
