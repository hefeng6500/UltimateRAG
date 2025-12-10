# Phase 1: 原型验证 (MVP) - 开发计划

## 📋 目标
跑通流程，构建一个能用的问答机器人。

**核心理念：** 先有再优（Make it work）

---

## 🏗️ 技术选型

| 组件 | 选择 | 说明 |
|------|------|------|
| **编排框架** | LangChain 1.1.3 | 最新版，功能完善 |
| **LLM** | DeepSeek / OpenAI | 通过 API 调用 |
| **向量库** | ChromaDB | 轻量级，易于本地开发 |
| **Embedding** | OpenAI text-embedding-3-small | 效果好，成本低 |

---

## 📁 文件结构

```
src/stage_1/
├── __init__.py           # 包初始化
├── config.py             # 配置管理（API key、模型参数等）
├── document_loader.py    # 文档加载器（支持 PDF、Markdown、TXT）
├── chunker.py            # 文本分块器（FixedSizeChunking）
├── embedder.py           # 嵌入模型封装
├── vectorstore.py        # ChromaDB 向量存储
├── rag_chain.py          # RAG 问答链
├── main.py               # 主入口，演示完整流程
└── tests/
    └── test_rag.py       # 单元测试
```

---

## 🔧 实现步骤

### Step 1: 环境配置
- 安装 langchain 全家桶依赖
- 配置 `.env` 环境变量

### Step 2: 文档加载
- 实现 PDF 加载器
- 实现 Markdown 加载器
- 实现 TXT 加载器

### Step 3: 文本分块
- 实现 FixedSizeChunking（512 token，重叠 50 token）
- 封装统一的分块接口

### Step 4: 向量存储
- 初始化 ChromaDB
- 实现文档向量化存储
- 实现向量检索接口

### Step 5: RAG 问答链
- 实现 Prompt 模板
- 组装检索 + 生成链
- 实现问答接口

### Step 6: 测试验证
- 编写单元测试
- 运行端到端测试

---

## ✅ 预期成果

一个能回答简单文档问题的 Bot，具备以下能力：
1. 加载多种格式文档
2. 自动分块和向量化
3. 语义检索相关内容
4. 基于上下文生成回答

**已知局限：** 可能答非所问，或因切片太碎丢失上下文（将在 Phase 2 解决）
