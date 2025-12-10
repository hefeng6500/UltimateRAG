# Phase 2: 质量飞跃 (Advanced RAG) - 开发计划

## 📋 目标
大幅提升检索的准确率（Precision）和召回率（Recall）。

**核心理念：** 优化检索链路的每一个环节

---

## 🏗️ 新增技术

| 组件 | 技术 | 说明 |
|------|------|------|
| **语义分块** | Semantic Chunking | 根据语义完整性切分 |
| **混合检索** | BM25 + 向量检索 | 解决关键词匹配问题 |
| **查询重写** | Query Rewrite / HyDE | 多路查询，提升召回 |
| **重排序** | BGE-Reranker | 精细排序，提升精度 |

---

## 📁 文件结构

```
src/stage_2/
├── __init__.py               # 包初始化
├── config.py                 # 扩展配置
├── semantic_chunker.py       # 语义分块器
├── metadata_extractor.py     # 元数据提取
├── hybrid_retriever.py       # 混合检索器
├── query_rewriter.py         # 查询重写
├── reranker.py               # 重排序器
├── advanced_rag_chain.py     # 高级 RAG 链
├── main.py                   # 主入口
└── tests/
    └── test_advanced_rag.py  # 单元测试
```

---

## 🔧 实现步骤

### Step 1: 语义分块 (Semantic Chunking)
- 使用 LangChain 的 SemanticChunker
- 基于句子嵌入判断语义边界
- 保持段落完整性

### Step 2: 元数据提取
- 提取文件名、页码、日期
- 提取标题和章节信息
- 支持过滤检索

### Step 3: 混合检索 (Hybrid Search)
- 实现 BM25 关键词检索
- 结合向量检索结果
- 融合排序算法

### Step 4: 查询重写 (Query Rewrite)
- 多路查询生成
- HyDE（假设文档嵌入）
- Query Expansion

### Step 5: Re-ranking
- 集成 BGE-Reranker
- 精细化排序 Top-K 结果
- 过滤低相关结果

### Step 6: 测试与对比
- 对比 Phase 1 的检索效果
- 量化评估准确率提升

---

## ✅ 预期成果

- 回答准确率大幅提升
- 不再因关键词匹配不上而瞎编
- 能够处理专有名词检索（如 SK-1024）
