# Phase 2: 质量飞跃 (Advanced RAG) - 学习总结

## 📚 本阶段学习收获

> [!TIP]
> Phase 2 实现了多项高级 RAG 技术，显著提升了检索质量。

---

## 🎯 完成的功能

### 1. 语义分块 (`semantic_chunker.py`)
- 基于句子嵌入判断语义边界
- 动态调整分块大小
- 保持段落完整性

### 2. 元数据提取 (`metadata_extractor.py`)
- 自动提取标题、日期、文件信息
- 支持元数据过滤检索
- 增强文档可追溯性

### 3. 混合检索 (`hybrid_retriever.py`)
- BM25 关键词检索 + 向量语义检索
- 倒排融合算法 (RRF) 合并结果
- 中英文混合分词支持

### 4. 查询重写 (`query_rewriter.py`)
- 多路查询生成
- HyDE 假设文档嵌入
- 查询扩展（同义词）

### 5. 重排序 (`reranker.py`)
- BGE-Reranker Cross-Encoder
- 精细化排序 Top-K
- 简单规则重排备选方案

---

## 💡 技术要点

### 混合检索融合算法 (RRF)
```python
# 倒排融合公式: score = Σ 1/(k + rank)
for rank, (doc, _) in enumerate(results):
    doc_scores[doc_key] += 1 / (k + rank + 1)
```

### 语义分块判断
```python
# 当相似度低于阈值且块足够大时断开
if similarity < threshold and len(chunk) >= min_size:
    chunks.append(current_chunk)
    current_chunk = new_sentence
```

### HyDE 工作原理
```
用户问题 -> LLM生成假设答案 -> 用假设答案检索 -> 找到真实文档
```

---

## 📊 测试结果

```
✅ 6 个单元测试全部通过
- TestMetadataExtractor: 元数据提取测试
- TestHybridRetriever: 混合检索测试
- TestQueryRewriter: 查询重写测试
- TestReranker: 重排序测试
```

---

## ⚠️ 关键技术点

| 技术 | 解决的问题 | ROI |
|------|-----------|-----|
| 混合检索 | 专有名词搜索不到 | ⭐⭐⭐⭐⭐ |
| Re-ranking | 粗检索结果排序不准 | ⭐⭐⭐⭐⭐ |
| 查询重写 | 用户表达不清晰 | ⭐⭐⭐⭐ |
| 语义分块 | 固定分块切断语义 | ⭐⭐⭐ |

---

## 🔗 文件结构

```
src/stage_2/
├── __init__.py               # 包初始化
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

## 🚀 使用方法

```bash
# 运行 Advanced RAG 系统
cd /path/to/UltimateRAG
source .venv/bin/activate
python -m src.stage_2.main --data ./data/documents

# 可选参数
--no-semantic    # 禁用语义分块
--no-rerank      # 禁用重排序
--reindex        # 强制重新索引
```

---

## 📖 参考资料

- [LangChain Retrieval 文档](https://docs.langchain.com/oss/python/langchain/retrieval)
- [BGE-Reranker 论文](https://arxiv.org/abs/2309.07597)
- [HyDE 论文](https://arxiv.org/abs/2212.10496)
