# Stage 4 快速入门指南

> 5 分钟上手 GraphRAG & Fine-tuning

---

## 🚀 环境准备

### 1. 安装依赖

```bash
cd /path/to/UltimateRAG
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# API 配置 (二选一)
OPENAI_API_KEY=your_openai_api_key
# 或使用阿里云
DASHSCOPE_API_KEY=your_dashscope_api_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型配置
MODEL_NAME=qwen-plus
EMBEDDING_MODEL=text-embedding-v3

# 图存储 (可选，默认 memory)
GRAPH_STORE_TYPE=memory
```

### 3. 准备数据

将文档放入 `data/documents/` 目录，支持格式：
- PDF (`.pdf`)
- Markdown (`.md`)
- Word (`.docx`)
- 文本 (`.txt`)

---

## 📊 GraphRAG 快速体验

### 方式一：交互式演示

```bash
python -m src.stage_4.main
```

选择 `1` 进入 GraphRAG 演示模式。

### 方式二：代码调用

```python
from src.stage_1.document_loader import DocumentLoader
from src.stage_1.chunker import TextChunker
from src.stage_4.graph_rag import GraphRAGChain

# 1. 加载文档
loader = DocumentLoader()
documents = loader.load_directory("./data/documents")

# 2. 分块
chunker = TextChunker()
chunks = chunker.split_documents(documents)

# 3. 初始化 GraphRAG
graph_rag = GraphRAGChain(
    documents=chunks,
    graph_name="my_knowledge_graph",
    force_rebuild=True,  # 首次运行设为 True
)

# 4. 构建知识图谱
graph_rag.build_knowledge_graph(chunks)

# 5. 查看图谱统计
stats = graph_rag.get_statistics()
print(f"实体数: {stats['num_nodes']}, 关系数: {stats['num_edges']}")

# 6. 提问
result = graph_rag.ask("文档中提到的主要人物有哪些？他们之间有什么关系？")
print(result.answer)
```

---

## 🎯 终极 RAG 快速体验

整合 Stage 1-4 所有能力的最强 RAG：

```python
from src.stage_4.ultimate_rag_chain import UltimateRAGChain, RetrievalMode

# 初始化（自动整合向量检索 + 图检索 + 自反思）
ultimate_rag = UltimateRAGChain(
    documents=chunks,
    enable_routing=True,      # 智能路由
    enable_self_rag=True,     # 自反思
    enable_graph_rag=True,    # 图检索
    enable_reranking=True,    # 重排序
)

# 自动模式（系统自动选择最佳检索策略）
result = ultimate_rag.ask("问题内容")

# 指定检索模式
result = ultimate_rag.ask("关系类问题", retrieval_mode=RetrievalMode.GRAPH)
result = ultimate_rag.ask("一般问题", retrieval_mode=RetrievalMode.VECTOR)
result = ultimate_rag.ask("复杂问题", retrieval_mode=RetrievalMode.FUSION)
```

**检索模式说明：**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `AUTO` | 自动选择 | 不确定问题类型时 |
| `VECTOR` | 纯向量检索 | 一般语义相似问题 |
| `HYBRID` | 混合检索 | 包含专有名词的问题 |
| `GRAPH` | 纯图检索 | 关系查询、路径查询 |
| `FUSION` | 融合检索 | 复杂的综合性问题 |

---

## 📚 微调数据生成

### 生成 LLM 微调数据

```python
from src.stage_4.fine_tuning import LLMFineTuner

# 初始化
finetuner = LLMFineTuner()

# 生成 QA 对
qa_pairs = finetuner.generate_qa_pairs(
    documents=chunks,
    pairs_per_doc=5,  # 每个文档生成 5 个 QA
)

# 导出为不同格式
finetuner.export_jsonl()                    # OpenAI 格式 (JSONL)
finetuner.export_json()                     # Alpaca 格式 (JSON)

# 查看统计
print(finetuner.get_statistics())
```

**输出文件位置：** `./data/finetune/`

### 生成 Embedding 训练数据

```python
from src.stage_4.fine_tuning import EmbeddingFineTuner

# 初始化
emb_finetuner = EmbeddingFineTuner(
    base_model="BAAI/bge-base-zh-v1.5",
    output_dir="./models/my_embedding",
)

# 生成训练数据
pairs, triplets = emb_finetuner.generate_training_data(chunks)

# 保存
emb_finetuner.save_training_data()

# 开始微调（需要 GPU）
# emb_finetuner.train(epochs=3)
```

---

## 🔍 常用操作速查

### 查询实体信息

```python
info = graph_rag.get_entity_info("华为")
print(f"实体: {info['entity']['name']}")
print(f"邻居: {[n['name'] for n in info['neighbors']]}")
print(f"关系: {len(info['relations'])} 条")
```

### 查找实体间路径

```python
path = graph_rag.find_path("任正非", "深圳")
if path:
    for step in path:
        entity = step['entity']['name']
        relation = step['relation']['relation_type'] if step['relation'] else '起点'
        print(f"  {entity} [{relation}]")
```

### 生成全局摘要

```python
summary = graph_rag.generate_global_summary()
print(summary)
```

### 获取图谱统计

```python
stats = graph_rag.get_statistics()
print(f"实体类型分布: {stats['entity_type_counts']}")
print(f"关系类型分布: {stats['relation_type_counts']}")
```

---

## 🛠️ 常见问题

### Q: 图谱构建太慢？

A: 图谱构建需要调用 LLM 进行实体/关系抽取，可以：
- 减少文档数量进行测试
- 使用更快的 LLM（如 qwen-turbo）
- 设置 `force_rebuild=False` 复用已有图谱

### Q: 如何使用 Neo4j？

A: 在 `.env` 中配置：

```env
GRAPH_STORE_TYPE=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

然后安装 Neo4j Desktop 或使用 Docker 启动 Neo4j。

### Q: 微调需要什么硬件？

A: 
- Embedding 微调：推荐 8GB+ 显存的 GPU
- 仅生成数据：CPU 即可

### Q: 如何评估效果？

A: 参考 Stage 5 的评估框架，或手动评估：
- 检查抽取的实体是否准确
- 检查关系是否有意义
- 对比有无图检索的答案质量

---

## 📁 输出文件说明

| 文件/目录 | 说明 |
|----------|------|
| `./data/graph_db/*.json` | 知识图谱持久化文件 |
| `./data/finetune/train_openai.jsonl` | OpenAI 格式微调数据 |
| `./data/finetune/train_alpaca.json` | Alpaca 格式微调数据 |
| `./data/finetune/qa_pairs.json` | 原始 QA 对 |
| `./models/finetuned_embedding/` | 微调后的 Embedding 模型 |

---

## 🎉 下一步

- 阅读 [完整文档](./README.md) 了解更多功能
- 查看 [开发计划](./plan.md) 了解实现细节
- 探索 Stage 5 的评估和监控功能

Happy Coding! 🚀

