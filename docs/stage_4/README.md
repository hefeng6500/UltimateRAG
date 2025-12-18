# Stage 4: GraphRAG & Fine-tuning

## 🎯 概述

Stage 4 是 UltimateRAG 的最高阶段，实现了知识图谱增强的 RAG (GraphRAG) 和领域微调能力。

### 核心能力

| 能力 | 描述 |
|------|------|
| **知识图谱** | 自动从文档中抽取实体和关系，构建知识图谱 |
| **图检索** | 基于图遍历的智能检索，发现隐性关联 |
| **全局摘要** | 基于社区检测生成全局性摘要 |
| **Embedding 微调** | 使用私有数据微调 Embedding 模型 |
| **LLM 微调数据** | 自动生成高质量的微调训练数据 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

对于 Neo4j 支持（可选）：
```bash
pip install neo4j
```

### 2. 配置环境

在 `.env` 文件中添加：

```env
# 基础配置（继承自 Stage 1-3）
OPENAI_API_KEY=your_api_key
MODEL_NAME=gpt-4o

# GraphRAG 配置
GRAPH_STORE_TYPE=memory  # 或 neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# 微调配置
EMBEDDING_FINETUNE_MODEL=BAAI/bge-base-zh-v1.5
```

### 3. 使用 GraphRAG

```python
from src.stage_4.main import run_graph_rag_demo

# 运行演示
run_graph_rag_demo()
```

或者手动使用：

```python
from src.stage_4.graph_rag import GraphRAGChain
from src.stage_1.document_loader import DocumentLoader

# 加载文档
loader = DocumentLoader()
documents = loader.load_directory("./data/documents")

# 初始化 GraphRAG
graph_rag = GraphRAGChain(documents)

# 构建知识图谱（首次使用）
graph_rag.build_knowledge_graph()

# 提问
result = graph_rag.ask("分析文档中提到的公司之间的关系")
print(result.answer)
```

---

## 📖 模块详解

### 1. GraphRAG 模块

#### 实体抽取 (EntityExtractor)

从文本中识别和提取命名实体：

```python
from src.stage_4.graph_rag import EntityExtractor

extractor = EntityExtractor()
entities = extractor.extract("华为公司在深圳成立，任正非是创始人。")

# 输出:
# [
#   Entity(name="华为公司", type="Organization", ...),
#   Entity(name="深圳", type="Location", ...),
#   Entity(name="任正非", type="Person", ...)
# ]
```

支持的实体类型：
- `Person` - 人物
- `Organization` - 组织/公司
- `Location` - 地点
- `Event` - 事件
- `Concept` - 概念/术语
- `Product` - 产品
- `Time` - 时间

#### 关系抽取 (RelationExtractor)

提取实体之间的关系：

```python
from src.stage_4.graph_rag import RelationExtractor

extractor = RelationExtractor()
relations = extractor.extract(
    text="华为公司在深圳成立，任正非是创始人。",
    entities=entities
)

# 输出:
# [
#   Relation(source="任正非", target="华为公司", type="founded", ...),
#   Relation(source="华为公司", target="深圳", type="located_in", ...)
# ]
```

#### 知识图谱 (KnowledgeGraph)

管理实体和关系的图结构：

```python
from src.stage_4.graph_rag import KnowledgeGraph

kg = KnowledgeGraph()

# 添加实体
kg.add_entity(entity)

# 添加关系
kg.add_relation(relation)

# 查询实体的邻居
neighbors = kg.get_neighbors("华为公司", hops=2)

# 查找路径
path = kg.find_path("任正非", "深圳")

# 获取子图
subgraph = kg.get_subgraph(["华为公司", "任正非"])
```

#### 图检索器 (GraphRetriever)

基于图结构进行智能检索：

```python
from src.stage_4.graph_rag import GraphRetriever

retriever = GraphRetriever(knowledge_graph)

# 检索相关实体和上下文
results = retriever.retrieve(
    query="华为的创始人是谁？",
    top_k=5
)
```

### 2. 微调模块

#### Embedding 微调

使用私有数据微调 Embedding 模型：

```python
from src.stage_4.fine_tuning import EmbeddingFineTuner

fine_tuner = EmbeddingFineTuner(
    base_model="BAAI/bge-base-zh-v1.5",
    output_dir="./models/my_embedding"
)

# 生成训练数据
training_data = fine_tuner.generate_training_data(documents)

# 训练
fine_tuner.train(training_data, epochs=3)
```

#### LLM 微调数据准备

自动生成微调训练数据：

```python
from src.stage_4.fine_tuning import LLMFineTuner

finetuner = LLMFineTuner()

# 生成 QA 对
qa_pairs = finetuner.generate_qa_pairs(documents)

# 导出为不同格式
finetuner.export_jsonl(qa_pairs, "train.jsonl")  # OpenAI 格式
finetuner.export_alpaca(qa_pairs, "train_alpaca.json")  # Alpaca 格式
```

---

## 🔧 配置说明

### Stage4Config 参数

```python
@dataclass
class Stage4Config(Stage3Config):
    # GraphRAG 配置
    graph_store_type: str = "memory"  # memory / neo4j
    entity_types: List[str] = ...      # 支持的实体类型
    relation_types: List[str] = ...    # 支持的关系类型
    max_entities_per_chunk: int = 20   # 每个文档块最大实体数
    max_relations_per_chunk: int = 30  # 每个文档块最大关系数
    graph_traversal_depth: int = 2     # 图遍历深度
    
    # Neo4j 配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    
    # Embedding 微调配置
    embedding_finetune_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_finetune_epochs: int = 3
    embedding_finetune_batch_size: int = 32
    
    # LLM 微调数据配置
    qa_pairs_per_doc: int = 5
    qa_difficulty_levels: List[str] = ["easy", "medium", "hard"]
```

---

## 📊 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        UltimateRAG Stage 4                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │    GraphRAG 模块      │    │    微调模块           │          │
│  ├──────────────────────┤    ├──────────────────────┤          │
│  │ • EntityExtractor    │    │ • EmbeddingFineTuner │          │
│  │ • RelationExtractor  │    │ • LLMFineTuner       │          │
│  │ • KnowledgeGraph     │    │ • TrainingDataGen    │          │
│  │ • GraphStore         │    │                      │          │
│  │ • GraphRetriever     │    │                      │          │
│  │ • GraphRAGChain      │    │                      │          │
│  └──────────┬───────────┘    └───────────┬──────────┘          │
│             │                            │                      │
│             └────────────┬───────────────┘                      │
│                          │                                      │
│             ┌────────────▼───────────────┐                      │
│             │     UltimateRAGChain       │                      │
│             │   (整合 Stage 1-4)         │                      │
│             └────────────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 继承
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Stage 1-3 组件                              │
│  VectorStore | HybridRetriever | Reranker | AgenticRAG | ...   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎓 典型使用场景

### 场景 1: 企业关系分析

```python
# 分析合同文档中的公司关系
result = graph_rag.ask("找出 A 公司和 B 公司之间所有的合作关系")

# 返回:
# - 详细的关系描述
# - 关系路径可视化
# - 相关证据文档
```

### 场景 2: 人物关系网络

```python
# 分析新闻中的人物关系
result = graph_rag.ask("张三和李四是什么关系？他们之间有什么交集？")
```

### 场景 3: 全局概括

```python
# 对大量文档进行全局性总结
result = graph_rag.ask("总结过去三年公司在 AI 领域的战略布局")
```

### 场景 4: 领域适配

```python
# 微调 Embedding 模型适配医疗领域
fine_tuner.train(medical_documents)

# 使用微调后的模型
graph_rag.set_embedding_model("./models/medical_embedding")
```

---

## ⚠️ 注意事项

1. **性能考虑**
   - 实体/关系抽取会增加 LLM 调用，建议批量处理
   - 大规模图谱建议使用 Neo4j
   - 微调需要较大显存

2. **成本控制**
   - 设置合理的 `max_entities_per_chunk`
   - 使用缓存避免重复抽取
   - 考虑使用本地 LLM 进行抽取

3. **数据质量**
   - 抽取质量依赖于 LLM 能力
   - 建议人工审核关键实体/关系
   - 定期清理冗余实体

---

## 📈 与前序阶段对比

| 维度 | Stage 3 | Stage 4 |
|------|---------|---------|
| 检索方式 | 向量 + BM25 | 向量 + BM25 + 图遍历 |
| 上下文理解 | 局部文档 | 全局关联 |
| 问题类型 | 单文档问答 | 跨文档关系推理 |
| 定制能力 | 通用模型 | 领域微调 |
| 复杂度 | 中等 | 高 |

---

## 🔗 相关资源

- [Microsoft GraphRAG 论文](https://arxiv.org/abs/2404.16130)
- [Neo4j 官方文档](https://neo4j.com/docs/)
- [Sentence Transformers 微调指南](https://www.sbert.net/docs/training/overview.html)

