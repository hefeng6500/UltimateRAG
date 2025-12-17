# UltimateRAG

一个按路线图逐步实现的 RAG 学习项目：从 **Naive RAG → Advanced RAG → Agentic RAG → GraphRAG & Fine-tuning → RAGOps**，在本地用 LangChain + 向量库把“文档问答”跑通，并逐步加入语义分块、混合检索、重排序、路由、自反思与工具调用等能力。

从最简单的 Demo 进化到企业级、甚至科研级的 RAG 系统。

路线图（五阶段总规划）见：`docs/README.md`。

---

## 项目目标

- **学习目标**：把 RAG 的关键链路（Indexing / Retrieval / Generation / Evaluation）按阶段拆解实现，形成可复用模块。
- **工程目标**：每个阶段都有可运行的 `main.py` 入口 + 可验证的单测，用最小闭环体现关键技术点与 trade-off。

---

## 路线图与当前进展（对齐 `docs/README.md`）

| Roadmap Phase                                  | 对应实现       | 状态      | 你能在代码里看到什么                              |
| ---------------------------------------------- | -------------- | --------- | ------------------------------------------------- |
| Phase 1: MVP（Hello World）                    | `src/stage_1/` | ✅ 已实现 | 文档加载/分块/向量化/检索/问答闭环                |
| Phase 2: Advanced RAG（解决检索不准）          | `src/stage_2/` | ✅ 已实现 | 语义分块、元数据、混合检索、Query Rewrite、Rerank |
| Phase 3: Modular & Agentic RAG（解决逻辑复杂） | `src/stage_3/` | ✅ 已实现 | 路由、自反思、工具调用、父子索引、上下文压缩      |
| Phase 4: GraphRAG & Fine-tuning（深度认知）    | -              | ⏳ 规划中 | 知识图谱、领域适配（Embedding/LLM）               |
| Phase 5: RAGOps（持续迭代）                    | -              | ⏳ 规划中 | 评估/可观测性/自动化回归                          |

每个已实现阶段的学习总结与使用方式：

- `docs/stage_1/README.md`
- `docs/stage_2/README.md`
- `docs/stage_3/README.md`

---

## 每阶段技术栈与知识点（你将学到什么）

### Stage 1（Phase 1 / MVP）

- **技术栈**
  - **Orchestration**：LangChain（LCEL/Runnable）
  - **LLM / Embedding**：OpenAI-compatible API（`langchain-openai`）
  - **Vector DB**：Chroma（`langchain-chroma`）
  - **文档解析**：`pypdf`、`unstructured`（md）、`docx2txt`
- **关键知识点**
  - **文档加载**：按后缀选择 loader，统一产出 `Document`
  - **分块策略**：固定分块 + overlap 的基本 trade-off
  - **向量化与持久化**：Chroma 落盘、避免重复建库
  - **最小 RAG 链**：检索 → 拼接上下文 → Prompt → 生成
- **如何启动**
  - `python -m src.stage_1.main --data ./data/documents`

### Stage 2（Phase 2 / Advanced RAG）

- **技术栈**
  - **语义分块**：基于 embedding 相似度/阈值的 chunk 边界判断（`SemanticChunker`）
  - **Hybrid Retrieval**：BM25（`rank-bm25`）+ 向量检索 + 融合（RRF）
  - **Query Rewrite / HyDE**：LLM 生成多路查询、假设答案检索
  - **Rerank**：Cross-Encoder（`sentence-transformers` CrossEncoder，BGE-Reranker）
- **关键知识点**
  - **召回 vs 精排**：粗检索 Top-N + 重排序取 Top-K 的“银弹”结构
  - **关键词检索补短板**：专有名词/缩写对向量检索不友好
  - **查询改写提升召回**：把用户表达翻译成“更可检索”的问题
- **如何启动**
  - `python -m src.stage_2.main --data ./data/documents`

### Stage 3（Phase 3 / Agentic RAG）

- **技术栈**
  - **Routing**：LLM 结构化输出（`pydantic`）做路由决策与兜底
  - **Self-RAG**：相关性/质量自评估 + 多轮重检索
  - **Tool Use**：Web 搜索（`ddgs`）、计算器、受限 Python 执行
  - **Context 管理**：父子索引（精准匹配 + 完整上下文）、上下文压缩
- **关键知识点**
  - **从 Pipeline 到 Agent**：动态决策、按需调用能力与自我纠错
  - **成本/延迟意识**：路由与多轮迭代会增加调用次数与耗时
  - **安全边界**：代码执行沙箱与白名单控制
- **如何启动**
  - `python -m src.stage_3.main --data ./data/documents`

### Stage 4（Phase 4 / GraphRAG & Fine-tuning）

- **状态**：⏳ 规划中（路线图见 `docs/README.md`）
- **目标**
  - **GraphRAG**：用结构化关系（实体-关系图）增强跨文档、跨章节的关联检索与推理
  - **Fine-tuning / 领域适配**：让 embedding/LLM 更懂领域术语、提升召回与回答风格一致性
- **技术栈（候选）**
  - **图数据库**：Neo4j / NebulaGraph（或微软 GraphRAG 思路）
  - **信息抽取**：LLM 抽取实体/关系、或者规则+模型混合抽取
  - **检索**：Graph traversal + 向量检索（Hybrid: Graph + Vector）
  - **模型适配**：Embedding 微调、提示词/对齐、必要时 SFT/LoRA
- **关键知识点**
  - **向量检索的边界**：相似度无法表达“关系路径”，图检索擅长“链路/因果/关联”问题
  - **建图成本与增量更新**：抽取质量、去重对齐、图更新策略
- **如何启动**
  - 暂未实现（后续会在根目录补 `src/stage_4/` 与入口）

### Stage 5（Phase 5 / RAGOps）

- **状态**：⏳ 规划中（路线图见 `docs/README.md`）
- **目标**
  - **可评估**：能量化“检索是否命中、回答是否可靠”，并沉淀可复现的基准集
  - **可观测**：记录每次检索/生成链路的关键数据，支持 badcase 回溯与成本监控
  - **可持续迭代**：每次改动能自动回归，避免效果回退
- **技术栈（候选）**
  - **评估**：RAGAS / TruLens（或自建黄金集 + 指标）
  - **可观测**：LangSmith / Arize Phoenix（或自建 trace/log）
  - **自动化**：CI 跑单测 + 基准评估 + 报告产出
- **关键知识点**
  - **离线评估 vs 在线反馈**：指标体系、数据闭环与回归策略
  - **成本与延迟预算**：token、调用次数、模型/检索开销的可视化
- **如何启动**
  - 暂未实现（后续会补评估脚本与流水线配置）

---

## 环境要求

- **Python**：建议 3.10+
- **网络**：需要访问你配置的 LLM/Embedding API（或 OpenAI-compatible 端点）

---

## 快速开始（统一启动流程）

### 1) 安装依赖

```bash
python -m venv .venv

# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

# Git Bash / macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) 配置 `.env`

以根目录 `.env.example` 为模板创建 `.env`：

```bash
# Git Bash / macOS / Linux
cp .env.example .env
```

Windows（PowerShell）：

```powershell
Copy-Item .env.example .env
```

至少配置一个 API Key（二选一即可）：

- **`OPENAI_API_KEY`**：OpenAI / OpenAI-compatible（DeepSeek、Qwen、Moonshot 等）
- **`DASHSCOPE_API_KEY`**：阿里云 DashScope（OpenAI 兼容）

常用可选项：

- **`OPENAI_BASE_URL`**：OpenAI-compatible Base URL
- **`MODEL_NAME`**：默认 `gpt-4o`
- **`EMBEDDING_MODEL`**：默认 `text-embedding-3-small`
- **`CHROMA_PERSIST_DIR`**：默认 `./data/chroma_db`

### 3) 准备数据

默认读取 `./data/documents`，支持：`.pdf / .md / .txt / .docx`。

### 4) 启动任一阶段

- Stage 1：`python -m src.stage_1.main --data ./data/documents`
- Stage 2：`python -m src.stage_2.main --data ./data/documents`
- Stage 3：`python -m src.stage_3.main --data ./data/documents`

---

## 启动参数与交互命令

### Stage 1

- **参数**：`--reindex`
- **交互**：`sources`（显示/隐藏来源）、`quit/exit/q`

### Stage 2

- **参数**：`--reindex`、`--no-semantic`、`--no-rerank`
- **交互**：`detail`（详细信息）

### Stage 3

- **参数**：`--reindex`、`--no-semantic`、`--no-routing`、`--no-self-rag`、`--no-tools`、`--no-parent-child`、`--no-compression`
- **交互**：`help`（能力说明）、`detail`（详细信息）

---

## 向量库与缓存说明

- Chroma 默认落盘在 `CHROMA_PERSIST_DIR`（默认 `./data/chroma_db`）
- 不同阶段使用不同集合名：
  - Stage 1：`rag_documents`
  - Stage 2：`advanced_rag`
  - Stage 3：`agentic_rag`

---

## 测试

建议显式设置 `PYTHONPATH=src`：

```bash
# Git Bash / macOS / Linux
PYTHONPATH=src pytest -q
```

Windows（PowerShell）：

```powershell
$env:PYTHONPATH="src"
pytest -q
```

---

## 常见问题（Troubleshooting）

- **启动时报 “OPENAI_API_KEY / DASHSCOPE_API_KEY 未设置”**
  - 确认根目录存在 `.env` 且填写了 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`
- **加载 `.pdf/.docx` 报错**
  - 依赖：`pypdf / unstructured / python-docx / docx2txt`；若遇到系统级依赖问题，可先用 `.md/.txt` 验证主流程
- **Stage 2 重排序首次运行很慢**
  - `sentence-transformers` 会下载 CrossEncoder 模型，属于正常现象

---

## 文档索引

- **路线图总览**：`docs/README.md`
- **阶段总结**：`docs/stage_1/README.md`、`docs/stage_2/README.md`、`docs/stage_3/README.md`

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hefeng6500/UltimateRAG&type=date&legend=top-left)](https://www.star-history.com/#hefeng6500/UltimateRAG&type=date&legend=top-left)
