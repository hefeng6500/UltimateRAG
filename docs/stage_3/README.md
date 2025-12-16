# Phase 3: 架构进化 (Modular & Agentic RAG) - 学习总结

## 📚 本阶段学习收获

> [!TIP]
> Phase 3 实现了让 RAG 系统具备"思考"能力的核心组件，系统不再是简单的"检索-生成"直线流程，而是一个能够动态决策的智能网络。

---

## 🎯 完成的功能

### 1. 智能路由器 (`router.py`)
- 使用 LLM 结构化输出进行问题分类
- 支持 5 种路由类型：知识库、Web 搜索、计算器、代码执行、直接回答
- 包含置信度评估和回退策略
- 提供基于关键词的快速路由备选方案

### 2. 自反思 RAG (`self_rag.py`)
- 检索相关性评估（Relevance Evaluation）
- 答案质量评估（Quality Evaluation）
- 自动迭代优化（最多 N 轮）
- 查询优化与重检索

### 3. 工具集成 (`tools/`)
- **Web 搜索工具**：使用 DuckDuckGo 免费搜索
- **计算器工具**：安全的数学表达式求值
- **代码执行器**：受限沙箱中的 Python 执行
- **统计计算器**：数据统计分析

### 4. 父子索引检索器 (`parent_child_retriever.py`)
- 大块（Parent）存储完整上下文
- 小块（Child）用于精准匹配
- 支持上下文窗口扩展
- 内存映射存储

### 5. 上下文压缩 (`context_compressor.py`)
- LLM 驱动的相关句子提取
- 基于关键词的快速压缩
- 压缩比统计与优化

---

## 💡 技术要点

### 1. LangChain 结构化输出

使用 Pydantic 模型定义输出格式，LLM 返回结构化数据：

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteDecision(BaseModel):
    route_type: RouteType = Field(description="路由类型")
    confidence: float = Field(description="置信度", ge=0.0, le=1.0)
    reasoning: str = Field(description="决策原因")

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(RouteDecision)
result = structured_llm.invoke(prompt)  # 返回 RouteDecision 实例
```

### 2. Self-RAG 工作流程

```
用户问题 → 检索文档 → 相关性评估 → 生成答案 → 质量评估
                ↑                                    ↓
                └─────── 质量不足时重新检索 ←────────┘
```

### 3. 父子索引原理

```
原始文档
    ↓
切分为大的父块（2000 字符）
    ↓
每个父块再切分为小的子块（400 字符）
    ↓
子块存入向量库，父块存入内存映射

检索时：用子块精准匹配 → 返回对应父块的完整上下文
```

### 4. 安全的代码执行

```python
# 受限的内置函数白名单
SAFE_BUILTINS = {
    "int": int, "float": float, "str": str,
    "len": len, "range": range, "sum": sum,
    "min": min, "max": max, "abs": abs,
    # ... 更多安全函数
}

# 允许的模块白名单
ALLOWED_MODULES = {"math", "random", "datetime", "statistics"}

# 执行代码
exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)
```

---

## 📊 测试结果

```
✅ 8 个测试类全部通过
- TestCalculatorTool: 计算器测试
- TestStatisticsCalculator: 统计计算测试
- TestCodeExecutor: 代码执行测试
- TestKeywordRouter: 路由器测试
- TestToolResult: 工具结果测试
- TestParentChildRetriever: 父子索引测试
- TestContextCompressor: 上下文压缩测试
```

---

## 🔑 关键技术点

| 技术 | 解决的问题 | 实现难度 | ROI |
|------|-----------|---------|-----|
| 智能路由 | 问题分类错误 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Self-RAG | 答案质量不稳定 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 工具调用 | 无法处理特定任务 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 父子索引 | 检索精度与上下文的平衡 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 上下文压缩 | Token 浪费 | ⭐⭐ | ⭐⭐⭐ |

---

## 🔗 文件结构

```
src/stage_3/
├── __init__.py                  # 包初始化
├── config.py                    # Stage 3 配置
├── router.py                    # 智能路由器
├── self_rag.py                  # 自反思 RAG
├── tools/
│   ├── __init__.py              # 工具包
│   ├── base.py                  # 工具基类
│   ├── search_tool.py           # Web 搜索
│   ├── calculator_tool.py       # 计算器
│   └── code_executor.py         # 代码执行
├── parent_child_retriever.py    # 父子索引
├── context_compressor.py        # 上下文压缩
├── agentic_rag_chain.py         # Agentic RAG 链
├── main.py                      # 主入口
└── tests/
    └── test_agentic_rag.py      # 单元测试
```

---

## 🚀 使用方法

```bash
# 运行 Agentic RAG 系统
cd /path/to/UltimateRAG
source .venv/bin/activate
python -m src.stage_3.main --data ./data/documents

# 可选参数
--no-routing       # 禁用智能路由
--no-self-rag      # 禁用自反思
--no-tools         # 禁用工具调用
--no-parent-child  # 禁用父子索引
--no-compression   # 禁用上下文压缩
--reindex          # 强制重新索引
```

---

## 📈 Phase 3 vs Phase 2 对比

| 特性 | Phase 2 | Phase 3 |
|------|---------|---------|
| 检索方式 | 混合检索 | 智能路由 + 多策略 |
| 答案质量 | 一次性生成 | 自反思迭代优化 |
| 处理能力 | 仅文档问答 | 文档 + 搜索 + 计算 + 代码 |
| 上下文 | 检索即返回 | 父子索引 + 压缩优化 |
| 智能程度 | 被动响应 | 主动决策 |

---

## ⚠️ 注意事项

### 1. 性能考量
- 智能路由会增加一次 LLM 调用
- Self-RAG 迭代会显著增加响应时间
- 建议根据场景选择性启用功能

### 2. 安全考量
- 代码执行器使用沙箱环境
- 已禁用危险函数和模块
- 设置执行超时防止死循环

### 3. 成本考量
- 每个功能都会消耗额外 Token
- 建议监控 Token 使用量
- 可通过禁用不需要的功能控制成本

---

## 📖 参考资料

- [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Self-RAG 论文](https://arxiv.org/abs/2310.11511)
- [LangGraph Workflows](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Parent Document Retriever](https://docs.langchain.com/oss/python/langchain/retrieval)

---

## 🎓 核心概念总结

### Agentic RAG 的本质

传统 RAG 是 **Pipeline**（流水线）：

```
Query → Retrieve → Generate → Answer
```

Agentic RAG 是 **Agent**（智能体）：

```
                    ┌─→ 知识库检索 ─┐
                    │              │
Query → 路由决策 ──→├─→ Web 搜索 ──┼─→ 质量评估 ─→ 迭代优化 ─→ Answer
                    │              │
                    ├─→ 工具调用 ──┤
                    │              │
                    └─→ 直接回答 ──┘
```

### 关键洞察

1. **智能路由是入口**：好的分类决定了后续处理的效率和质量
2. **自反思是保障**：让系统能够自我纠错，提高答案可靠性
3. **工具是扩展**：突破纯文本问答的局限
4. **父子索引是平衡**：在检索精度和上下文完整性之间找到平衡点
5. **压缩是优化**：节省 Token，聚焦关键信息

---

## 🔮 Phase 4 展望

下一阶段将进入更高级的领域：

- **GraphRAG**：知识图谱增强检索
- **领域微调**：Embedding 和 LLM 的定制化
- **多模态**：图片、表格等非文本数据处理

> [!NOTE]
> Phase 3 的组件化设计为后续扩展奠定了良好基础。

