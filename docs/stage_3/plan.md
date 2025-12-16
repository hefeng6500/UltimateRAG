# Phase 3: 架构进化 (Modular & Agentic RAG) - 开发计划

## 📋 目标
让系统具备"思考"能力，处理复杂任务（不仅仅是找文档）。

**核心理念：** RAG 不再是一条直线，而是一个动态的网络。

---

## 🏗️ 新增技术

| 组件 | 技术 | 说明 |
|------|------|------|
| **路由机制** | Query Router | 根据问题类型智能分发到不同处理器 |
| **自反思 RAG** | Self-RAG | LLM 自评估答案质量，必要时重新检索 |
| **工具调用** | Tool Use | 集成搜索、计算器、代码执行等工具 |
| **父子索引** | Parent-Child Indexing | 检索子块、返回父块，兼顾精度和上下文 |
| **上下文压缩** | Context Compression | 精简检索内容，只保留相关信息 |

---

## 📁 文件结构

```
src/stage_3/
├── __init__.py                  # 包初始化
├── config.py                    # Stage 3 扩展配置
├── router.py                    # 智能路由器
├── self_rag.py                  # 自反思 RAG 实现
├── tools/
│   ├── __init__.py              # 工具包初始化
│   ├── base.py                  # 工具基类
│   ├── search_tool.py           # Web 搜索工具
│   ├── calculator_tool.py       # 计算器工具
│   └── code_executor.py         # 代码执行工具
├── parent_child_retriever.py    # 父子索引检索器
├── context_compressor.py        # 上下文压缩器
├── agentic_rag_chain.py         # 代理式 RAG 链
├── main.py                      # 主入口
└── tests/
    └── test_agentic_rag.py      # 单元测试
```

---

## 🔧 实现步骤

### Step 1: 智能路由器 (Query Router)
- 使用 LLM 进行问题分类
- 支持多种路由策略：
  - 技术问题 → 知识库检索
  - 计算问题 → 计算器工具
  - 代码问题 → 代码执行器
  - 通用闲聊 → 直接 LLM 回答
  - 实时信息 → Web 搜索

### Step 2: Self-RAG (自反思 RAG)
- 实现答案质量自评估
- 支持多轮检索增强
- 实现检索相关性判断
- 实现答案完整性评分

### Step 3: 工具集成 (Tool Use)
- 定义统一的工具接口
- 实现 Web 搜索工具
- 实现计算器工具
- 实现代码执行沙箱

### Step 4: 父子索引 (Parent-Child Retriever)
- 实现父文档存储
- 实现子块精准检索
- 实现父块上下文返回
- 支持动态窗口扩展

### Step 5: 上下文压缩 (Context Compression)
- 基于 LLM 的内容筛选
- 基于关键词的快速过滤
- 实现相关句子提取

### Step 6: 代理式 RAG 链 (Agentic RAG Chain)
- 使用 LangGraph 构建有状态工作流
- 集成所有组件
- 实现动态决策流程

### Step 7: 测试与优化
- 编写单元测试
- 性能基准测试
- 对比 Phase 2 效果

---

## 🎯 技术选型

| 功能 | 技术方案 | 说明 |
|------|---------|------|
| Agent 框架 | LangGraph | LangChain 官方推荐的 Agent 框架 |
| 工具定义 | @tool 装饰器 | LangChain 标准工具定义方式 |
| 路由分类 | Structured Output | 使用 LLM 结构化输出进行分类 |
| 状态管理 | TypedDict | Python 类型安全的状态定义 |

---

## ✅ 预期成果

- 系统变"聪明"了，懂得拒绝回答（如果没搜到）
- 懂得自己去网上找补充信息
- 能处理多步推理问题
- 能调用工具解决特定问题
- 检索更精准，上下文更完整

