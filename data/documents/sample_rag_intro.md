# RAG 系统示例文档

## 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的技术。它通过从外部知识库中检索相关信息，然后将这些信息作为上下文提供给大语言模型（LLM），从而生成更准确、更可靠的回答。

## RAG 的工作流程

1. **文档索引**：将文档切分成小块，并转换为向量存储
2. **用户查询**：用户提出问题
3. **相似度检索**：根据问题检索最相关的文档块
4. **上下文增强**：将检索到的内容作为上下文
5. **答案生成**：LLM 基于上下文生成回答

## RAG 的优势

- **减少幻觉**：通过提供真实的参考文档，减少 LLM 编造信息的可能
- **知识可更新**：无需重新训练模型，只需更新知识库
- **可追溯性**：可以追踪答案的来源文档
- **领域特定**：可以针对特定领域构建专业知识库

## 常见的 RAG 技术

### 基础 RAG

最简单的实现方式，直接将检索到的文档块拼接到 Prompt 中。

### Advanced RAG

包含以下优化：
- 语义分块（Semantic Chunking）
- 混合检索（Hybrid Search）
- 重排序（Re-ranking）
- 查询重写（Query Rewrite）

### Agentic RAG

让 LLM 具备自主决策能力：
- 判断是否需要检索
- 自动改写查询
- 调用外部工具
- 多轮自反思

## 技术栈推荐

| 组件 | 推荐工具 |
|------|----------|
| 编排框架 | LangChain、LlamaIndex |
| 向量数据库 | ChromaDB、Pinecone、Milvus |
| Embedding | OpenAI、BGE、Jina |
| LLM | GPT-4、Claude、DeepSeek |

## 参考资料

- LangChain 官方文档
- RAG 最佳实践指南
- 向量数据库对比评测
