"""
Phase 2 单元测试

测试语义分块、混合检索、查询重写、重排序等高级功能。
"""

import pytest
import os
import sys

# 设置测试环境变量
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")

sys.path.insert(0, str(__file__).replace("/src/stage_2/tests/test_advanced_rag.py", "/src"))

from stage_2.metadata_extractor import MetadataExtractor
from langchain_core.documents import Document


class TestMetadataExtractor:
    """元数据提取器测试"""
    
    def test_extract_from_content(self):
        """测试从内容提取元数据"""
        extractor = MetadataExtractor()
        
        content = """# 文档标题
        
## 第一章节

这是 2024-01-15 记录的内容。
包含了一些重要信息。
"""
        metadata = extractor.extract_from_content(content)
        
        assert "headers" in metadata
        assert "文档标题" in metadata["headers"]
        assert "extracted_dates" in metadata
        assert metadata["char_count"] > 0
    
    def test_extract_dates(self):
        """测试日期提取"""
        extractor = MetadataExtractor()
        
        content = "这是 2024-01-15 和 2024年2月20日 的记录。"
        dates = extractor._extract_dates(content)
        
        assert len(dates) >= 2
    
    def test_enrich_documents(self):
        """测试文档元数据增强"""
        extractor = MetadataExtractor()
        
        docs = [
            Document(
                page_content="# 测试文档\n\n这是测试内容。",
                metadata={"source_file": "/path/to/test.md"}
            )
        ]
        
        enriched = extractor.enrich_documents(docs)
        
        assert len(enriched) == 1
        assert "file_name" in enriched[0].metadata
        assert "headers" in enriched[0].metadata


class TestHybridRetriever:
    """混合检索器测试"""
    
    def test_bm25_tokenize(self):
        """测试分词功能"""
        from stage_2.hybrid_retriever import HybridRetriever
        
        # 创建一个实例但不初始化完整
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._bm25 = None
        retriever._documents = []
        
        # 测试中英文混合分词
        tokens = retriever._tokenize("Hello 世界 Test 测试")
        
        assert len(tokens) > 0
        assert "hello" in tokens
        assert "test" in tokens


class TestQueryRewriter:
    """查询重写器测试（不需要 API）"""
    
    def test_expand_query(self):
        """测试查询扩展"""
        from stage_2.query_rewriter import QueryRewriter
        
        # 创建实例但跳过 LLM 初始化
        rewriter = QueryRewriter.__new__(QueryRewriter)
        rewriter.config = None
        rewriter._llm = None
        
        # 测试扩展功能
        expanded = rewriter.expand_query("什么是 RAG？")
        
        assert "RAG" in expanded
        assert len(expanded) > len("什么是 RAG？")


class TestReranker:
    """重排序器测试"""
    
    def test_simple_reranker(self):
        """测试简单重排序器"""
        from stage_2.reranker import SimpleReranker
        
        reranker = SimpleReranker.__new__(SimpleReranker)
        reranker.config = type('Config', (), {'top_k': 3})()
        
        docs = [
            Document(page_content="Python 是一种编程语言"),
            Document(page_content="RAG 是检索增强生成"),
            Document(page_content="Python 和 RAG 结合使用"),
        ]
        
        results = reranker.rerank("Python RAG", docs)
        
        assert len(results) == 3
        # Python 和 RAG 结合应该排在前面
        top_content = results[0][0].page_content
        assert "Python" in top_content and "RAG" in top_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
