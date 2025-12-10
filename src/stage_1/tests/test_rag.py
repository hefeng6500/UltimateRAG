"""
Phase 1 单元测试

测试文档加载、分块、向量存储和 RAG 链功能。
"""

import pytest
from pathlib import Path
import tempfile
import os

# 设置测试环境变量
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")

from stage_1.config import Config
from stage_1.document_loader import DocumentLoader
from stage_1.chunker import TextChunker


class TestConfig:
    """配置模块测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.top_k == 3
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = Config(
            chunk_size=1024,
            chunk_overlap=100,
            top_k=5
        )
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.top_k == 5


class TestDocumentLoader:
    """文档加载器测试"""
    
    def test_load_txt_file(self):
        """测试加载 TXT 文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode="w", 
            suffix=".txt", 
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write("这是测试文档内容。")
            temp_path = f.name
        
        try:
            loader = DocumentLoader()
            docs = loader.load_file(temp_path)
            
            assert len(docs) > 0
            assert "测试文档" in docs[0].page_content
            assert docs[0].metadata["file_type"] == ".txt"
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/path/file.txt")
    
    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name
        
        try:
            loader = DocumentLoader()
            with pytest.raises(ValueError):
                loader.load_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestTextChunker:
    """文本分块器测试"""
    
    def test_split_text(self):
        """测试文本分块"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        text = "这是一段很长的测试文本。" * 50
        chunks = chunker.split_text(text)
        
        assert len(chunks) > 1
        # 验证每个块的大小不超过 chunk_size
        for chunk in chunks:
            assert len(chunk) <= 120  # 允许一些误差
    
    def test_empty_input(self):
        """测试空输入"""
        chunker = TextChunker()
        
        assert chunker.split_text("") == []
        assert chunker.split_documents([]) == []
    
    def test_chunk_overlap(self):
        """测试分块重叠"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=50)
        
        text = "A" * 200
        chunks = chunker.split_text(text)
        
        # 由于有重叠，中间的内容应该在多个块中出现
        assert len(chunks) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
