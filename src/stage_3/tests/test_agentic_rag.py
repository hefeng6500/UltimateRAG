"""
Phase 3: Agentic RAG 单元测试

测试各个组件的功能：
- 工具模块
- 路由器
- 父子索引
- 上下文压缩

注意：这些测试独立运行，不依赖完整的 RAG 链
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import math
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestCalculatorTool(unittest.TestCase):
    """计算器工具测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        from src.stage_3.tools.calculator_tool import CalculatorTool
        cls.CalculatorTool = CalculatorTool
    
    def setUp(self):
        self.calculator = self.CalculatorTool()
    
    def test_basic_arithmetic(self):
        """测试基本算术运算"""
        from src.stage_3.tools.base import ToolStatus
        
        # 加法
        result = self.calculator.run("2 + 3")
        self.assertTrue(result.is_success)
        self.assertIn("5", result.output)
        
        # 乘法
        result = self.calculator.run("4 * 5")
        self.assertTrue(result.is_success)
        self.assertIn("20", result.output)
        
        # 除法
        result = self.calculator.run("10 / 2")
        self.assertTrue(result.is_success)
        self.assertIn("5", result.output)
    
    def test_math_functions(self):
        """测试数学函数"""
        # 平方根
        result = self.calculator.run("sqrt(16)")
        self.assertTrue(result.is_success)
        self.assertIn("4", result.output)
        
        # 幂运算
        result = self.calculator.run("pow(2, 10)")
        self.assertTrue(result.is_success)
        self.assertIn("1024", result.output)
    
    def test_constants(self):
        """测试数学常量"""
        result = self.calculator.run("pi")
        self.assertTrue(result.is_success)
        self.assertIn("3.14", result.output)
    
    def test_division_by_zero(self):
        """测试除零错误"""
        from src.stage_3.tools.base import ToolStatus
        
        result = self.calculator.run("1 / 0")
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, ToolStatus.ERROR)
    
    def test_syntax_error(self):
        """测试语法错误"""
        from src.stage_3.tools.base import ToolStatus
        
        result = self.calculator.run("2 +")
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, ToolStatus.ERROR)


class TestStatisticsCalculator(unittest.TestCase):
    """统计计算器测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        from src.stage_3.tools.calculator_tool import StatisticsCalculator
        cls.StatisticsCalculator = StatisticsCalculator
    
    def setUp(self):
        self.stats_calc = self.StatisticsCalculator()
    
    def test_basic_statistics(self):
        """测试基本统计"""
        result = self.stats_calc.run("1, 2, 3, 4, 5")
        self.assertTrue(result.is_success)
        
        # 检查输出包含各项统计量
        self.assertIn("平均值", result.output)
        self.assertIn("中位数", result.output)
        self.assertIn("最小值", result.output)
        self.assertIn("最大值", result.output)
        
        # 检查元数据
        self.assertEqual(result.metadata["count"], 5)
        self.assertEqual(result.metadata["sum"], 15)
        self.assertEqual(result.metadata["mean"], 3.0)
    
    def test_empty_input(self):
        """测试空输入"""
        result = self.stats_calc.run("")
        self.assertFalse(result.is_success)


class TestCodeExecutor(unittest.TestCase):
    """代码执行器测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        from src.stage_3.tools.code_executor import CodeExecutorTool
        cls.CodeExecutorTool = CodeExecutorTool
    
    def setUp(self):
        self.executor = self.CodeExecutorTool(timeout=5)
    
    def test_simple_print(self):
        """测试简单打印"""
        result = self.executor.run("print('Hello, World!')")
        self.assertTrue(result.is_success)
        self.assertIn("Hello, World!", result.output)
    
    def test_math_calculation(self):
        """测试数学计算"""
        result = self.executor.run("print(sum(range(1, 11)))")
        self.assertTrue(result.is_success)
        self.assertIn("55", result.output)
    
    def test_list_comprehension(self):
        """测试列表推导式"""
        result = self.executor.run("print([x**2 for x in range(5)])")
        self.assertTrue(result.is_success)
        self.assertIn("[0, 1, 4, 9, 16]", result.output)
    
    def test_import_allowed_module(self):
        """测试导入允许的模块"""
        code = """
import math
print(math.sqrt(16))
"""
        result = self.executor.run(code)
        self.assertTrue(result.is_success)
        self.assertIn("4", result.output)
    
    def test_syntax_error(self):
        """测试语法错误"""
        from src.stage_3.tools.base import ToolStatus
        
        result = self.executor.run("print(")
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, ToolStatus.ERROR)


class TestToolResult(unittest.TestCase):
    """工具结果测试"""
    
    def test_success_result(self):
        """测试成功结果"""
        from src.stage_3.tools.base import ToolResult, ToolStatus
        
        result = ToolResult(
            status=ToolStatus.SUCCESS,
            output="测试输出",
            metadata={"key": "value"}
        )
        
        self.assertTrue(result.is_success)
        self.assertEqual(result.output, "测试输出")
        self.assertIn("测试输出", result.to_context())
    
    def test_error_result(self):
        """测试错误结果"""
        from src.stage_3.tools.base import ToolResult, ToolStatus
        
        result = ToolResult(
            status=ToolStatus.ERROR,
            output="",
            error="测试错误"
        )
        
        self.assertFalse(result.is_success)
        self.assertIn("失败", result.to_context())


class TestKeywordRouter(unittest.TestCase):
    """关键词路由器测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        from src.stage_3.router import KeywordRouter, RouteType
        cls.KeywordRouter = KeywordRouter
        cls.RouteType = RouteType
    
    def setUp(self):
        self.router = self.KeywordRouter()
    
    def test_calculator_routing(self):
        """测试计算器路由"""
        route, confidence = self.router.route("请帮我计算 123 + 456")
        self.assertEqual(route, self.RouteType.CALCULATOR)
        
        route, confidence = self.router.route("2 加 3 等于多少？")
        self.assertEqual(route, self.RouteType.CALCULATOR)
    
    def test_web_search_routing(self):
        """测试 Web 搜索路由"""
        route, confidence = self.router.route("今天的天气怎么样？")
        self.assertEqual(route, self.RouteType.WEB_SEARCH)
        
        route, confidence = self.router.route("最新的科技新闻有哪些？")
        self.assertEqual(route, self.RouteType.WEB_SEARCH)
    
    def test_direct_answer_routing(self):
        """测试直接回答路由"""
        route, confidence = self.router.route("你好，你是谁？")
        self.assertEqual(route, self.RouteType.DIRECT_ANSWER)
    
    def test_default_routing(self):
        """测试默认路由（知识库）"""
        route, confidence = self.router.route("什么是 RAG 技术？")
        self.assertEqual(route, self.RouteType.KNOWLEDGE_BASE)


class TestKeywordCompressor(unittest.TestCase):
    """关键词压缩器测试"""
    
    def test_keyword_extraction(self):
        """测试关键词提取"""
        from src.stage_3.context_compressor import KeywordBasedCompressor
        
        with patch('src.stage_3.context_compressor.get_stage3_config'):
            compressor = KeywordBasedCompressor()
            
            # 测试中文关键词提取
            keywords = compressor._extract_keywords("什么是检索增强生成技术")
            self.assertIn("检索", keywords)
            self.assertIn("增强", keywords)
            self.assertIn("生成", keywords)
            
            # 测试英文关键词提取
            keywords = compressor._extract_keywords("What is RAG technology")
            self.assertIn("rag", keywords)
            self.assertIn("technology", keywords)
    
    def test_sentence_splitting(self):
        """测试句子分割"""
        from src.stage_3.context_compressor import KeywordBasedCompressor
        
        with patch('src.stage_3.context_compressor.get_stage3_config'):
            compressor = KeywordBasedCompressor()
            
            # 测试中文句子分割
            sentences = compressor._split_sentences("第一句话。第二句话！第三句话？")
            self.assertEqual(len(sentences), 3)
            
            # 测试英文句子分割
            sentences = compressor._split_sentences("First sentence. Second sentence!")
            self.assertEqual(len(sentences), 2)


class TestRouteType(unittest.TestCase):
    """路由类型枚举测试"""
    
    def test_route_types_exist(self):
        """测试路由类型存在"""
        from src.stage_3.router import RouteType
        
        self.assertEqual(RouteType.KNOWLEDGE_BASE.value, "knowledge_base")
        self.assertEqual(RouteType.WEB_SEARCH.value, "web_search")
        self.assertEqual(RouteType.CALCULATOR.value, "calculator")
        self.assertEqual(RouteType.CODE_EXECUTION.value, "code_execution")
        self.assertEqual(RouteType.DIRECT_ANSWER.value, "direct_answer")


if __name__ == "__main__":
    unittest.main()
