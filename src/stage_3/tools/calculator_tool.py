"""
计算器工具

提供数学计算能力，支持复杂表达式和统计计算。
"""

import math
import re
from typing import Optional
from loguru import logger

from .base import BaseTool, ToolResult, ToolStatus


class CalculatorTool(BaseTool):
    """
    计算器工具
    
    安全执行数学表达式，支持基本运算和常用数学函数。
    """
    
    name: str = "calculator"
    description: str = """数学计算器。输入数学表达式进行计算。
支持的操作：
- 基本运算: +, -, *, /, **, %, //
- 数学函数: sqrt, sin, cos, tan, log, exp, abs, round
- 常量: pi, e
示例: "2 + 3 * 4", "sqrt(16)", "sin(pi/2)", "log(100, 10)"
"""
    
    # 允许的函数和常量
    ALLOWED_NAMES = {
        # 数学函数
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "pow": pow,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        # 常量
        "pi": math.pi,
        "e": math.e,
        # 聚合函数
        "sum": sum,
        "min": min,
        "max": max,
        "len": len,
    }
    
    def __init__(self):
        """初始化计算器"""
        logger.info("🔢 计算器工具初始化完成")
    
    def _sanitize_expression(self, expr: str) -> str:
        """
        清理和验证表达式
        
        Args:
            expr: 原始表达式
            
        Returns:
            str: 清理后的表达式
        """
        # 移除可能的危险字符
        expr = re.sub(r'[;`]', '', expr)
        
        # 提取数学表达式
        # 支持中文数字描述
        chinese_to_expr = {
            "加": "+",
            "减": "-",
            "乘": "*",
            "除": "/",
            "的": "**",
            "次方": "",
            "平方": "**2",
            "立方": "**3",
            "开根号": "sqrt",
        }
        
        for cn, en in chinese_to_expr.items():
            expr = expr.replace(cn, en)
        
        return expr.strip()
    
    def run(self, input: str) -> ToolResult:
        """
        执行计算
        
        Args:
            input: 数学表达式
            
        Returns:
            ToolResult: 计算结果
        """
        try:
            # 清理表达式
            expr = self._sanitize_expression(input)
            
            if not expr:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error="请提供有效的数学表达式"
                )
            
            logger.info(f"🔢 执行计算: {expr}")
            
            # 安全评估表达式
            # 只允许特定的函数和运算符
            result = eval(
                expr,
                {"__builtins__": {}},
                self.ALLOWED_NAMES
            )
            
            # 格式化结果
            if isinstance(result, float):
                # 处理浮点数精度
                if result == int(result):
                    result = int(result)
                else:
                    result = round(result, 10)
            
            output = f"计算结果: {expr} = {result}"
            
            logger.info(f"✅ 计算完成: {result}")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "expression": expr,
                    "result": result
                }
            )
            
        except SyntaxError:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"表达式语法错误: {input}"
            )
        except NameError as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"不支持的函数或变量: {e}"
            )
        except ZeroDivisionError:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="除数不能为零"
            )
        except Exception as e:
            logger.error(f"❌ 计算失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"计算错误: {e}"
            )


class StatisticsCalculator(BaseTool):
    """
    统计计算器
    
    提供基本统计计算功能。
    """
    
    name: str = "statistics_calculator"
    description: str = """统计计算器。对一组数字进行统计分析。
输入格式: 逗号分隔的数字，如 "1, 2, 3, 4, 5"
返回: 计数、总和、平均值、最大/最小值、标准差等"""
    
    def __init__(self):
        """初始化统计计算器"""
        logger.info("📊 统计计算器初始化完成")
    
    def run(self, input: str) -> ToolResult:
        """
        执行统计计算
        
        Args:
            input: 逗号分隔的数字
            
        Returns:
            ToolResult: 统计结果
        """
        try:
            # 解析数字
            numbers = []
            for part in input.replace(" ", "").split(","):
                try:
                    numbers.append(float(part))
                except ValueError:
                    continue
            
            if not numbers:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error="请提供有效的数字列表（逗号分隔）"
                )
            
            logger.info(f"📊 执行统计计算: {len(numbers)} 个数字")
            
            # 计算统计量
            count = len(numbers)
            total = sum(numbers)
            mean = total / count
            sorted_nums = sorted(numbers)
            median = sorted_nums[count // 2] if count % 2 == 1 else \
                     (sorted_nums[count // 2 - 1] + sorted_nums[count // 2]) / 2
            min_val = min(numbers)
            max_val = max(numbers)
            range_val = max_val - min_val
            
            # 标准差
            variance = sum((x - mean) ** 2 for x in numbers) / count
            std_dev = math.sqrt(variance)
            
            output = f"""统计结果:
- 数据个数: {count}
- 总和: {total}
- 平均值: {mean:.4f}
- 中位数: {median:.4f}
- 最小值: {min_val}
- 最大值: {max_val}
- 范围: {range_val}
- 标准差: {std_dev:.4f}"""
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "count": count,
                    "sum": total,
                    "mean": mean,
                    "median": median,
                    "min": min_val,
                    "max": max_val,
                    "std": std_dev
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 统计计算失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"统计计算错误: {e}"
            )

