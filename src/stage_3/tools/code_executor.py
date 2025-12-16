"""
代码执行工具

提供安全的 Python 代码执行能力。
使用受限环境防止恶意代码执行。
"""

import sys
import io
import traceback
from typing import Optional
from contextlib import contextmanager
import signal
from loguru import logger

from .base import BaseTool, ToolResult, ToolStatus


@contextmanager
def timeout_handler(seconds: int):
    """
    超时处理上下文管理器
    
    Args:
        seconds: 超时秒数
    """
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"代码执行超时 ({seconds}秒)")
    
    # 设置信号处理器（仅 Unix 系统有效）
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows 不支持 SIGALRM，直接执行
        yield


class CodeExecutorTool(BaseTool):
    """
    代码执行工具
    
    在受限环境中执行 Python 代码。
    """
    
    name: str = "code_executor"
    description: str = """Python 代码执行器。执行 Python 代码并返回结果。
适用场景：
- 数据处理和分析
- 算法演示
- 数学计算
注意：代码在受限环境中执行，无法访问文件系统或网络。
输入: Python 代码字符串
输出: 代码执行结果或 print 输出"""
    
    # 受限的内置函数
    SAFE_BUILTINS = {
        # 基本类型
        "True": True,
        "False": False,
        "None": None,
        # 类型转换
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        # 基本函数
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "pow": pow,
        "print": print,
        # 类型检查
        "isinstance": isinstance,
        "type": type,
    }
    
    # 允许导入的模块
    ALLOWED_MODULES = {
        "math",
        "random",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "statistics",
        "json",
        "re",
    }
    
    def __init__(self, timeout: int = 10):
        """
        初始化代码执行器
        
        Args:
            timeout: 执行超时时间（秒）
        """
        self.timeout = timeout
        logger.info(f"💻 代码执行工具初始化完成 (timeout={timeout}s)")
    
    def _create_safe_globals(self) -> dict:
        """
        创建安全的全局命名空间
        
        Returns:
            dict: 安全的全局变量字典
        """
        safe_globals = {"__builtins__": self.SAFE_BUILTINS.copy()}
        
        # 添加允许的模块
        import math
        import random
        import datetime
        import collections
        import itertools
        import functools
        import statistics
        import json
        import re
        
        safe_globals.update({
            "math": math,
            "random": random,
            "datetime": datetime,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "statistics": statistics,
            "json": json,
            "re": re,
        })
        
        return safe_globals
    
    def run(self, input: str) -> ToolResult:
        """
        执行代码
        
        Args:
            input: Python 代码
            
        Returns:
            ToolResult: 执行结果
        """
        if not input.strip():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="请提供要执行的代码"
            )
        
        logger.info(f"💻 执行代码: {input[:50]}...")
        
        # 捕获标准输出
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # 创建安全环境
            safe_globals = self._create_safe_globals()
            local_vars = {}
            
            # 执行代码（带超时）
            with timeout_handler(self.timeout):
                exec(input, safe_globals, local_vars)
            
            # 获取输出
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # 构建结果
            output_parts = []
            
            if stdout_output:
                output_parts.append(f"输出:\n{stdout_output}")
            
            if stderr_output:
                output_parts.append(f"警告/错误:\n{stderr_output}")
            
            # 如果没有 print 输出，尝试获取最后一个表达式的值
            if not output_parts:
                # 尝试 eval 最后一行
                lines = input.strip().split('\n')
                last_line = lines[-1].strip()
                if last_line and not last_line.startswith(('import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', '#')):
                    try:
                        result = eval(last_line, safe_globals, local_vars)
                        if result is not None:
                            output_parts.append(f"结果: {result}")
                    except:
                        pass
            
            if not output_parts:
                output_parts.append("代码执行成功（无输出）")
            
            output = "\n".join(output_parts)
            
            logger.info("✅ 代码执行完成")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "code": input,
                    "variables": {k: str(v)[:100] for k, v in local_vars.items() if not k.startswith('_')}
                }
            )
            
        except TimeoutError as e:
            logger.warning(f"⏱️ 代码执行超时")
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                output="",
                error=str(e)
            )
        except SyntaxError as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"语法错误: {e}"
            )
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"❌ 代码执行失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"执行错误: {e}\n{error_msg}"
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class SafeREPL(BaseTool):
    """
    安全的 REPL 工具
    
    提供单行表达式求值功能。
    """
    
    name: str = "python_repl"
    description: str = "Python 表达式求值器。输入 Python 表达式，返回计算结果。适用于简单计算。"
    
    def __init__(self):
        """初始化 REPL"""
        self._executor = CodeExecutorTool(timeout=5)
        logger.info("🐍 Python REPL 初始化完成")
    
    def run(self, input: str) -> ToolResult:
        """
        执行表达式
        
        Args:
            input: Python 表达式
            
        Returns:
            ToolResult: 求值结果
        """
        # 包装为 print 语句
        code = f"print({input})"
        return self._executor.run(code)

