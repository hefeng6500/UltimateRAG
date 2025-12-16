"""
工具基类

定义统一的工具接口和结果格式。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class ToolStatus(str, Enum):
    """工具执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


@dataclass
class ToolResult:
    """
    工具执行结果
    
    Attributes:
        status: 执行状态
        output: 执行输出
        error: 错误信息（如果有）
        metadata: 额外元数据
    """
    status: ToolStatus
    output: str
    error: Optional[str] = None
    metadata: Optional[dict] = None
    
    @property
    def is_success(self) -> bool:
        """是否执行成功"""
        return self.status == ToolStatus.SUCCESS
    
    def to_context(self) -> str:
        """转换为上下文字符串"""
        if self.is_success:
            return f"工具执行结果:\n{self.output}"
        else:
            return f"工具执行失败: {self.error or '未知错误'}"


class BaseTool(ABC):
    """
    工具基类
    
    所有工具都需要继承此类并实现 run 方法。
    """
    
    name: str = "base_tool"
    description: str = "基础工具"
    
    @abstractmethod
    def run(self, input: str) -> ToolResult:
        """
        执行工具
        
        Args:
            input: 工具输入
            
        Returns:
            ToolResult: 执行结果
        """
        pass
    
    def __call__(self, input: str) -> ToolResult:
        """允许直接调用"""
        return self.run(input)
    
    def to_langchain_tool(self):
        """
        转换为 LangChain Tool 格式
        
        Returns:
            LangChain Tool 实例
        """
        from langchain_core.tools import tool
        
        @tool(name=self.name, description=self.description)
        def _tool_func(input: str) -> str:
            result = self.run(input)
            return result.output if result.is_success else f"Error: {result.error}"
        
        return _tool_func

