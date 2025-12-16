"""
Web 搜索工具

提供实时互联网搜索能力。
支持多种搜索后端：DuckDuckGo、Tavily 等。
"""

from typing import Optional, List
from loguru import logger

from .base import BaseTool, ToolResult, ToolStatus


class WebSearchTool(BaseTool):
    """
    Web 搜索工具
    
    使用 DuckDuckGo 进行网络搜索（无需 API Key）。
    """
    
    name: str = "web_search"
    description: str = "搜索互联网获取最新信息。输入搜索关键词，返回相关网页摘要。适用于需要实时信息的问题，如新闻、天气、股价等。"
    
    def __init__(self, max_results: int = 5):
        """
        初始化搜索工具
        
        Args:
            max_results: 最大返回结果数
        """
        self.max_results = max_results
        self._search_client = None
        logger.info(f"🔍 Web 搜索工具初始化完成 (max_results={max_results})")
    
    def _get_client(self):
        """延迟加载搜索客户端"""
        if self._search_client is None:
            try:
                from ddgs import DDGS
                self._search_client = DDGS()
            except ImportError:
                logger.warning("⚠️ duckduckgo_search 未安装，Web 搜索不可用")
                return None
        return self._search_client
    
    def run(self, input: str) -> ToolResult:
        """
        执行搜索
        
        Args:
            input: 搜索关键词
            
        Returns:
            ToolResult: 搜索结果
        """
        client = self._get_client()
        
        if client is None:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                output="",
                error="Web 搜索功能不可用，请安装 duckduckgo_search: pip install duckduckgo_search"
            )
        
        try:
            logger.info(f"🔍 执行搜索: {input}")
            
            # 执行搜索
            results = list(client.text(
                input,
                max_results=self.max_results
            ))
            
            if not results:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output="未找到相关搜索结果。",
                    metadata={"query": input, "count": 0}
                )
            
            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                body = result.get("body", "无摘要")
                href = result.get("href", "")
                
                formatted_results.append(
                    f"[{i}] {title}\n"
                    f"    {body}\n"
                    f"    链接: {href}"
                )
            
            output = "\n\n".join(formatted_results)
            
            logger.info(f"✅ 搜索完成: 找到 {len(results)} 个结果")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "query": input,
                    "count": len(results),
                    "sources": [r.get("href", "") for r in results]
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e)
            )


class TavilySearchTool(BaseTool):
    """
    Tavily 搜索工具
    
    使用 Tavily API 进行搜索（需要 API Key）。
    Tavily 专为 AI 应用优化，返回结果更适合 LLM 处理。
    """
    
    name: str = "tavily_search"
    description: str = "使用 Tavily 搜索互联网。提供专为 AI 优化的搜索结果。"
    
    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        """
        初始化 Tavily 搜索工具
        
        Args:
            api_key: Tavily API Key
            max_results: 最大返回结果数
        """
        self.api_key = api_key
        self.max_results = max_results
        self._client = None
        
        if api_key:
            logger.info("🔍 Tavily 搜索工具初始化完成")
        else:
            logger.warning("⚠️ Tavily API Key 未配置")
    
    def run(self, input: str) -> ToolResult:
        """
        执行搜索
        
        Args:
            input: 搜索关键词
            
        Returns:
            ToolResult: 搜索结果
        """
        if not self.api_key:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                output="",
                error="Tavily API Key 未配置"
            )
        
        try:
            from tavily import TavilyClient
            
            if self._client is None:
                self._client = TavilyClient(api_key=self.api_key)
            
            logger.info(f"🔍 执行 Tavily 搜索: {input}")
            
            response = self._client.search(
                query=input,
                max_results=self.max_results
            )
            
            results = response.get("results", [])
            
            if not results:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output="未找到相关搜索结果。",
                    metadata={"query": input, "count": 0}
                )
            
            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                content = result.get("content", "无内容")
                url = result.get("url", "")
                
                formatted_results.append(
                    f"[{i}] {title}\n"
                    f"    {content}\n"
                    f"    链接: {url}"
                )
            
            output = "\n\n".join(formatted_results)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "query": input,
                    "count": len(results)
                }
            )
            
        except ImportError:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                output="",
                error="请安装 tavily-python: pip install tavily-python"
            )
        except Exception as e:
            logger.error(f"❌ Tavily 搜索失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e)
            )

