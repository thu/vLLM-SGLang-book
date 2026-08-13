"""
src/tools/__init__.py
工具管理器：注册、调用、Schema 汇总的统一入口。
扩展新工具时只需在 _register_tools() 中添加一行，对话管理器无需修改。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import BaseTool, ToolOutput
from .hotel import HotelTool
from .map import MapTool
from .weather import WeatherTool


class ToolManager:
    """
    工具注册与调用的统一入口。
    设计原则:
    - 对话管理器通过 run_tool(name, **kwargs) 调用工具，不依赖具体工具类
    - get_all_schemas() 汇总所有工具的 JSON Schema，供 LLM 工具决策使用
    """

    def __init__(self, use_mock: bool = True) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._register_tools(use_mock)

    def _register_tools(self, use_mock: bool) -> None:
        """注册所有工具。新增工具只需在此处添加一行。"""
        self._tools["get_weather"] = WeatherTool(use_mock=use_mock)
        self._tools["search_map"] = MapTool(use_mock=use_mock)
        self._tools["search_hotel"] = HotelTool(use_mock=use_mock)
        logger.info(
            f"已注册 {len(self._tools)} 个工具: "
            f"{list(self._tools.keys())}"
        )

    async def run_tool(self, name: str, **kwargs: Any) -> ToolOutput:
        """
        统一调用入口。工具不存在时返回明确的错误信息（不抛异常）。
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning(f"调用了未注册的工具: {name}")
            return ToolOutput(
                success=False,
                error=f"工具 '{name}' 未注册",
                message="抱歉，该功能暂不支持，请换一个方式描述您的需求。",
            )
        return await tool.run(**kwargs)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的 JSON Schema 列表，供 LLM 工具选择函数使用。"""
        return [t.get_schema() for t in self._tools.values()]

    def get_tools_description(self) -> str:
        """获取工具列表的简洁文本描述，用于注入系统提示。"""
        lines = [
            f"- {t.name}: {t.description}"
            for t in self._tools.values()
        ]
        return "\n".join(lines)

    def get_schemas_json(self) -> str:
        """获取所有工具 Schema 的 JSON 字符串，供 extract_tool_args 函数使用。"""
        return json.dumps(self.get_all_schemas(), ensure_ascii=False, indent=2)

    def get_tool_schema_json(self, tool_name: str) -> Optional[str]:
        """获取指定工具的 Schema JSON，供参数提取函数使用。"""
        tool = self._tools.get(tool_name)
        if tool is None:
            return None
        return json.dumps(tool.get_schema(), ensure_ascii=False, indent=2)

    async def close_all(self) -> None:
        """关闭所有工具的 HTTP 客户端，在应用关闭时调用。"""
        for tool in self._tools.values():
            await tool.close()


__all__ = ["ToolManager", "ToolOutput", "BaseTool"]
