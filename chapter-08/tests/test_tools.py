"""
tests/test_tools.py
工具层的单元测试（使用模拟数据，不需要真实 API）。

运行: pytest tests/test_tools.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import ToolManager
from src.tools.base import ToolOutput
from src.tools.hotel import HotelTool
from src.tools.map import MapTool
from src.tools.weather import WeatherTool


class TestWeatherTool:

    @pytest.fixture
    def tool(self):
        return WeatherTool(use_mock=True)

    @pytest.mark.asyncio
    async def test_known_city_returns_success(self, tool):
        result = await tool.run(city="昆明")
        assert result.success is True
        assert result.data is not None
        assert result.data["city"] == "昆明"
        assert "temperature" in result.data
        assert "weather" in result.data
        assert "suggestion" in result.data

    @pytest.mark.asyncio
    async def test_unknown_city_returns_success_with_default(self, tool):
        """未知城市使用默认数据，不返回失败。"""
        result = await tool.run(city="某个不存在的城市")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_with_date_param(self, tool):
        result = await tool.run(city="大理", date="3月初")
        assert result.success is True
        assert result.data["date"] == "3月初"

    def test_schema_has_required_fields(self, tool):
        schema = tool.get_schema()
        assert schema["name"] == "get_weather"
        required = schema["parameters"]["required"]
        assert "city" in required

    def test_fallback_message_not_empty(self, tool):
        msg = tool._get_fallback_message()
        assert len(msg) > 30
        assert "云南" in msg  # 应包含有价值的参考信息


class TestMapTool:

    @pytest.fixture
    def tool(self):
        return MapTool(use_mock=True)

    @pytest.mark.asyncio
    async def test_known_city_returns_pois(self, tool):
        result = await tool.run(city="昆明", keyword="景点")
        assert result.success is True
        assert result.data["count"] > 0
        pois = result.data["pois"]
        assert len(pois) > 0
        # 每个 POI 应包含必要字段
        for poi in pois:
            assert "name" in poi
            assert "rating" in poi
            assert "address" in poi

    @pytest.mark.asyncio
    async def test_category_filter(self, tool):
        result = await tool.run(
            city="大理", keyword="景点", category="自然景观"
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unknown_city_returns_failure(self, tool):
        result = await tool.run(city="火星城市", keyword="景点")
        assert result.success is False
        assert result.message is not None

    def test_schema_requires_city(self, tool):
        schema = tool.get_schema()
        assert "city" in schema["parameters"]["required"]


class TestHotelTool:

    @pytest.fixture
    def tool(self):
        return HotelTool(use_mock=True)

    @pytest.mark.asyncio
    async def test_basic_search(self, tool):
        result = await tool.run(
            city="昆明",
            check_in="2024-03-01",
            check_out="2024-03-05",
            price_range="舒适型",
        )
        assert result.success is True
        assert result.data["nights"] == 4
        hotels = result.data["hotels"]
        assert len(hotels) > 0
        for h in hotels:
            assert "name" in h
            assert "price" in h
            assert "total_price" in h
            assert h["total_price"] == h["price"] * 4

    @pytest.mark.asyncio
    async def test_invalid_dates_fallback(self, tool):
        """日期解析失败时使用默认3晚，不报错。"""
        result = await tool.run(
            city="大理",
            check_in="下个月初",
            check_out="五天后",
            price_range="舒适型",
        )
        assert result.success is True
        # 无法解析时默认 3 晚
        assert result.data["nights"] == 3

    @pytest.mark.asyncio
    async def test_price_range_fallback(self, tool):
        """请求的价格档次无数据时，回退到其他档次而非报错。"""
        result = await tool.run(
            city="昆明",
            check_in="2024-03-01",
            check_out="2024-03-03",
            price_range="豪华型",
        )
        # 即使豪华型无数据也应返回结果（回退到其他档次）
        assert result.success is True

    def test_schema_required_fields(self, tool):
        schema = tool.get_schema()
        required = set(schema["parameters"]["required"])
        assert required == {"city", "check_in", "check_out"}


class TestToolManager:

    @pytest.fixture
    def mgr(self):
        return ToolManager(use_mock=True)

    def test_all_tools_registered(self, mgr):
        schemas = mgr.get_all_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"get_weather", "search_map", "search_hotel"}

    @pytest.mark.asyncio
    async def test_run_existing_tool(self, mgr):
        result = await mgr.run_tool("get_weather", city="昆明")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_nonexistent_tool(self, mgr):
        result = await mgr.run_tool("nonexistent_tool")
        assert result.success is False
        assert "未注册" in result.error

    def test_get_tools_description_not_empty(self, mgr):
        desc = mgr.get_tools_description()
        assert "get_weather" in desc
        assert "search_map" in desc
        assert "search_hotel" in desc

    @pytest.mark.asyncio
    async def test_close_all_no_error(self, mgr):
        """close_all 不应抛出异常。"""
        await mgr.close_all()
