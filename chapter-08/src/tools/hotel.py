"""
src/tools/hotel.py
酒店查询工具，覆盖三个价格档次。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolOutput

_MOCK_HOTELS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "昆明": {
        "经济型": [
            {"name": "如家酒店（昆明翠湖店）", "rating": 4.2, "price": 188,
             "address": "五华区翠湖北路", "facilities": ["WiFi", "24小时前台", "停车场"]},
            {"name": "汉庭酒店（昆明火车站店）", "rating": 4.1, "price": 168,
             "address": "官渡区北京路", "facilities": ["WiFi", "早餐", "24小时前台"]},
        ],
        "舒适型": [
            {"name": "昆明锦江酒店", "rating": 4.5, "price": 388,
             "address": "五华区东风东路", "facilities": ["WiFi", "游泳池", "健身房", "餐厅"]},
            {"name": "昆明金鹰大酒店", "rating": 4.4, "price": 358,
             "address": "盘龙区北京路", "facilities": ["WiFi", "停车场", "健身房", "餐厅"]},
        ],
        "豪华型": [
            {"name": "昆明柏联精品酒店", "rating": 4.8, "price": 880,
             "address": "西山区滇池路", "facilities": ["WiFi", "游泳池", "SPA", "多个餐厅", "洱海景观"]},
        ],
    },
    "大理": {
        "经济型": [
            {"name": "大理古城青年旅社", "rating": 4.3, "price": 120,
             "address": "古城区人民路", "facilities": ["WiFi", "公共厨房", "洗衣机"]},
        ],
        "舒适型": [
            {"name": "大理洱海边客栈", "rating": 4.6, "price": 468,
             "address": "大理市海东镇", "facilities": ["WiFi", "洱海景观房", "餐厅", "自行车租赁"]},
            {"name": "大理古城精品民宿", "rating": 4.5, "price": 328,
             "address": "古城区博爱路", "facilities": ["WiFi", "庭院", "茶室"]},
        ],
        "豪华型": [
            {"name": "大理希尔顿酒店", "rating": 4.8, "price": 1080,
             "address": "大理市下关镇", "facilities": ["WiFi", "游泳池", "SPA", "多个餐厅", "苍山洱海景观"]},
        ],
    },
    "丽江": {
        "舒适型": [
            {"name": "丽江古城客栈（玉河边）", "rating": 4.5, "price": 358,
             "address": "古城区七一街", "facilities": ["WiFi", "庭院", "雪山景观"]},
            {"name": "束河古镇精品民宿", "rating": 4.4, "price": 298,
             "address": "束河古镇", "facilities": ["WiFi", "庭院", "安静环境"]},
        ],
        "豪华型": [
            {"name": "丽江悦榕庄", "rating": 4.9, "price": 1580,
             "address": "束河古镇旁", "facilities": ["WiFi", "私人别墅", "SPA", "餐厅", "玉龙雪山景观"]},
        ],
    },
}


class HotelTool(BaseTool):
    """酒店查询工具，支持按价格档次筛选。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = True,
    ) -> None:
        super().__init__(
            name="search_hotel",
            description="搜索指定城市的酒店，支持按经济型/舒适型/豪华型筛选，返回价格和设施信息",
            use_mock=use_mock,
        )
        self.api_key = api_key

    async def _execute(
        self,
        city: str,
        check_in: str,
        check_out: str,
        price_range: str = "舒适型",
        num_rooms: int = 1,
        **kwargs: Any,
    ) -> ToolOutput:
        # 计算住宿天数
        try:
            nights = self._calc_nights(check_in, check_out)
        except Exception:
            nights = 3  # 无法解析时默认3晚

        if self.use_mock:
            return await self._mock_search(
                city, price_range, nights, num_rooms
            )
        raise NotImplementedError("真实酒店 API 未集成，请设置 use_mock=True")

    @staticmethod
    def _calc_nights(check_in: str, check_out: str) -> int:
        """计算住宿天数，支持 YYYY-MM-DD 格式；解析失败返回 3。"""
        try:
            d1 = datetime.strptime(check_in, "%Y-%m-%d")
            d2 = datetime.strptime(check_out, "%Y-%m-%d")
            nights = (d2 - d1).days
            return max(nights, 1)
        except (ValueError, TypeError):
            return 3

    async def _mock_search(
        self,
        city: str,
        price_range: str,
        nights: int,
        num_rooms: int,
    ) -> ToolOutput:
        city_data = _MOCK_HOTELS.get(city)
        if not city_data:
            return ToolOutput(
                success=False,
                error=f"暂无{city}的酒店数据",
                message=f"暂无{city}的酒店数据，建议通过携程或去哪儿搜索。",
            )

        hotels = city_data.get(price_range, [])
        if not hotels:
            # 找相邻价位
            for fallback in ["舒适型", "经济型", "豪华型"]:
                if fallback != price_range and city_data.get(fallback):
                    hotels = city_data[fallback]
                    price_range = fallback
                    break

        results = []
        for h in hotels[:3]:
            h_copy = h.copy()
            h_copy["total_price"] = h["price"] * nights * num_rooms
            h_copy["nights"] = nights
            h_copy["num_rooms"] = num_rooms
            results.append(h_copy)

        message = (
            f"在{city}找到 {len(results)} 家{price_range}酒店，"
            f"{nights}晚总价约 {results[0]['total_price']}-"
            f"{results[-1]['total_price']}元"
            if results else f"未找到{city}的{price_range}酒店"
        )
        return ToolOutput(
            success=True,
            data={
                "city": city,
                "price_range": price_range,
                "nights": nights,
                "num_rooms": num_rooms,
                "count": len(results),
                "hotels": results,
            },
            message=message,
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "check_in": {
                    "type": "string",
                    "description": "入住日期，格式 YYYY-MM-DD 或自然语言描述",
                },
                "check_out": {
                    "type": "string",
                    "description": "退房日期，格式 YYYY-MM-DD 或自然语言描述",
                },
                "price_range": {
                    "type": "string",
                    "enum": ["经济型", "舒适型", "豪华型"],
                    "description": "价格档次，默认舒适型",
                },
                "num_rooms": {
                    "type": "integer",
                    "description": "所需房间数，默认1",
                },
            },
            "required": ["city", "check_in", "check_out"],
        }

    def _get_fallback_message(self) -> str:
        return (
            "酒店信息暂时无法获取。价格参考（旺季）: "
            "经济型约150-280元/晚，舒适型约300-500元/晚，豪华型600元+/晚。"
            "建议通过携程或去哪儿搜索最新价格和可用性。"
        )
