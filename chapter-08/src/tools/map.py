"""
src/tools/map.py
地图/景点搜索工具，覆盖云南主要旅游城市的景点模拟数据。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolOutput

_MOCK_POIS: Dict[str, List[Dict[str, Any]]] = {
    "昆明": [
        {"name": "石林风景区", "rating": 4.6, "category": "自然景观",
         "address": "昆明市石林县", "description": "世界自然遗产，典型喀斯特地貌", "price": "130元"},
        {"name": "滇池", "rating": 4.3, "category": "自然景观",
         "address": "昆明市西山区", "description": "云南最大淡水湖，可骑行环湖", "price": "免费"},
        {"name": "翠湖公园", "rating": 4.5, "category": "公园",
         "address": "昆明市五华区", "description": "市区公园，冬季红嘴鸥聚集", "price": "免费"},
        {"name": "云南民族村", "rating": 4.2, "category": "文化",
         "address": "昆明市西山区", "description": "展示云南 26 个民族文化", "price": "80元"},
    ],
    "大理": [
        {"name": "洱海", "rating": 4.7, "category": "自然景观",
         "address": "大理市", "description": "云南第二大淡水湖，环湖风景极佳", "price": "免费"},
        {"name": "大理古城", "rating": 4.5, "category": "历史文化",
         "address": "大理市古城区", "description": "白族文化历史名城，保存完好", "price": "免费"},
        {"name": "苍山", "rating": 4.6, "category": "自然景观",
         "address": "大理市", "description": "最高峰海拔 4122 米，可索道上山", "price": "索道182元"},
        {"name": "双廊古镇", "rating": 4.4, "category": "古镇",
         "address": "大理市双廊镇", "description": "洱海边小镇，艺术气息浓厚", "price": "免费"},
    ],
    "丽江": [
        {"name": "丽江古城", "rating": 4.5, "category": "历史文化",
         "address": "丽江市古城区", "description": "世界文化遗产，纳西族文化核心", "price": "古城维护费80元"},
        {"name": "玉龙雪山", "rating": 4.6, "category": "自然景观",
         "address": "丽江市玉龙县", "description": "纳西族神山，最高峰5596米", "price": "索道+门票约520元"},
        {"name": "泸沽湖", "rating": 4.7, "category": "自然景观",
         "address": "丽江市宁蒗县", "description": "母系社会摩梭族聚居，高原湖泊", "price": "100元"},
        {"name": "束河古镇", "rating": 4.3, "category": "古镇",
         "address": "丽江市古城区", "description": "比丽江古城更安静，适合慢游", "price": "免费"},
    ],
    "西双版纳": [
        {"name": "野象谷", "rating": 4.4, "category": "自然景观",
         "address": "西双版纳勐养镇", "description": "亚洲象自然保护区，可近距离观象", "price": "80元"},
        {"name": "曼听公园", "rating": 4.2, "category": "文化",
         "address": "景洪市", "description": "傣族皇家御花园，大型文化展演", "price": "50元"},
    ],
}


class MapTool(BaseTool):
    """景点/地图搜索工具。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = True,
    ) -> None:
        super().__init__(
            name="search_map",
            description="搜索指定城市的景点、餐厅等 POI 信息，返回评分、地址和价格",
            use_mock=use_mock,
        )
        self.api_key = api_key

    async def _execute(
        self,
        city: str,
        keyword: str = "景点",
        category: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolOutput:
        if self.use_mock:
            return await self._mock_search(city, keyword, category)
        raise NotImplementedError("真实地图 API 未集成，请设置 use_mock=True")

    async def _mock_search(
        self,
        city: str,
        keyword: str,
        category: Optional[str],
    ) -> ToolOutput:
        pois = _MOCK_POIS.get(city)
        if not pois:
            return ToolOutput(
                success=False,
                error=f"暂无{city}的景点数据",
                message=f"抱歉，暂时没有{city}的景点数据，建议参考旅游攻略网站。",
            )

        # 按 category 或 keyword 过滤
        if category:
            filtered = [p for p in pois if category in p.get("category", "")]
        elif keyword and keyword not in ("景点", "推荐"):
            filtered = [
                p for p in pois
                if keyword in p["name"] or keyword in p.get("description", "")
            ]
        else:
            filtered = pois

        results = filtered[:5] if filtered else pois[:3]

        message = f"在{city}找到 {len(results)} 个相关地点"
        return ToolOutput(
            success=True,
            data={
                "city": city,
                "keyword": keyword,
                "count": len(results),
                "pois": results,
            },
            message=message,
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词，如'景点'、'古城'、'自然风光'",
                },
                "category": {
                    "type": "string",
                    "description": "类别过滤，如'自然景观'、'历史文化'、'古镇'",
                },
            },
            "required": ["city"],
        }

    def _get_fallback_message(self) -> str:
        return (
            "景点信息暂时无法获取。云南主要景点参考: "
            "昆明（石林、滇池）、大理（洱海、古城、苍山）、"
            "丽江（古城、玉龙雪山、泸沽湖）。建议出发前查阅最新攻略。"
        )
