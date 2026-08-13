"""
src/tools/weather.py
天气查询工具。模拟数据覆盖云南主要旅游城市。
真实 API 集成时，替换 _real_weather() 的内部实现即可，上层代码无需修改。
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

from .base import BaseTool, ToolOutput

# 各城市的模拟天气数据
_MOCK_WEATHER_DB: Dict[str, Dict[str, Any]] = {
    "昆明": {
        "seasons": {
            "spring": {"weather": "晴天", "temp_min": 12, "temp_max": 22},
            "summer": {"weather": "多云转阵雨", "temp_min": 18, "temp_max": 26},
            "autumn": {"weather": "晴到多云", "temp_min": 10, "temp_max": 20},
            "winter": {"weather": "晴天", "temp_min": 5, "temp_max": 17},
        },
        "note": "昆明四季如春，气候宜人",
    },
    "大理": {
        "seasons": {
            "spring": {"weather": "晴天", "temp_min": 10, "temp_max": 21},
            "summer": {"weather": "多云，午后阵雨", "temp_min": 16, "temp_max": 24},
            "autumn": {"weather": "晴天", "temp_min": 8, "temp_max": 20},
            "winter": {"weather": "晴天", "temp_min": 3, "temp_max": 16},
        },
        "note": "大理风大，建议携带外套",
    },
    "丽江": {
        "seasons": {
            "spring": {"weather": "晴天", "temp_min": 8, "temp_max": 20},
            "summer": {"weather": "多云，雨季", "temp_min": 14, "temp_max": 22},
            "autumn": {"weather": "晴到多云", "temp_min": 5, "temp_max": 18},
            "winter": {"weather": "晴天", "temp_min": -2, "temp_max": 14},
        },
        "note": "丽江海拔较高，昼夜温差大，建议备厚外套",
    },
    "西双版纳": {
        "seasons": {
            "spring": {"weather": "晴天，干季", "temp_min": 18, "temp_max": 30},
            "summer": {"weather": "多雨，雨季", "temp_min": 22, "temp_max": 33},
            "autumn": {"weather": "晴雨交替", "temp_min": 18, "temp_max": 28},
            "winter": {"weather": "晴天，雾多", "temp_min": 13, "temp_max": 25},
        },
        "note": "热带气候，5-10月为雨季",
    },
}

_DEFAULT_WEATHER = {
    "seasons": {
        "spring": {"weather": "晴天", "temp_min": 14, "temp_max": 24},
        "summer": {"weather": "多云", "temp_min": 18, "temp_max": 28},
        "autumn": {"weather": "晴天", "temp_min": 12, "temp_max": 22},
        "winter": {"weather": "多云", "temp_min": 5, "temp_max": 15},
    },
    "note": "气候适宜",
}


def _date_to_season(date_str: Optional[str]) -> str:
    """将日期描述（可为原始文本）粗略映射到季节，用于模拟数据查询。"""
    if not date_str:
        return "spring"
    s = date_str.lower()
    if any(k in s for k in ["3月", "4月", "5月", "march", "april", "may", "春"]):
        return "spring"
    if any(k in s for k in ["6月", "7月", "8月", "june", "july", "aug", "夏", "暑"]):
        return "summer"
    if any(k in s for k in ["9月", "10月", "11月", "sep", "oct", "nov", "秋"]):
        return "autumn"
    if any(k in s for k in ["12月", "1月", "2月", "dec", "jan", "feb", "冬"]):
        return "winter"
    # 默认下个月 → 根据实际月份推断（此处简化为 spring）
    return "spring"


class WeatherTool(BaseTool):
    """天气查询工具，支持云南主要旅游城市的模拟数据。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = True,
    ) -> None:
        super().__init__(
            name="get_weather",
            description="查询指定城市的天气信息，包括温度、天气状况和出行建议",
            use_mock=use_mock,
        )
        self.api_key = api_key

    async def _execute(
        self, city: str, date: Optional[str] = None, **kwargs: Any
    ) -> ToolOutput:
        if self.use_mock:
            return await self._mock_weather(city, date)
        return await self._real_weather(city, date)

    async def _mock_weather(
        self, city: str, date: Optional[str]
    ) -> ToolOutput:
        city_data = _MOCK_WEATHER_DB.get(city, _DEFAULT_WEATHER)
        season = _date_to_season(date)
        season_data = city_data["seasons"][season]

        # 在基础数据上加少量随机扰动，模拟真实 API 的数据多样性
        temp_min = season_data["temp_min"] + random.randint(-2, 2)
        temp_max = season_data["temp_max"] + random.randint(-2, 2)
        humidity = random.randint(45, 75)

        suggestion = self._build_suggestion(
            season_data["weather"], temp_max
        )
        query_date = date or "近期"

        data = {
            "city": city,
            "date": query_date,
            "weather": season_data["weather"],
            "temperature": f"{temp_min}-{temp_max}°C",
            "humidity": f"{humidity}%",
            "suggestion": suggestion,
            "note": city_data.get("note", ""),
        }
        message = (
            f"{city}{query_date}天气: {season_data['weather']}，"
            f"气温 {temp_min}-{temp_max}°C，{suggestion}"
        )
        return ToolOutput(success=True, data=data, message=message)

    async def _real_weather(
        self, city: str, date: Optional[str]
    ) -> ToolOutput:
        """
        真实 API 调用占位（此处以和风天气 API 为例）。
        集成时替换此方法内部实现，上层代码无需修改。
        """
        # url = "https://devapi.qweather.com/v7/weather/now"
        # params = {"location": city, "key": self.api_key}
        # resp = await self._client.get(url, params=params)
        # ...解析响应...
        raise NotImplementedError("真实天气 API 未集成，请设置 use_mock=True")

    @staticmethod
    def _build_suggestion(weather: str, temp_max: int) -> str:
        parts = []
        if "雨" in weather:
            parts.append("建议携带雨具")
        if temp_max > 28:
            parts.append("温度较高，注意防晒补水")
        elif temp_max < 10:
            parts.append("气温较低，注意保暖")
        return "，".join(parts) if parts else "天气适宜，适合出行"

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'昆明'、'大理'、'丽江'",
                },
                "date": {
                    "type": "string",
                    "description": "查询日期，支持自然语言描述，如'3月初'",
                },
            },
            "required": ["city"],
        }

    def _get_fallback_message(self) -> str:
        return (
            "天气信息暂时无法获取。根据历史气候参考：云南地区全年气候温和，"
            "3-5月春季晴天为主，气温约12-22°C；7-9月为雨季，18-26°C；"
            "冬季（12-2月）昆明晴天较多，丽江较冷，建议出发前再次查询实时天气。"
        )
