"""
src/core/state.py
对话状态的完整数据模型。

设计原则:
- TravelPreferences 采用覆盖而非追加语义，支持用户中途更改决策
- DialogueTurn 携带结构化元数据，供上下文压缩时使用
- DialogueState 是系统在任意时刻做出决策所需的完整上下文
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# 槽位模型
# ────────────────────────────────────────────────────────────────────

class TravelPreferences(BaseModel):
    """
    旅行规划所需的槽位状态。
    前四项（destination/start_date/end_date/num_travelers）为必填项，
    其余四项为可选增强信息。
    """

    # 必填槽位（四项）
    destination: Optional[str] = None
    """目的地，如'云南'或'大理'"""

    start_date: Optional[str] = None
    """出发日期，保留原始描述（如'下个月初'），不强制转 YYYY-MM-DD"""

    end_date: Optional[str] = None
    """返回日期或行程结束描述"""

    num_travelers: Optional[int] = None
    """出行人数，整数类型"""

    # 可选槽位（四项）
    budget: Optional[str] = None
    """预算描述，如'一万元以内'"""

    accommodation_level: Optional[str] = None
    """住宿等级：经济型 / 舒适型 / 豪华型"""

    travel_style: Optional[str] = None
    """旅行风格：休闲 / 冒险 / 文化 / 美食"""

    interests: Optional[List[str]] = None
    """兴趣点列表，如 ['自然风光', '历史文化']"""

    # ── 必填槽位检查 ─────────────────────────────────────────────────

    def required_empty_slots(self) -> List[str]:
        """返回尚未填充的必填槽位名称列表，用于决策下一步询问什么。"""
        required = ["destination", "start_date", "end_date", "num_travelers"]
        return [s for s in required if getattr(self, s) is None]

    def is_complete(self) -> bool:
        """必填信息是否已全部收集，决定是否可以触发方案生成。"""
        return len(self.required_empty_slots()) == 0

    def completion_rate(self) -> float:
        """
        信息完整度（0.0 至 1.0），所有槽位（含可选）中已填充的比例。
        用于决策询问策略（完整度低时优先收集必填，高时可收集可选）。
        """
        all_fields = [
            "destination", "start_date", "end_date", "num_travelers",
            "budget", "accommodation_level", "travel_style", "interests",
        ]
        filled = sum(1 for f in all_fields if getattr(self, f) is not None)
        return filled / len(all_fields)

    def to_summary_text(self) -> str:
        """
        生成槽位状态的简洁文本摘要，用于注入系统提示。
        即使早期历史被上下文窗口截断，槽位信息通过此摘要保留。
        """
        parts: List[str] = []
        if self.destination:
            parts.append(f"目的地：{self.destination}")
        if self.start_date:
            parts.append(f"出发日期：{self.start_date}")
        if self.end_date:
            parts.append(f"行程结束：{self.end_date}")
        if self.num_travelers:
            parts.append(f"人数：{self.num_travelers}人")
        if self.budget:
            parts.append(f"预算：{self.budget}")
        if self.accommodation_level:
            parts.append(f"住宿：{self.accommodation_level}")
        if self.travel_style:
            parts.append(f"风格：{self.travel_style}")
        if self.interests:
            parts.append(f"兴趣：{'、'.join(self.interests)}")
        return "，".join(parts) if parts else "（暂无已确认信息）"


# ────────────────────────────────────────────────────────────────────
# 对话轮次
# ────────────────────────────────────────────────────────────────────

class DialogueTurn(BaseModel):
    """
    单轮对话记录，携带结构化元数据。
    intent / slots_extracted / tool_calls 在上下文摘要生成时使用，
    使摘要能基于结构化信息重建，而非重新阅读原始对话文本。
    """

    role: str
    """角色：'user' 或 'assistant'"""

    content: str
    """对话内容"""

    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    """ISO 格式时间戳，用于调试时的时序分析"""

    intent: Optional[str] = None
    """本轮识别到的意图类别"""

    slots_extracted: Optional[Dict[str, Any]] = None
    """本轮从用户输入中提取到的槽位键值对"""

    tool_calls: Optional[List[str]] = None
    """本轮触发的工具名称列表（仅记录名称，详情在 ToolCallRecord 中）"""


# ────────────────────────────────────────────────────────────────────
# 工具调用记录
# ────────────────────────────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    """工具调用的完整记录，用于去重保护和调试追溯。"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


# ────────────────────────────────────────────────────────────────────
# 对话状态（主模型）
# ────────────────────────────────────────────────────────────────────

class DialogueState(BaseModel):
    """
    完整的对话会话状态，是系统在任意时刻做出决策所需的全部上下文。

    持久化策略：
    - 存储于 Redis，键为 session:{session_id}，TTL 1800 秒
    - 每次成功处理一轮对话后整体写回（轮次粒度的原子性）
    - 处理过程中发生异常时不写回，避免部分更新
    """

    session_id: str
    user_id: str

    preferences: TravelPreferences = Field(
        default_factory=TravelPreferences
    )
    dialogue_history: List[DialogueTurn] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    turn_count: int = 0

    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    # ── 对话历史操作 ────────────────────────────────────────────────

    def add_turn(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        slots_extracted: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[str]] = None,
    ) -> None:
        """追加一轮对话记录，同时更新 turn_count 和 updated_at。"""
        self.dialogue_history.append(
            DialogueTurn(
                role=role,
                content=content,
                intent=intent,
                slots_extracted=slots_extracted,
                tool_calls=tool_calls,
            )
        )
        if role == "user":
            self.turn_count += 1
        self.updated_at = datetime.now().isoformat()

    def get_context_window(
        self, max_turns: int = 10
    ) -> List[DialogueTurn]:
        """
        返回最近 max_turns 条对话记录，用于构造推理提示。
        注意：更复杂的压缩策略由 ContextWindowManager 负责；
        此方法只做简单的尾部截取，适用于会话早期。
        """
        return self.dialogue_history[-max_turns:]

    # ── 槽位操作 ────────────────────────────────────────────────────

    def update_preferences(self, **kwargs: Any) -> List[str]:
        """
        覆盖式更新槽位，返回实际发生变更的槽位名称列表。
        覆盖而非追加语义：支持用户中途更改决策（如目的地从昆明改为大理）。
        变更日志记录于 logger，便于调试时追溯槽位变化过程。
        """
        changed: List[str] = []
        for key, value in kwargs.items():
            if not hasattr(self.preferences, key):
                logger.warning(f"未知槽位字段: {key}，已跳过")
                continue
            if value is None:
                continue
            old_val = getattr(self.preferences, key)
            if old_val != value:
                setattr(self.preferences, key, value)
                changed.append(key)
                if old_val is not None:
                    logger.info(
                        f"槽位覆盖 | session={self.session_id} "
                        f"| {key}: '{old_val}' → '{value}'"
                    )
        if changed:
            self.updated_at = datetime.now().isoformat()
        return changed

    # ── 工具调用操作 ────────────────────────────────────────────────

    def add_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """记录工具调用，供去重保护和调试使用。"""
        self.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                success=success,
                error=error,
            )
        )
        self.updated_at = datetime.now().isoformat()

    def has_tool_been_called(
        self, tool_name: str, within_turns: int = 5
    ) -> bool:
        """
        检查最近 within_turns 轮内是否已调用过指定工具。
        用于去重保护：避免同一对话中对天气等信息重复查询。
        """
        recent = self.tool_calls[-within_turns:]
        return any(tc.tool_name == tool_name for tc in recent)

    # ── 序列化辅助 ──────────────────────────────────────────────────

    def to_redis_value(self) -> str:
        """序列化为 Redis 存储字符串（Pydantic v2 高性能路径）。"""
        return self.model_dump_json()

    @classmethod
    def from_redis_value(cls, raw: str) -> "DialogueState":
        """从 Redis 读取的字符串反序列化（Pydantic v2）。"""
        return cls.model_validate_json(raw)
