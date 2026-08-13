"""
tests/test_state.py
对话状态数据模型的单元测试。
不依赖 SGLang 或 Redis，可在任何环境中运行。

运行: pytest tests/test_state.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state import (
    DialogueState,
    DialogueTurn,
    ToolCallRecord,
    TravelPreferences,
)


class TestTravelPreferences:
    """TravelPreferences 槽位模型测试。"""

    def test_required_empty_slots_all_empty(self):
        prefs = TravelPreferences()
        empty = prefs.required_empty_slots()
        assert set(empty) == {
            "destination", "start_date", "end_date", "num_travelers"
        }

    def test_required_empty_slots_partial(self):
        prefs = TravelPreferences(destination="云南", num_travelers=2)
        empty = prefs.required_empty_slots()
        assert set(empty) == {"start_date", "end_date"}

    def test_is_complete_false(self):
        prefs = TravelPreferences(destination="云南")
        assert prefs.is_complete() is False

    def test_is_complete_true(self):
        prefs = TravelPreferences(
            destination="云南",
            start_date="下个月初",
            end_date="五天后",
            num_travelers=2,
        )
        assert prefs.is_complete() is True

    def test_completion_rate(self):
        prefs = TravelPreferences(destination="云南", num_travelers=2)
        rate = prefs.completion_rate()
        # 2/8 = 0.25
        assert abs(rate - 0.25) < 0.01

    def test_to_summary_text_empty(self):
        prefs = TravelPreferences()
        text = prefs.to_summary_text()
        assert text == "（暂无已确认信息）"

    def test_to_summary_text_filled(self):
        prefs = TravelPreferences(
            destination="云南",
            num_travelers=2,
            budget="一万元",
        )
        text = prefs.to_summary_text()
        assert "云南" in text
        assert "2人" in text
        assert "一万元" in text


class TestDialogueState:
    """DialogueState 主模型测试。"""

    def _make_state(self) -> DialogueState:
        return DialogueState(session_id="test-001", user_id="user-001")

    def test_add_turn_increments_count(self):
        state = self._make_state()
        assert state.turn_count == 0
        state.add_turn("user", "你好")
        assert state.turn_count == 1
        state.add_turn("assistant", "你好！")
        # assistant 不增加 turn_count
        assert state.turn_count == 1

    def test_add_turn_stores_intent(self):
        state = self._make_state()
        state.add_turn("user", "我想去旅游", intent="plan_trip")
        turn = state.dialogue_history[0]
        assert turn.intent == "plan_trip"

    def test_get_context_window_truncates(self):
        state = self._make_state()
        for i in range(15):
            state.add_turn("user", f"消息{i}")
        window = state.get_context_window(max_turns=10)
        assert len(window) == 10
        # 应返回最后 10 条
        assert window[-1].content == "消息14"

    def test_update_preferences_basic(self):
        state = self._make_state()
        changed = state.update_preferences(destination="云南", num_travelers=2)
        assert "destination" in changed
        assert "num_travelers" in changed
        assert state.preferences.destination == "云南"
        assert state.preferences.num_travelers == 2

    def test_update_preferences_overwrite(self):
        """覆盖语义：允许用户中途更改决策。"""
        state = self._make_state()
        state.update_preferences(destination="北京")
        changed = state.update_preferences(destination="云南")
        assert "destination" in changed
        assert state.preferences.destination == "云南"

    def test_update_preferences_no_change(self):
        """相同值更新不产生变更记录。"""
        state = self._make_state()
        state.update_preferences(destination="云南")
        changed = state.update_preferences(destination="云南")
        assert "destination" not in changed

    def test_update_preferences_unknown_field(self):
        """未知字段应被跳过，不抛异常。"""
        state = self._make_state()
        changed = state.update_preferences(unknown_field="value")
        assert changed == []

    def test_update_preferences_none_value(self):
        """None 值不应覆盖已有值。"""
        state = self._make_state()
        state.update_preferences(destination="云南")
        state.update_preferences(destination=None)
        assert state.preferences.destination == "云南"

    def test_add_tool_call(self):
        state = self._make_state()
        state.add_tool_call(
            "get_weather",
            {"city": "昆明"},
            {"weather": "晴天"},
            True,
        )
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].tool_name == "get_weather"
        assert state.tool_calls[0].success is True

    def test_has_tool_been_called_true(self):
        state = self._make_state()
        state.add_tool_call("get_weather", {"city": "昆明"}, {}, True)
        assert state.has_tool_been_called("get_weather") is True

    def test_has_tool_been_called_false(self):
        state = self._make_state()
        assert state.has_tool_been_called("get_weather") is False

    def test_has_tool_been_called_within_turns_limit(self):
        """超出 within_turns 范围的调用不算。"""
        state = self._make_state()
        state.add_tool_call("get_weather", {}, {}, True)
        # 添加 10 条记录后，within_turns=5 应该找不到之前的调用
        for _ in range(10):
            state.add_tool_call("other_tool", {}, {}, True)
        assert state.has_tool_been_called("get_weather", within_turns=5) is False

    def test_serialization_roundtrip(self):
        """序列化后反序列化，数据应完全一致。"""
        state = self._make_state()
        state.add_turn("user", "我想去云南", intent="plan_trip")
        state.update_preferences(destination="云南", num_travelers=2)
        state.add_tool_call("get_weather", {"city": "昆明"}, {}, True)

        serialized = state.to_redis_value()
        restored = DialogueState.from_redis_value(serialized)

        assert restored.session_id == state.session_id
        assert restored.turn_count == state.turn_count
        assert restored.preferences.destination == "云南"
        assert restored.preferences.num_travelers == 2
        assert len(restored.tool_calls) == 1
        assert len(restored.dialogue_history) == 1
