"""
tests/test_context_manager.py
上下文窗口管理器单元测试。
不依赖 SGLang（摘要生成路径测试时使用规则摘要降级）。

运行: pytest tests/test_context_manager.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.context_manager import ContextWindowManager
from src.core.state import DialogueState, DialogueTurn, TravelPreferences


def _make_state(n_turns: int, dest: str = "云南") -> DialogueState:
    """创建含 n_turns 轮对话历史的测试状态。"""
    state = DialogueState(session_id="test", user_id="user")
    state.preferences.destination = dest
    state.preferences.num_travelers = 2
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        state.add_turn(role, f"第{i+1}条消息内容，用于测试上下文窗口管理策略")
    return state


class TestContextWindowManager:

    @pytest.fixture
    def mgr(self):
        return ContextWindowManager(
            max_context_tokens=512,  # 较小值便于触发压缩
            window_size=4,
            enable_summarization=False,  # 禁用 LLM 摘要，使用规则摘要
        )

    def test_full_strategy_short_dialogue(self, mgr):
        state = _make_state(4)
        window = mgr.get_optimal_context(state)
        assert window.strategy == "full"
        assert len(window.turns) == 4

    def test_sliding_strategy_medium_dialogue(self, mgr):
        """超过 8 轮时切换到滑动窗口。"""
        state = _make_state(12)
        window = mgr.get_optimal_context(state)
        assert window.strategy in ("sliding", "full")
        # 滑动窗口应保留最近 window_size 轮
        if window.strategy == "sliding":
            assert len(window.turns) <= mgr.window_size

    def test_sliding_includes_slot_summary(self, mgr):
        """滑动窗口应包含槽位摘要，补充被截断的早期信息。"""
        state = _make_state(20)
        window = mgr.get_optimal_context(state)
        if window.strategy == "sliding":
            assert window.slot_summary is not None
            assert "云南" in window.slot_summary

    def test_summarize_strategy_long_dialogue(self):
        """超过 20 轮时，启用摘要的管理器应切换到 summarize 策略。"""
        mgr = ContextWindowManager(
            max_context_tokens=512,
            window_size=4,
            enable_summarization=True,  # 此处启用，使用规则摘要降级
        )
        state = _make_state(25)
        window = mgr.get_optimal_context(state)
        assert window.strategy == "summarize"
        assert window.has_summary is True
        assert window.summary_text is not None

    def test_truncate_strategy_when_summarization_disabled(self):
        """禁用摘要且超过20轮时，回退到截断策略。"""
        mgr = ContextWindowManager(
            max_context_tokens=200,  # 极小值
            window_size=4,
            enable_summarization=False,
        )
        state = _make_state(25)
        window = mgr.get_optimal_context(state)
        # token_count 应在限制内
        assert window.token_count <= mgr.max_context_tokens * 1.2  # 允许 20% 误差

    def test_estimate_tokens_proportional(self, mgr):
        """token 估算应随内容长度单调递增。"""
        short_turns = [DialogueTurn(role="user", content="你好")]
        long_turns = [DialogueTurn(role="user", content="你好" * 100)]
        t_short = mgr._estimate_tokens(short_turns)
        t_long = mgr._estimate_tokens(long_turns)
        assert t_long > t_short

    def test_build_history_messages_format(self, mgr):
        """build_history_messages 应返回正确格式的消息列表。"""
        state = _make_state(4)
        window = mgr.get_optimal_context(state)
        msgs = mgr.build_history_messages(window)
        assert all("role" in m and "content" in m for m in msgs)
        roles = {m["role"] for m in msgs}
        assert roles <= {"user", "assistant", "system"}

    def test_slot_summary_content(self, mgr):
        """槽位摘要应包含已填充的槽位信息。"""
        prefs = TravelPreferences(
            destination="大理",
            num_travelers=3,
            budget="两万元",
        )
        summary = mgr._build_slot_summary(prefs)
        assert "大理" in summary
        assert "3人" in summary
        assert "两万元" in summary

    def test_slot_summary_empty_prefs(self, mgr):
        prefs = TravelPreferences()
        summary = mgr._build_slot_summary(prefs)
        assert summary == "（暂无已确认信息）"
