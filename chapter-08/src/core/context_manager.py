"""
src/core/context_manager.py
三策略自适应上下文窗口管理器。

策略:
  full      - 全量历史（对话 ≤8 轮或 token 充足）
  sliding   - 滑动窗口（8-20 轮，截取最近 N 轮 + 槽位摘要）
  summarize - LLM 摘要压缩（>20 轮，调用 generate_context_summary）

Token 估算:
  使用 0.4 字符/token 的经验值（中英文混合场景），误差约 ±20%。
  策略切换阈值预留 20% 余量以补偿估算偏差。
  如需精确统计，应使用 transformers.AutoTokenizer，但热路径上开销约 10ms。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .state import DialogueState, DialogueTurn, TravelPreferences


@dataclass
class ContextWindow:
    """上下文窗口的结构化表示。"""
    turns: List[DialogueTurn]
    token_count: int
    strategy: str                    # "full" | "sliding" | "summarize"
    has_summary: bool = False
    summary_text: Optional[str] = None
    slot_summary: Optional[str] = None


class ContextWindowManager:
    """
    三策略自适应上下文管理器。
    根据对话轮次数和 token 估算量自动选择最优策略。
    """

    def __init__(
        self,
        max_context_tokens: int = 2048,
        window_size: int = 8,
        enable_summarization: bool = True,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.window_size = window_size
        self.enable_summarization = enable_summarization

    def get_optimal_context(self, state: DialogueState) -> ContextWindow:
        """
        根据当前对话状态自动选择最优上下文策略并返回上下文窗口。
        决策依据: 历史轮次数 + token 估算量。
        """
        history = state.dialogue_history
        n = len(history)
        total_tokens = self._estimate_tokens(history)
        # 预留 20% 余量补偿 token 估算误差
        effective_limit = int(self.max_context_tokens * 0.8)

        if n <= 8 or total_tokens <= effective_limit:
            # 全量历史：轮次少或 token 充足
            return ContextWindow(
                turns=history,
                token_count=total_tokens,
                strategy="full",
            )
        elif n <= 20:
            # 滑动窗口：截取最近 window_size 轮，槽位摘要补充早期信息
            recent = history[-self.window_size:]
            return ContextWindow(
                turns=recent,
                token_count=self._estimate_tokens(recent),
                strategy="sliding",
                slot_summary=state.preferences.to_summary_text(),
            )
        else:
            # LLM 摘要压缩
            if self.enable_summarization:
                return self._summarize_window(state)
            else:
                return self._truncate_window(history)

    def _summarize_window(self, state: DialogueState) -> ContextWindow:
        """
        对滑出窗口的早期历史生成摘要（调用 generate_context_summary）。
        注意: 此方法为同步方法，实际调用 SGLang 函数需在异步上下文中执行。
        调用方应使用 get_optimal_context_async() 版本。
        """
        history = state.dialogue_history
        cutoff = max(0, len(history) - self.window_size)
        early_turns = history[:cutoff]
        recent_turns = history[cutoff:]

        # 构建早期对话文本（供摘要生成函数使用）
        early_text = "\n".join(
            f"{'用户' if t.role=='user' else '助手'}: {t.content}"
            for t in early_turns
        )

        # 同步降级：使用规则摘要替代 LLM 摘要
        # 真实部署中应使用 async 版本调用 generate_context_summary
        summary = self._rule_based_summary(early_turns, state.preferences)

        return ContextWindow(
            turns=recent_turns,
            token_count=self._estimate_tokens(recent_turns) + len(summary) // 3,
            strategy="summarize",
            has_summary=True,
            summary_text=summary,
            slot_summary=state.preferences.to_summary_text(),
        )

    async def get_optimal_context_async(
        self, state: DialogueState
    ) -> ContextWindow:
        """
        异步版本：>20 轮时调用 LLM 生成摘要（而非规则摘要）。
        在对话管理器中使用此版本以获得最高质量的摘要。
        """
        from .dialogue_functions import generate_context_summary

        history = state.dialogue_history
        n = len(history)
        total_tokens = self._estimate_tokens(history)
        effective_limit = int(self.max_context_tokens * 0.8)

        if n <= 8 or total_tokens <= effective_limit:
            return ContextWindow(
                turns=history, token_count=total_tokens, strategy="full"
            )
        elif n <= 20:
            recent = history[-self.window_size:]
            return ContextWindow(
                turns=recent,
                token_count=self._estimate_tokens(recent),
                strategy="sliding",
                slot_summary=state.preferences.to_summary_text(),
            )
        else:
            # 使用 LLM 生成高质量摘要
            cutoff = max(0, len(history) - self.window_size)
            early_text = "\n".join(
                f"{'用户' if t.role=='user' else '助手'}: {t.content}"
                for t in history[:cutoff]
            )
            prefs_json = state.preferences.model_dump_json(
                exclude_none=True
            )
            try:
                result = generate_context_summary.run(
                    early_dialogue_text=early_text,
                    current_slots_json=prefs_json,
                )
                summary = result.get("summary", "").strip()
            except Exception as exc:
                logger.warning(f"LLM 摘要生成失败，使用规则摘要 | {exc}")
                summary = self._rule_based_summary(
                    history[:cutoff], state.preferences
                )

            recent = history[cutoff:]
            return ContextWindow(
                turns=recent,
                token_count=self._estimate_tokens(recent) + len(summary) // 3,
                strategy="summarize",
                has_summary=True,
                summary_text=summary,
                slot_summary=state.preferences.to_summary_text(),
            )

    def _truncate_window(
        self, history: List[DialogueTurn]
    ) -> ContextWindow:
        """强制截断：从最近轮次开始向前累积，直到 token 接近上限。"""
        selected: List[DialogueTurn] = []
        current_tokens = 0
        for turn in reversed(history):
            t = self._estimate_tokens([turn])
            if current_tokens + t <= self.max_context_tokens:
                selected.insert(0, turn)
                current_tokens += t
            else:
                break
        logger.warning(
            f"强制截断上下文 | 保留 {len(selected)}/{len(history)} 轮"
        )
        return ContextWindow(
            turns=selected, token_count=current_tokens, strategy="truncate"
        )

    def _rule_based_summary(
        self,
        turns: List[DialogueTurn],
        prefs: TravelPreferences,
    ) -> str:
        """
        规则摘要（LLM 摘要失败时的降级方案）。
        提取工具调用结论和用户特殊表达，槽位信息不重复（已在系统提示中）。
        """
        parts = [f"已进行 {len(turns)} 轮对话。"]

        # 提取工具调用记录（结构化元数据）
        tool_calls = []
        for t in turns:
            if t.tool_calls:
                tool_calls.extend(t.tool_calls)
        if tool_calls:
            parts.append(f"已查询: {', '.join(set(tool_calls))}。")

        return "".join(parts)

    def _estimate_tokens(self, turns: List[DialogueTurn]) -> int:
        """
        估算对话轮次的 token 数（0.4 字符/token 经验值）。
        误差约 ±20%，策略切换阈值预留对应余量。
        精确计数应使用 transformers.AutoTokenizer，但热路径开销约 10ms。
        """
        total_chars = sum(len(t.content) for t in turns)
        return int(total_chars * 0.4)

    def build_history_messages(
        self, window: ContextWindow
    ) -> List[Dict[str, str]]:
        """
        将上下文窗口转换为 {"role": ..., "content": ...} 格式的消息列表，
        供 generate_direct_reply 等 SGLang 函数使用。
        """
        messages: List[Dict[str, str]] = []

        # 如果有摘要，作为 system 补充信息插入
        if window.has_summary and window.summary_text:
            messages.append({
                "role": "system",
                "content": f"[早期对话摘要] {window.summary_text}",
            })

        for turn in window.turns:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages
