"""
src/core/dialogue_manager.py
对话管理器：协调 SGLang 函数与工具调用，编排完整对话处理流程。

重要约束:
  本类不包含任何 @sgl.function。
  所有 SGLang 函数定义在 dialogue_functions.py 的模块顶层，
  本类通过直接调用这些函数来编排流程。

处理流程 (process_user_input):
  1. 意图识别（利用最近 3 轮上下文）
  2. 槽位提取（仅在相关意图下执行）
  3. 根据意图决策下一步行动
  4. 更新对话历史
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import sglang as sgl
from loguru import logger

from .context_manager import ContextWindowManager
from .dialogue_functions import (
    ask_missing_slot,
    extract_slots,
    generate_direct_reply,
    generate_travel_plan,
    recognize_intent,
)
from .state import DialogueState, TravelPreferences
from .tool_decision import ToolDecisionChain
from ..tools import ToolManager

# SGLang 后端初始化（在模块加载时执行一次）
sgl.set_default_backend(sgl.RuntimeEndpoint("http://127.0.0.1:30000"))


class Intent(str, Enum):
    PLAN_TRIP = "plan_trip"
    QUERY_WEATHER = "query_weather"
    SEARCH_ATTRACTION = "search_attraction"
    SEARCH_HOTEL = "search_hotel"
    MODIFY_PLAN = "modify_plan"
    CONFIRM_PLAN = "confirm_plan"
    HELP = "help"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


# 旅游规划助手系统提示基础部分（固定内容，最大化前缀缓存收益）
_BASE_SYSTEM_PROMPT = """你是专业、友好的旅游规划助手。

职责:
1. 理解用户的旅游需求
2. 主动收集必要信息（目的地、日期、人数等）
3. 在合适时机调用工具查询实时信息
4. 生成个性化的旅游规划方案
5. 根据用户反馈调整和优化方案

对话原则:
- 自然友好，像朋友间交流
- 主动询问缺失信息，但一次只问一个问题
- 提供有价值的旅行建议和注意事项"""


class DialogueManager:
    """
    对话管理器。
    不持有 @sgl.function，所有 SGLang 函数在 dialogue_functions.py 模块顶层定义。
    """

    def __init__(self, tool_manager: ToolManager) -> None:
        self.tool_manager = tool_manager
        self.tool_chain = ToolDecisionChain(tool_manager)
        self.context_mgr = ContextWindowManager(
            max_context_tokens=2048,
            window_size=8,
            enable_summarization=True,
        )

    # ── 主入口 ───────────────────────────────────────────────────────

    async def process_user_input(
        self, user_input: str, state: DialogueState
    ) -> str:
        """
        处理用户输入的完整流程（四步）。
        返回助手应回复的文本字符串。
        状态更新（add_turn）在响应生成确认后执行，确保轮次粒度的一致性。
        """
        # 步骤 1: 意图识别（利用最近 3 轮上下文）
        recent = state.get_context_window(max_turns=3)
        context_list = [
            f"{'用户' if t.role=='user' else '助手'}: {t.content[:80]}"
            for t in recent
        ]
        intent, confidence = await self._parse_intent(
            user_input, context_list
        )
        logger.info(
            f"意图识别 | session={state.session_id} "
            f"| intent={intent.value} confidence={confidence:.2f}"
        )

        # 步骤 2: 槽位提取（仅在相关意图且置信度足够时执行）
        new_slots: Dict[str, Any] = {}
        if (
            intent in {Intent.PLAN_TRIP, Intent.MODIFY_PLAN}
            and confidence >= 0.6
        ):
            new_slots = await self._parse_slots(user_input, state.preferences)
            if new_slots:
                changed = state.update_preferences(**new_slots)
                logger.info(f"槽位更新 | 变更项: {changed}")

        # 步骤 3: 根据意图决策下一步行动
        response = await self._decide_action(
            user_input, intent, confidence, state
        )

        # 步骤 4: 更新对话历史
        state.add_turn(
            role="user",
            content=user_input,
            intent=intent.value,
            slots_extracted=new_slots if new_slots else None,
        )
        state.add_turn(role="assistant", content=response)
        return response

    # ── 决策路由 ─────────────────────────────────────────────────────

    async def _decide_action(
        self,
        user_input: str,
        intent: Intent,
        confidence: float,
        state: DialogueState,
    ) -> str:
        """根据意图和当前状态决定具体行动。"""

        if intent == Intent.PLAN_TRIP:
            empty = state.preferences.required_empty_slots()
            if empty:
                # 还有必填槽位未收集，生成上下文相关询问
                return await self._ask_for_slot(empty[0], state)
            else:
                # 信息完整，触发工具查询和方案生成
                return await self._generate_plan(state)

        elif intent == Intent.MODIFY_PLAN:
            # 修改计划：槽位已在步骤 2 更新，重新生成方案
            return await self._generate_plan(state)

        elif intent in {
            Intent.QUERY_WEATHER,
            Intent.SEARCH_ATTRACTION,
            Intent.SEARCH_HOTEL,
        }:
            # 主动查询意图：触发工具调用决策链
            return await self._handle_tool_query(user_input, state)

        elif intent == Intent.CONFIRM_PLAN:
            return (
                "好的！已为您记录这份旅行方案。😊\n"
                "如需预订酒店或查询更多信息，随时告诉我！"
            )

        else:
            # 闲聊、帮助、未知意图：通用对话响应
            return await self._direct_dialogue(user_input, state)

    # ── 子动作实现 ───────────────────────────────────────────────────

    async def _ask_for_slot(
        self, slot_name: str, state: DialogueState
    ) -> str:
        """生成针对特定槽位的自然询问。"""
        recent = state.get_context_window(max_turns=2)
        ctx_text = "; ".join(
            f"{'用户' if t.role=='user' else '助手'}: {t.content[:60]}"
            for t in recent
        )
        try:
            result = ask_missing_slot.run(
                slot_name=slot_name,
                known_info_text=state.preferences.to_summary_text(),
                dialogue_context=ctx_text,
            )
            return result.get("question", f"请问您的{slot_name}是什么？").strip()
        except Exception as exc:
            logger.error(f"ask_missing_slot 失败 | {exc}")
            # 降级：使用固定模板
            fallback_map = {
                "destination": "请问您想去哪里旅游？",
                "start_date": "请问您计划什么时候出发？",
                "end_date": "请问大概玩几天，什么时候返回？",
                "num_travelers": "请问有几个人一起去？",
            }
            return fallback_map.get(slot_name, f"请问您的{slot_name}是什么？")

    async def _handle_tool_query(
        self, user_input: str, state: DialogueState
    ) -> str:
        """处理主动工具查询意图（天气/景点/酒店）。"""
        called, result, err = await self.tool_chain.decide_and_call(
            user_input, state
        )
        if called and result and result.success:
            return self.tool_chain.format_tool_result(result, user_input)
        elif called and result and not result.success:
            return result.message or "查询失败，请稍后再试。"
        else:
            # 工具决策认为不需要工具，或决策本身失败 → 直接对话
            return await self._direct_dialogue(user_input, state)

    async def _generate_plan(self, state: DialogueState) -> str:
        """
        触发工具查询并生成完整旅游方案。
        按顺序查询天气、景点、酒店，汇总结果后调用 generate_travel_plan。
        """
        prefs = state.preferences
        tool_results: Dict[str, Any] = {}

        # 查询天气（如目的地已知且最近 5 轮未查询过）
        if prefs.destination and not state.has_tool_been_called(
            "get_weather"
        ):
            weather = await self.tool_manager.run_tool(
                "get_weather",
                city=prefs.destination,
                date=prefs.start_date,
            )
            if weather.success and weather.data:
                tool_results["weather"] = weather.data
                state.add_tool_call(
                    "get_weather",
                    {"city": prefs.destination},
                    weather.data,
                    True,
                )

        # 查询景点
        if prefs.destination and not state.has_tool_been_called("search_map"):
            attractions = await self.tool_manager.run_tool(
                "search_map",
                city=prefs.destination,
                keyword="景点",
                category=prefs.interests[0] if prefs.interests else None,
            )
            if attractions.success and attractions.data:
                tool_results["attractions"] = attractions.data
                state.add_tool_call(
                    "search_map",
                    {"city": prefs.destination},
                    attractions.data,
                    True,
                )

        # 查询酒店（需要日期信息）
        if (
            prefs.destination
            and prefs.start_date
            and prefs.end_date
            and not state.has_tool_been_called("search_hotel")
        ):
            hotels = await self.tool_manager.run_tool(
                "search_hotel",
                city=prefs.destination,
                check_in=prefs.start_date,
                check_out=prefs.end_date,
                price_range=prefs.accommodation_level or "舒适型",
            )
            if hotels.success and hotels.data:
                tool_results["hotels"] = hotels.data
                state.add_tool_call(
                    "search_hotel",
                    {"city": prefs.destination},
                    hotels.data,
                    True,
                )

        # 获取对话上下文
        window = await self.context_mgr.get_optimal_context_async(state)
        ctx_text = "; ".join(
            f"{'用户' if t.role=='user' else '助手'}: {t.content[:60]}"
            for t in window.turns[-4:]
        )

        try:
            result = generate_travel_plan.run(
                preferences_json=prefs.model_dump_json(exclude_none=True),
                tool_results_json=json.dumps(
                    tool_results, ensure_ascii=False
                ),
                dialogue_context=ctx_text,
            )
            plan = result.get("plan", "").strip()
            if not plan:
                raise ValueError("方案生成结果为空")
            return plan
        except Exception as exc:
            logger.error(f"generate_travel_plan 失败 | {exc}")
            return (
                f"好的，我来为您规划{prefs.destination or ''}的旅行！\n"
                "（方案生成遇到问题，请稍后重试或重新描述您的需求）"
            )

    async def _direct_dialogue(
        self, user_input: str, state: DialogueState
    ) -> str:
        """通用对话响应，用于闲聊、帮助、未知意图等场景。"""
        window = await self.context_mgr.get_optimal_context_async(state)
        history_msgs = self.context_mgr.build_history_messages(window)
        system_prompt = self.build_system_prompt(
            state, tool_result_context=""
        )
        try:
            result = generate_direct_reply.run(
                system_prompt=system_prompt,
                user_input=user_input,
                history_messages=history_msgs,
            )
            return result.get("response", "").strip()
        except Exception as exc:
            logger.error(f"generate_direct_reply 失败 | {exc}")
            return "不好意思，我暂时无法回答这个问题，请换个方式试试？"

    # ── 辅助方法 ─────────────────────────────────────────────────────

    async def _parse_intent(
        self, user_input: str, context_list: List[str]
    ) -> Tuple[Intent, float]:
        """调用 recognize_intent，解析并返回 (Intent, confidence)。"""
        try:
            result = recognize_intent.run(
                user_input=user_input,
                recent_context=context_list,
            )
            raw = result.get("intent_json", "{}").strip()
            raw = _clean_llm_json(raw)
            data = json.loads(raw)
            intent_str = data.get("intent", "unknown").lower().strip()
            confidence = float(data.get("confidence", 0.5))
            try:
                return Intent(intent_str), confidence
            except ValueError:
                return Intent.UNKNOWN, confidence
        except Exception as exc:
            logger.error(f"意图识别失败 | {exc}")
            return Intent.UNKNOWN, 0.3

    async def _parse_slots(
        self, user_input: str, prefs: TravelPreferences
    ) -> Dict[str, Any]:
        """调用 extract_slots，解析并返回槽位字典。"""
        try:
            result = extract_slots.run(
                user_input=user_input,
                current_prefs_json=prefs.model_dump_json(exclude_none=True),
            )
            raw = result.get("slots_json", "{}").strip()
            raw = _clean_llm_json(raw)
            slots = json.loads(raw)
            # 类型修正：确保 num_travelers 为整数
            if "num_travelers" in slots:
                try:
                    slots["num_travelers"] = int(slots["num_travelers"])
                except (ValueError, TypeError):
                    del slots["num_travelers"]
            return slots
        except Exception as exc:
            logger.error(f"槽位提取失败 | {exc}")
            return {}

    def build_system_prompt(
        self, state: DialogueState, tool_result_context: str = ""
    ) -> str:
        """
        构建系统提示。
        固定部分（_BASE_SYSTEM_PROMPT）在提示开头，最大化前缀缓存命中率。
        动态部分（槽位状态、工具结果）在末尾，避免破坏前缀一致性。
        """
        parts = [_BASE_SYSTEM_PROMPT]

        # 槽位状态摘要（动态部分）
        slot_summary = state.preferences.to_summary_text()
        if slot_summary != "（暂无已确认信息）":
            parts.append(f"\n当前已确认信息:\n{slot_summary}")

        # 待收集的必填槽位提示
        empty = state.preferences.required_empty_slots()
        if empty:
            parts.append(f"\n待收集信息: {', '.join(empty)}")

        # 工具查询结果（如有）
        if tool_result_context:
            parts.append(f"\n工具查询结果:\n{tool_result_context}")

        return "\n".join(parts)


def _clean_llm_json(raw: str) -> str:
    """清理 LLM 输出中的 Markdown 代码块标记。"""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner).strip()
    return raw
