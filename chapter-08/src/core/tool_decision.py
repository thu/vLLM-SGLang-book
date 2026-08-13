"""
src/core/tool_decision.py
五步工具调用决策链路。

五步流程:
  1. should_call_tool  - 判断是否需要工具（LLM）
  2. select_tool       - 选择最合适的工具（LLM）
  3. extract_tool_args - 提取调用参数（LLM，最多重试 2 次）
  4. 参数 Pydantic 验证（规则）
  5. call_with_fallback - 执行工具（BaseTool 内置重试）

软失败策略: 任何步骤失败均提前返回 (False, None, err_msg)，
不影响对话主流程，调用方生成降级响应。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import ValidationError

from .dialogue_functions import (
    extract_tool_args,
    select_tool,
    should_call_tool,
)
from .state import DialogueState
from ..tools import ToolManager, ToolOutput


def _clean_json_output(raw: str) -> str:
    """
    清理 LLM 输出中可能包含的 Markdown 代码块标记。
    模型有时会输出 ```json ... ``` 格式，需要提取其中的 JSON。
    """
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # 移除首行（```json 或 ```）和末行（```）
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner).strip()
    return raw


class ToolDecisionChain:
    """
    工具调用决策链。
    对话管理器持有此对象，通过 decide_and_call() 触发完整五步流程。
    """

    def __init__(self, tool_manager: ToolManager) -> None:
        self.tool_manager = tool_manager

    async def decide_and_call(
        self,
        user_input: str,
        state: DialogueState,
    ) -> Tuple[bool, Optional[ToolOutput], Optional[str]]:
        """
        完整的五步工具调用决策与执行。
        返回: (是否执行了工具调用, 工具结果, 错误信息)
        任何步骤失败均提前返回，不抛异常。
        """
        # 构建上下文摘要（最近 2 轮）
        recent = state.get_context_window(max_turns=2)
        ctx_summary = "; ".join(
            f"{'用户' if t.role=='user' else '助手'}: {t.content[:60]}"
            for t in recent
        )

        # ── 步骤 1: 判断是否需要工具 ─────────────────────────────────
        tools_desc = self.tool_manager.get_tools_description()
        try:
            result = should_call_tool.run(
                user_input=user_input,
                context_summary=ctx_summary,
                available_tools_desc=tools_desc,
            )
            decision = result.get("decision", "NO").strip().upper()
        except Exception as exc:
            logger.error(f"should_call_tool 失败 | {exc}")
            return False, None, str(exc)

        if "NO" in decision:
            return False, None, None

        # ── 步骤 2: 选择工具 ─────────────────────────────────────────
        schemas_json = self.tool_manager.get_schemas_json()
        try:
            sel_result = select_tool.run(
                user_input=user_input,
                available_tools_json=schemas_json,
                context_summary=ctx_summary,
            )
            tool_name = sel_result.get("tool_name", "").strip()
        except Exception as exc:
            logger.error(f"select_tool 失败 | {exc}")
            return False, None, str(exc)

        schema_json = self.tool_manager.get_tool_schema_json(tool_name)
        if schema_json is None:
            # 工具名可能有多余字符，尝试模糊匹配
            tool_name = self._fuzzy_match_tool(tool_name)
            schema_json = self.tool_manager.get_tool_schema_json(tool_name)
            if schema_json is None:
                logger.warning(f"工具不存在: {tool_name}")
                return False, None, f"工具 '{tool_name}' 不存在"

        logger.info(f"选择工具: {tool_name}")

        # ── 步骤 3+4: 提取并验证参数（最多重试 2 次） ────────────────
        args = await self._extract_and_validate(
            tool_name, schema_json, user_input, state, max_retries=2
        )
        if args is None:
            return True, None, "参数提取失败"

        # ── 步骤 5: 执行工具 ─────────────────────────────────────────
        tool_result = await self.tool_manager.run_tool(tool_name, **args)
        state.add_tool_call(
            tool_name=tool_name,
            arguments=args,
            result=tool_result.data,
            success=tool_result.success,
            error=tool_result.error,
        )
        return True, tool_result, None

    async def _extract_and_validate(
        self,
        tool_name: str,
        schema_json: str,
        user_input: str,
        state: DialogueState,
        max_retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """
        提取参数并进行 Pydantic 类型验证，失败时重试。
        重试时将上一次的错误信息作为额外上下文传入，引导模型修正。
        """
        context_state = json.dumps(
            state.preferences.model_dump(exclude_none=True),
            ensure_ascii=False,
        )
        last_error = ""

        for attempt in range(max_retries + 1):
            input_hint = (
                f"{user_input}\n[上次提取错误，请修正: {last_error}]"
                if last_error
                else user_input
            )
            try:
                result = extract_tool_args.run(
                    tool_name=tool_name,
                    tool_schema_json=schema_json,
                    user_input=input_hint,
                    context_state_json=context_state,
                )
                raw_json = _clean_json_output(result.get("args_json", "{}"))
                args = json.loads(raw_json)

                # 基础类型验证：必填字段检查
                schema = json.loads(schema_json)
                required = schema.get("parameters", {}).get("required", [])
                missing = [k for k in required if k not in args]
                if missing:
                    last_error = f"缺少必填参数: {missing}"
                    logger.warning(
                        f"参数验证失败（尝试{attempt+1}）| {last_error}"
                    )
                    continue

                logger.info(f"参数提取成功 | {tool_name} | {args}")
                return args

            except (json.JSONDecodeError, Exception) as exc:
                last_error = str(exc)
                logger.warning(
                    f"参数提取失败（尝试{attempt+1}）| {exc}"
                )

        logger.error(f"参数提取失败（{max_retries+1}次重试耗尽）| {tool_name}")
        return None

    def _fuzzy_match_tool(self, raw_name: str) -> str:
        """
        对模型输出的工具名进行模糊匹配，容忍少量多余字符。
        例如 '"get_weather"' → 'get_weather'
        """
        raw_lower = raw_name.lower().strip('"\'` ')
        known_tools = ["get_weather", "search_map", "search_hotel"]
        for tool in known_tools:
            if tool in raw_lower:
                return tool
        return raw_name

    def format_tool_result(
        self,
        tool_result: ToolOutput,
        user_input: str,
    ) -> str:
        """
        将工具结果格式化为用户友好的自然语言文本。
        根据工具名称选择对应的格式化模板。
        """
        if not tool_result.success:
            return tool_result.message or "查询失败，请稍后再试。"

        data = tool_result.data or {}
        tool_name = ""
        # 根据 data 结构推断工具类型
        if "weather" in data:
            tool_name = "get_weather"
        elif "pois" in data:
            tool_name = "search_map"
        elif "hotels" in data:
            tool_name = "search_hotel"

        if tool_name == "get_weather":
            return (
                f"**{data.get('city', '')}天气查询结果**\n"
                f"天气: {data.get('weather', '')}\n"
                f"温度: {data.get('temperature', '')}\n"
                f"湿度: {data.get('humidity', '')}\n"
                f"出行建议: {data.get('suggestion', '')}\n"
                + (f"备注: {data.get('note', '')}" if data.get("note") else "")
            )
        elif tool_name == "search_map":
            pois = data.get("pois", [])
            lines = [f"**{data.get('city', '')}景点推荐**（共{len(pois)}处）\n"]
            for i, p in enumerate(pois[:4], 1):
                lines.append(
                    f"{i}. {p['name']} ⭐{p['rating']}\n"
                    f"   {p.get('description', '')}\n"
                    f"   地址: {p.get('address', '')} | 价格: {p.get('price', '未知')}\n"
                )
            return "\n".join(lines)
        elif tool_name == "search_hotel":
            hotels = data.get("hotels", [])
            nights = data.get("nights", 1)
            lines = [
                f"**{data.get('city', '')}{data.get('price_range', '')}酒店推荐**"
                f"（{nights}晚）\n"
            ]
            for i, h in enumerate(hotels[:3], 1):
                lines.append(
                    f"{i}. {h['name']} ⭐{h['rating']}\n"
                    f"   价格: {h['price']}元/晚 | {nights}晚合计: {h['total_price']}元\n"
                    f"   设施: {', '.join(h.get('facilities', [])[:4])}\n"
                )
            return "\n".join(lines)
        else:
            return tool_result.message or str(data)
