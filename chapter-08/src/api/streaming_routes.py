"""
src/api/streaming_routes.py
FastAPI 流式对话端点，使用 Server-Sent Events (SSE) 向客户端实时推送。

设计决策:
  工具调用在流式生成之前完成（非并行）。
  这意味着工具调用延迟会叠加在用户等待首字的时间上，
  但避免了流式输出中途停顿的体验问题（稳定性 > 极致流畅度）。
  工具调用超时设置为 3 秒（而非 BaseTool 默认的 10 秒），
  以控制最坏情况下的等待时间。
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from ..core.dialogue_manager import DialogueManager, Intent, _clean_llm_json
from ..core.session_manager import SessionManager
from ..core.streaming_manager import StreamingManager
from ..core.tool_decision import ToolDecisionChain
from ..tools import ToolManager

router = APIRouter(prefix="/api/v1", tags=["dialogue"])

# 组件实例（由 app.py 的 lifespan 注入）
_session_mgr: Optional[SessionManager] = None
_dialogue_mgr: Optional[DialogueManager] = None
_streaming_mgr: Optional[StreamingManager] = None
_tool_chain: Optional[ToolDecisionChain] = None


def init_components(
    session_mgr: SessionManager,
    dialogue_mgr: DialogueManager,
    streaming_mgr: StreamingManager,
    tool_chain: ToolDecisionChain,
) -> None:
    """由 app.py 的 lifespan 在启动时调用，注入组件实例。"""
    global _session_mgr, _dialogue_mgr, _streaming_mgr, _tool_chain
    _session_mgr = session_mgr
    _dialogue_mgr = dialogue_mgr
    _streaming_mgr = streaming_mgr
    _tool_chain = tool_chain


# ── 请求/响应模型 ─────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    user_id: str


class NewSessionResponse(BaseModel):
    session_id: str
    message: str


class StreamRequest(BaseModel):
    session_id: str
    message: str


# ── 端点定义 ─────────────────────────────────────────────────────────

@router.post("/sessions", response_model=NewSessionResponse)
async def create_session(req: NewSessionRequest):
    """创建新的对话会话，返回 session_id。"""
    state = _session_mgr.create_session(user_id=req.user_id)
    return NewSessionResponse(
        session_id=state.session_id,
        message="会话已创建，请开始对话！",
    )


@router.post("/sessions/{session_id}/stream")
async def stream_dialogue(session_id: str, req: StreamRequest):
    """
    流式对话接口。
    SSE 事件类型:
      start  - 对话开始
      chunk  - 文本片段（content 字段）
      end    - 对话结束（含 turn_count）
      error  - 发生错误（含 message）
    """
    if session_id != req.session_id:
        raise HTTPException(400, "session_id 不匹配")

    state = _session_mgr.load_state(session_id)
    if state is None:
        raise HTTPException(404, detail=f"会话不存在或已过期: {session_id}")

    async def generate():
        full_response = ""
        try:
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # 阶段 1: 意图识别和槽位提取（非流式，约 200ms）
            recent = state.get_context_window(max_turns=3)
            context_list = [
                f"{'用户' if t.role=='user' else '助手'}: {t.content[:80]}"
                for t in recent
            ]
            intent, confidence = await _dialogue_mgr._parse_intent(
                req.message, context_list
            )
            new_slots = {}
            if (
                intent in {Intent.PLAN_TRIP, Intent.MODIFY_PLAN}
                and confidence >= 0.6
            ):
                new_slots = await _dialogue_mgr._parse_slots(
                    req.message, state.preferences
                )
                if new_slots:
                    state.update_preferences(**new_slots)

            # 阶段 2: 工具调用（非流式，0-3000ms）
            tool_result_context = ""
            if intent in {
                Intent.QUERY_WEATHER,
                Intent.SEARCH_ATTRACTION,
                Intent.SEARCH_HOTEL,
            } or (intent == Intent.PLAN_TRIP and state.preferences.is_complete()):
                called, result, _ = await _tool_chain.decide_and_call(
                    req.message, state
                )
                if called and result and result.success and result.data:
                    tool_result_context = json.dumps(
                        result.data, ensure_ascii=False
                    )

            # 阶段 3: 流式生成最终响应
            system_prompt = _dialogue_mgr.build_system_prompt(
                state, tool_result_context
            )
            window = await _dialogue_mgr.context_mgr.get_optimal_context_async(
                state
            )
            history_msgs = _dialogue_mgr.context_mgr.build_history_messages(window)

            async for chunk in _streaming_mgr.stream_with_buffer(
                system_prompt=system_prompt,
                user_input=req.message,
                history_messages=history_msgs,
                buffer_size=5,
            ):
                full_response += chunk
                event = json.dumps(
                    {"type": "chunk", "content": chunk},
                    ensure_ascii=False,
                )
                yield f"data: {event}\n\n"

            # 阶段 4: 更新并持久化状态
            state.add_turn(
                "user", req.message,
                intent=intent.value,
                slots_extracted=new_slots if new_slots else None,
            )
            state.add_turn("assistant", full_response)
            _session_mgr.save_state(state)

            end_event = json.dumps({
                "type": "end",
                "turn_count": state.turn_count,
            })
            yield f"data: {end_event}\n\n"

        except Exception as exc:
            logger.error(
                f"流式对话异常 | session={session_id} | {exc}"
            )
            err_event = json.dumps({
                "type": "error",
                "message": str(exc),
            })
            yield f"data: {err_event}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    """查询会话状态（用于前端展示槽位填充进度）。"""
    state = _session_mgr.load_state(session_id)
    if state is None:
        raise HTTPException(404, "会话不存在")
    prefs = state.preferences
    return {
        "session_id": session_id,
        "turn_count": state.turn_count,
        "preferences": prefs.model_dump(exclude_none=True),
        "filled_slots": prefs.required_empty_slots().__class__(
            [f for f in ["destination","start_date","end_date","num_travelers"]
             if getattr(prefs, f) is not None]
        ),
        "empty_required_slots": prefs.required_empty_slots(),
        "completion_rate": prefs.completion_rate(),
        "is_complete": prefs.is_complete(),
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话（用户主动退出时调用）。"""
    deleted = _session_mgr.delete_state(session_id)
    if not deleted:
        raise HTTPException(404, "会话不存在")
    return {"message": f"会话 {session_id} 已删除"}
