"""
src/api/app.py
FastAPI 应用入口，管理组件生命周期。
启动命令: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from loguru import logger

from ..core.dialogue_manager import DialogueManager
from ..core.session_manager import SessionManager
from ..core.streaming_manager import StreamingManager
from ..core.tool_decision import ToolDecisionChain
from ..tools import ToolManager
from .streaming_routes import init_components, router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理：启动时初始化所有组件，关闭时释放资源。"""
    logger.info("=== 旅游规划助手启动 ===")

    # 初始化工具管理器（开发阶段使用模拟数据）
    tool_mgr = ToolManager(use_mock=True)

    # 初始化 Redis 会话管理器
    session_mgr = SessionManager(
        redis_host="localhost",
        redis_port=6379,
        default_ttl=1800,
    )

    # 初始化对话管理器
    dialogue_mgr = DialogueManager(tool_manager=tool_mgr)

    # 初始化流式输出管理器
    streaming_mgr = StreamingManager(backend_url="http://127.0.0.1:30000")

    # 初始化工具决策链
    tool_chain = ToolDecisionChain(tool_manager=tool_mgr)

    # 注入组件到路由模块
    init_components(session_mgr, dialogue_mgr, streaming_mgr, tool_chain)

    logger.info("所有组件初始化完成，服务就绪")
    yield

    # 关闭时释放资源
    await tool_mgr.close_all()
    logger.info("=== 服务关闭 ===")


app = FastAPI(
    title="旅游规划智能助手 API",
    description="基于 SGLang + ChatGLM-3-6B 的多轮对话 Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health_check():
    """健康检查端点，供 systemd 服务和监控系统使用。"""
    return {"status": "ok", "service": "travel-assistant"}
