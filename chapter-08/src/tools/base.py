"""
src/tools/base.py
工具基类：统一接口、重试机制、软失败策略。

设计原则:
- run() 永远不抛异常，失败时返回 success=False 的 ToolOutput
- 失败时 message 字段包含用户友好的降级提示，不是技术性错误堆栈
- 重试策略: 指数退避，最多 3 次，初始 2 秒，最大 10 秒
- use_mock=True 时使用模拟数据，结构与真实 API 一致
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
)


# ────────────────────────────────────────────────────────────────────
# 统一输出格式
# ────────────────────────────────────────────────────────────────────

class ToolOutput(BaseModel):
    """
    工具调用的统一返回格式。
    - success=True:  data 含结构化结果，message 含用户可读摘要
    - success=False: error 含内部错误（仅用于日志），message 含降级提示
    message 字段在两种情况下均存在，可直接传给对话管理器用于生成响应。
    """

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


# ────────────────────────────────────────────────────────────────────
# 工具基类
# ────────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    """
    所有工具的基类。
    子类需实现:
        _execute(**kwargs) -> ToolOutput   (核心逻辑)
        _get_parameters_schema() -> dict   (供工具决策的 LLM 函数使用)
        _get_fallback_message() -> str     (失败时的降级提示)
    """

    def __init__(
        self,
        name: str,
        description: str,
        timeout: int = 10,
        use_mock: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self.timeout = timeout
        self.use_mock = use_mock
        # 异步 HTTP 客户端，供子类的真实 API 调用使用
        self._client = httpx.AsyncClient(timeout=timeout)

    # ── 对外接口 ────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    async def run(self, **kwargs: Any) -> ToolOutput:
        """
        工具调用的统一入口，包含重试机制。
        保证: 永远不抛出异常，失败时返回 success=False 的 ToolOutput。
        重试参数选择依据: 绝大多数瞬时故障（网络抖动、临时限流）
        在 2-6 秒内可自愈；超过 10 秒的等待影响对话体验，故以 10 秒为上限。
        """
        try:
            logger.info(f"工具调用开始 | {self.name} | 参数: {kwargs}")
            result = await self._execute(**kwargs)
            log_fn = logger.info if result.success else logger.warning
            log_fn(
                f"工具调用完成 | {self.name} | "
                f"success={result.success}"
                + (f" | error={result.error}" if result.error else "")
            )
            return result
        except RetryError as exc:
            logger.error(
                f"工具调用失败（重试耗尽）| {self.name} | {exc}"
            )
            return ToolOutput(
                success=False,
                error=str(exc),
                message=self._get_fallback_message(),
            )
        except Exception as exc:
            logger.error(f"工具调用异常 | {self.name} | {exc}")
            return ToolOutput(
                success=False,
                error=str(exc),
                message=self._get_fallback_message(),
            )

    def get_schema(self) -> Dict[str, Any]:
        """返回工具的 JSON Schema，供工具决策的 LLM 函数理解工具用途和参数。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    async def close(self) -> None:
        """关闭底层 HTTP 客户端，在应用关闭时调用。"""
        await self._client.aclose()

    # ── 子类需实现的方法 ─────────────────────────────────────────────

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolOutput:
        """核心执行逻辑。use_mock=True 时返回模拟数据，否则调用真实 API。"""

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """返回参数的 JSON Schema，供 LLM 理解调用方式。"""

    @abstractmethod
    def _get_fallback_message(self) -> str:
        """
        失败时的降级提示。
        要求: 包含基于历史知识的参考信息，而非空洞的'请重试'。
        """
