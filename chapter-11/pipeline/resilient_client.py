# pipeline/resilient_client.py
# 显存压力感知的降级策略客户端（ch11_code_09）
#
# 三级降级策略：
#   NORMAL  （<75%）：正常运行，优先使用 SGLang RadixAttention
#   SOFT    （75–90%）：降低并发度，继续使用 SGLang
#   HARD    （90–95%）：切换至 vLLM，利用其队列机制
#   CRITICAL（>=95%）：指数退避等待，超时后放弃请求
#
# 注意：asyncio.Semaphore 的内部计数器不支持在创建后动态
# 修改，因此并发度调整通过替换信号量对象实现。替换时需
# 等待当前持有信号量的协程自然释放，不强制中断进行中的请求。

from __future__ import annotations

import asyncio
from enum import Enum

import httpx
from openai import AsyncOpenAI

from config.model_config import MODEL_CONFIG
from inference.sglang_server import get_sglang_base_url
from inference.vllm_server import get_vllm_base_url


class SystemPressureLevel(str, Enum):
    NORMAL   = "normal"    # 显存使用率 < 75%
    SOFT     = "soft"      # 75% <= 使用率 < 90%
    HARD     = "hard"      # 90% <= 使用率 < 95%
    CRITICAL = "critical"  # 使用率 >= 95%


# 各压力等级对应的最大并发文书数
_CONCURRENCY_BY_PRESSURE: dict[SystemPressureLevel, int] = {
    SystemPressureLevel.NORMAL:   8,
    SystemPressureLevel.SOFT:     6,
    SystemPressureLevel.HARD:     4,
    SystemPressureLevel.CRITICAL: 2,
}


class ResilientInferenceClient:
    """
    显存压力感知的弹性推理客户端。

    在正常和软边界状态下使用 SGLang（RadixAttention 命中率高）；
    在硬边界状态下切换至 vLLM（其队列机制对显存压力更保守）；
    在临界状态下采用指数退避等待显存压力缓解。

    并发度调整通过替换信号量对象实现。由于 asyncio.Semaphore
    不支持动态修改内部计数，每次调整会创建新的信号量对象，
    正在等待旧信号量的协程继续等待旧对象，不受影响；
    新到达的请求使用新信号量，并发限制在新请求上生效。
    """

    # Prometheus 查询地址
    PROMETHEUS_URL = "http://localhost:9090"

    def __init__(
        self,
        sglang_url: str | None = None,
        vllm_url: str | None = None,
    ) -> None:
        self._sglang_url = sglang_url or get_sglang_base_url()
        self._vllm_url = vllm_url or get_vllm_base_url()
        self._model = MODEL_CONFIG["model_path"].split("/")[-1]
        # 初始信号量：正常压力下的并发上限
        self._semaphore = asyncio.Semaphore(
            _CONCURRENCY_BY_PRESSURE[SystemPressureLevel.NORMAL]
        )
        self._current_pressure = SystemPressureLevel.NORMAL

    def _sglang_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self._sglang_url, api_key="EMPTY"
        )

    def _vllm_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self._vllm_url, api_key="EMPTY"
        )

    async def get_pressure_level(self) -> SystemPressureLevel:
        """
        从 Prometheus 查询当前 GPU 平均显存使用率，
        返回对应的压力等级。

        若 Prometheus 不可达（如开发环境），默认返回 NORMAL。
        """
        query = "avg(memory.used / memory.total)"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(
                    f"{self.PROMETHEUS_URL}/api/v1/query",
                    params={"query": query},
                )
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            if not results:
                return SystemPressureLevel.NORMAL
            ratio = float(results[0]["value"][1])
        except Exception:
            # Prometheus 不可达时不阻断业务，保守返回 NORMAL
            return SystemPressureLevel.NORMAL

        if ratio < 0.75:
            return SystemPressureLevel.NORMAL
        if ratio < 0.90:
            return SystemPressureLevel.SOFT
        if ratio < 0.95:
            return SystemPressureLevel.HARD
        return SystemPressureLevel.CRITICAL

    def _adjust_semaphore(
        self, pressure: SystemPressureLevel
    ) -> None:
        """
        根据压力等级调整并发信号量。

        仅在压力等级发生变化时替换信号量对象，
        避免不必要的对象创建。
        """
        if pressure == self._current_pressure:
            return
        new_limit = _CONCURRENCY_BY_PRESSURE[pressure]
        self._semaphore = asyncio.Semaphore(new_limit)
        self._current_pressure = pressure
        print(
            f"[弹性客户端] 压力等级变更为 {pressure.value}，"
            f"并发上限调整为 {new_limit}"
        )

    async def _call_backend(
        self,
        client: AsyncOpenAI,
        content: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """向指定后端发起单次推理请求，返回生成文本。"""
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def generate(
        self,
        content: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        压力感知的弹性推理入口。

        路由逻辑：
          NORMAL / SOFT → SGLang（主路由），失败后降级至 vLLM
          HARD          → 直接路由至 vLLM
          CRITICAL      → 指数退避（1→2→4→8 秒），
                          压力缓解后路由至 vLLM，
                          四次重试后仍为 CRITICAL 则放弃
        """
        pressure = await self.get_pressure_level()
        self._adjust_semaphore(pressure)

        async with self._semaphore:
            if pressure in (
                SystemPressureLevel.NORMAL,
                SystemPressureLevel.SOFT,
            ):
                try:
                    return await self._call_backend(
                        self._sglang_client(),
                        content, max_tokens, temperature,
                    )
                except Exception as exc:
                    print(
                        f"[弹性客户端] SGLang 调用失败，"
                        f"降级至 vLLM：{exc}"
                    )
                    return await self._call_backend(
                        self._vllm_client(),
                        content, max_tokens, temperature,
                    )

            if pressure == SystemPressureLevel.HARD:
                return await self._call_backend(
                    self._vllm_client(),
                    content, max_tokens, temperature,
                )

            # CRITICAL：指数退避
            for wait_s in [1, 2, 4, 8]:
                await asyncio.sleep(wait_s)
                current = await self.get_pressure_level()
                if current != SystemPressureLevel.CRITICAL:
                    return await self._call_backend(
                        self._vllm_client(),
                        content, max_tokens, temperature,
                    )
            raise RuntimeError(
                "系统持续处于临界显存压力（>95%），请求已放弃。"
                "请降低并发度或减少缓存中的文书数量。"
            )
