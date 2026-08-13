# inference/base_client.py
# 统一推理客户端接口
#
# 封装 openai.AsyncOpenAI，提供统一的流式与非流式调用接口，
# 并在响应中附加时间戳供上层监控使用。

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from config.model_config import MODEL_CONFIG


@dataclass
class InferenceResult:
    """单次推理结果。"""
    request_id: str
    content: str
    # 首 token 延迟（毫秒）
    ttft_ms: float
    # 端到端延迟（毫秒）
    e2e_ms: float
    # 输入 token 数（由模型计数）
    prompt_tokens: int
    # 输出 token 数
    completion_tokens: int


class InferenceClient:
    """
    基于 OpenAI 兼容接口的异步推理客户端。

    支持 SGLang 和 vLLM 的统一调用，通过 base_url 区分后端。
    """

    def __init__(self, base_url: str, api_key: str = "EMPTY"):
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._model = MODEL_CONFIG["model_path"].split("/")[-1]

    async def generate(
        self,
        content: str,
        request_id: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> InferenceResult:
        """
        非流式生成（等待完整响应）。

        适用于离线批处理场景，延迟测量包含完整的网络往返。
        """
        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        e2e_ms = (time.perf_counter() - start) * 1000

        # 非流式模式下无法直接测量 TTFT，
        # 用 e2e_ms 作为 TTFT 的保守上界
        return InferenceResult(
            request_id=request_id,
            content=response.choices[0].message.content or "",
            ttft_ms=e2e_ms,
            e2e_ms=e2e_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    async def generate_stream(
        self,
        content: str,
        request_id: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> InferenceResult:
        """
        流式生成，精确测量首 token 延迟（TTFT）。

        TTFT 是 prefill 阶段完成后生成第一个有效内容 token
        所需的时间，是衡量缓存命中效果的核心指标。
        """
        start = time.perf_counter()
        first_token_time: float | None = None
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            delta_content = (
                chunk.choices[0].delta.content
                if chunk.choices
                else None
            )
            if delta_content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(delta_content)
            # 流式响应的 usage 在最后一个 chunk 中
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        end = time.perf_counter()
        ttft_ms = (
            ((first_token_time or end) - start) * 1000
        )

        return InferenceResult(
            request_id=request_id,
            content="".join(chunks),
            ttft_ms=ttft_ms,
            e2e_ms=(end - start) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
