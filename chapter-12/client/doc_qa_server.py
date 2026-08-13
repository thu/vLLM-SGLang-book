#!/usr/bin/env python3
"""
第12章：长文档问答 API 服务
基于 vLLM OpenAI 兼容接口，面向法律/技术文档批量分析场景
版本基线：vLLM v0.8.x，Python 3.11+
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import aiohttp
from aiohttp import web


# ──────────────────────────────────────────────
# 配置常量
# ──────────────────────────────────────────────
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "deepseek-v4-flash"
MAX_TOKENS_ANSWER = 2048
REQUEST_TIMEOUT_SEC = 300       # 长文档推理超时 5 分钟

# 法律文档分析提示词模板
LEGAL_SYSTEM_PROMPT = """你是一位专业的法律文档分析助手。
请基于用户提供的合同文本，准确、客观地回答问题。
若文本中无相关信息，请明确说明"文本中未提及"，不要推测。
回答使用中文，条理清晰，必要时引用原文段落。"""

TECH_SYSTEM_PROMPT = """你是一位技术文档分析专家。
请基于用户提供的技术文档，准确提取所需信息并给出结构化回答。
优先引用文档中的原始数据和结论，避免主观推断。"""


# ──────────────────────────────────────────────
# 请求/响应数据模型（轻量实现，不依赖 Pydantic）
# ──────────────────────────────────────────────
def make_chat_payload(
    document: str,
    question: str,
    doc_type: str = "legal",
    max_tokens: int = MAX_TOKENS_ANSWER,
    thinking_mode: str = "non-think",   # V4 支持 non-think / think-high / think-max
) -> dict:
    """
    构造发往 vLLM 的 chat completions 请求体。
    thinking_mode 对应 DeepSeek V4 三档推理模式。
    """
    system_prompt = (
        LEGAL_SYSTEM_PROMPT if doc_type == "legal" else TECH_SYSTEM_PROMPT
    )
    # V4 thinking mode 通过特殊 system token 控制
    thinking_prefix = {
        "non-think": "",
        "think-high": "<think_high>",
        "think-max": "<think_max>",
    }.get(thinking_mode, "")

    user_content = (
        f"{thinking_prefix}"
        f"【文档内容】\n{document}\n\n"
        f"【问题】\n{question}"
    )
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,     # 文档分析场景低温度，减少幻觉
        "stream": False,
    }


# ──────────────────────────────────────────────
# vLLM 客户端：异步批量请求
# ──────────────────────────────────────────────
class VLLMClient:
    """封装对 vLLM OpenAI API 的异步调用。"""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        max_concurrency: int = 16,
    ):
        self.base_url = base_url
        # 用信号量控制并发，避免 OOM
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def chat(
        self, payload: dict, request_id: str = ""
    ) -> dict:
        """
        发送单条 chat completions 请求，返回解析后的结果字典。
        包含延迟指标：ttft（首 token 延迟）和 total_latency。
        """
        async with self._semaphore:
            t0 = time.perf_counter()
            async with self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                total_latency = time.perf_counter() - t0

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return {
            "request_id": request_id or str(uuid.uuid4())[:8],
            "answer": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_latency_s": total_latency,
            # vLLM 在 usage 中返回 time_to_first_token（ms）
            "ttft_ms": usage.get("time_to_first_token_ms", None),
        }

    async def batch_chat(
        self,
        tasks: list[dict],    # 每个元素含 document/question/doc_type
        thinking_mode: str = "non-think",
    ) -> list[dict]:
        """
        并发处理批量文档问答任务，保序返回结果列表。
        """
        coros = [
            self.chat(
                make_chat_payload(
                    t["document"], t["question"],
                    t.get("doc_type", "legal"),
                    thinking_mode=thinking_mode,
                ),
                request_id=t.get("id", str(i)),
            )
            for i, t in enumerate(tasks)
        ]
        return await asyncio.gather(*coros, return_exceptions=True)


# ──────────────────────────────────────────────
# HTTP API 服务（aiohttp）
# ──────────────────────────────────────────────
async def handle_analyze(request: web.Request) -> web.Response:
    """
    POST /analyze
    请求体：{"document": "...", "question": "...",
             "doc_type": "legal|tech", "thinking_mode": "non-think"}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "invalid JSON"}, status=400
        )

    required = {"document", "question"}
    if missing := required - set(body):
        return web.json_response(
            {"error": f"缺少必填字段: {missing}"}, status=400
        )

    client: VLLMClient = request.app["vllm_client"]
    payload = make_chat_payload(
        document=body["document"],
        question=body["question"],
        doc_type=body.get("doc_type", "legal"),
        thinking_mode=body.get("thinking_mode", "non-think"),
    )
    try:
        result = await client.chat(payload)
    except aiohttp.ClientResponseError as e:
        return web.json_response(
            {"error": f"vLLM 服务错误: {e.status} {e.message}"},
            status=502,
        )
    return web.json_response(result)


async def handle_batch_analyze(request: web.Request) -> web.Response:
    """
    POST /batch_analyze
    请求体：{"tasks": [...], "thinking_mode": "non-think"}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid JSON"}, status=400)

    tasks = body.get("tasks", [])
    if not tasks:
        return web.json_response({"error": "tasks 不能为空"}, status=400)

    client: VLLMClient = request.app["vllm_client"]
    results = await client.batch_chat(
        tasks, thinking_mode=body.get("thinking_mode", "non-think")
    )

    # 区分成功与失败结果
    successes, errors = [], []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
        else:
            successes.append(r)

    return web.json_response({
        "total": len(tasks),
        "success": len(successes),
        "failed": len(errors),
        "results": successes,
        "errors": errors,
    })


async def handle_health(request: web.Request) -> web.Response:
    """GET /health  服务健康检查"""
    return web.json_response({"status": "ok", "model": MODEL_NAME})


# ──────────────────────────────────────────────
# 应用生命周期管理
# ──────────────────────────────────────────────
async def on_startup(app: web.Application) -> None:
    max_concurrency = app["max_concurrency"]
    client = VLLMClient(max_concurrency=max_concurrency)
    await client.__aenter__()
    app["vllm_client"] = client
    print(f"[startup] VLLMClient 初始化完成，最大并发={max_concurrency}")


async def on_cleanup(app: web.Application) -> None:
    client = app.get("vllm_client")
    if client:
        await client.__aexit__(None, None, None)
        print("[cleanup] VLLMClient 已关闭")


def create_app(max_concurrency: int = 16) -> web.Application:
    app = web.Application()
    app["max_concurrency"] = max_concurrency
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/analyze", handle_analyze)
    app.router.add_post("/batch_analyze", handle_batch_analyze)
    app.router.add_get("/health", handle_health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="长文档问答 API 服务"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--max-concurrency", type=int, default=16,
        help="向 vLLM 的最大并发请求数"
    )
    parser.add_argument(
        "--vllm-url", default=VLLM_BASE_URL,
        help="vLLM 服务地址"
    )
    args = parser.parse_args()

    global VLLM_BASE_URL
    VLLM_BASE_URL = args.vllm_url

    app = create_app(max_concurrency=args.max_concurrency)
    print(f"长文档问答服务启动：http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
