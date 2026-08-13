#!/usr/bin/env python3
# ch06/gateway/main.py
# FastAPI 推理网关
#
# 依赖版本：FastAPI 0.115.x，Pydantic v2，Uvicorn 0.32.x，httpx 0.27.x
# 运行：uvicorn ch06.gateway.main:app --host 0.0.0.0 --port 8001 --workers 4
#
# 职责：鉴权、限流、Prometheus 指标埋点、请求转发到 vLLM

import time
import hashlib
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ch06.gateway.truncation import truncate_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Prometheus 指标 ----------
# 使用 Histogram 而非 Summary：
# Histogram 支持跨实例聚合（通过 histogram_quantile），
# Summary 的分位数在多实例部署下无法合并。
REQUEST_LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "端到端请求延迟（网关接收到返回响应）",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    labelnames=["status_code"],
)
REQUEST_TOTAL = Counter(
    "gateway_requests_total",
    "请求计数",
    labelnames=["status_code"],
)
# -------------------------------------

VLLM_URL = "http://localhost:8000"

# httpx 连接池：在 2000 QPS 下频繁建立 TCP 连接的开销不可忽视。
# max_connections=512 对应 vLLM 内部队列容量上限；
# max_keepalive_connections=256 维持持久连接减少握手开销。
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        base_url=VLLM_URL,
        limits=httpx.Limits(
            max_connections=512,
            max_keepalive_connections=256,
        ),
        timeout=httpx.Timeout(connect=2.0, read=60.0, write=5.0, pool=2.0),
    )
    logger.info("HTTP 连接池已初始化")
    yield
    await _http_client.aclose()
    logger.info("HTTP 连接池已关闭")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/v1/chat/completions")
@limiter.limit("200/minute")
async def chat_completions(request: Request):
    """
    转发 chat/completions 请求到 vLLM。
    限流设为 200/min（每 IP），对应正常客服用户的合理使用上限。
    """
    body = await request.json()

    # 截断超长输入，防止少量长请求阻塞批处理队列
    if "messages" in body:
        body["messages"] = truncate_messages(body["messages"])

    t0 = time.perf_counter()
    status_code = "unknown"
    try:
        resp = await _http_client.post(
            "/v1/chat/completions", json=body
        )
        status_code = str(resp.status_code)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=resp.text[:500],
            )
        return resp.json()
    except httpx.TimeoutException:
        status_code = "timeout"
        raise HTTPException(status_code=504, detail="推理超时，请稍后重试")
    except httpx.ConnectError:
        status_code = "connect_error"
        raise HTTPException(status_code=503, detail="推理服务不可达")
    finally:
        elapsed = time.perf_counter() - t0
        REQUEST_LATENCY.labels(status_code=status_code).observe(elapsed)
        REQUEST_TOTAL.labels(status_code=status_code).inc()


@app.get("/v1/models")
async def list_models():
    """透传模型列表查询"""
    resp = await _http_client.get("/v1/models")
    return resp.json()


@app.get("/health")
async def health():
    """网关健康检查，同时探测 vLLM 可达性"""
    try:
        resp = await _http_client.get("/health", timeout=5)
        vllm_healthy = resp.status_code == 200
    except Exception:
        vllm_healthy = False
    return {
        "gateway": "ok",
        "vllm": "ok" if vllm_healthy else "unreachable",
    }


@app.get("/metrics")
async def metrics():
    """暴露 Prometheus 指标"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
