"""
文件: chapter10/router/server.py

路由分类服务 FastAPI 入口。

部署模式:
  该服务作为独立微服务部署，由 Istio ExtAuthz 在请求进入推理端点前调用，
  根据返回的路由决策设置响应头 x-route-target，
  VirtualService 依据该 Header 做最终转发。

接口:
  POST /route      接收请求体，返回路由决策
  GET  /healthz    健康检查（供 Kubernetes readinessProbe 使用）
  GET  /metrics    Prometheus 指标端点（路由决策分布统计）

启动方式:
  uvicorn router.server:app --host 0.0.0.0 --port 8080 --workers 4
"""

from __future__ import annotations

import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .classifier import classify, RouteTarget, RoutingDecision

app = FastAPI(title="AI Platform Router", version="1.0.0")

# ── Prometheus 指标 ────────────────────────────────────────────────────────

# 路由决策计数器：按目标框架和触发原因分组统计
ROUTE_DECISIONS = Counter(
    "router_decisions_total",
    "Total routing decisions made",
    ["target", "reason"],
)

# 路由服务延迟直方图
ROUTE_LATENCY = Histogram(
    "router_latency_seconds",
    "Time spent making routing decisions",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
)

# 路由错误计数器
ROUTE_ERRORS = Counter(
    "router_errors_total",
    "Total routing errors (malformed requests)",
)


# ── 路由接口 ───────────────────────────────────────────────────────────────

@app.post("/route", response_model=RoutingDecision)
async def route_request(request: Request) -> RoutingDecision:
    """
    接收 OpenAI 格式请求体，返回路由决策。

    由 Istio ExtAuthz 在转发前调用。响应体中的 target 字段
    供网关依据 x-route-target 响应头动态选择后端。

    错误处理:
      - 请求体不是合法 JSON → 400
      - 分类器内部异常     → 500（默认路由至 vLLM 保障可用性）
    """
    start = time.perf_counter()

    try:
        payload = await request.json()
    except Exception:
        ROUTE_ERRORS.inc()
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        decision = classify(payload)
    except Exception as exc:
        ROUTE_ERRORS.inc()
        # 分类器异常时安全降级：路由至 vLLM（更稳定的默认后端）
        decision = RoutingDecision(
            target=RouteTarget.VLLM,
            reason="classifier_error_fallback",
            confidence=0.5,
        )

    elapsed = time.perf_counter() - start
    ROUTE_LATENCY.observe(elapsed)
    ROUTE_DECISIONS.labels(
        target=decision.target.value,
        reason=decision.reason,
    ).inc()

    return decision


# ── 健康检查 ───────────────────────────────────────────────────────────────

@app.get("/healthz")
async def health_check():
    """Kubernetes readinessProbe 端点"""
    return {"status": "ok"}


# ── Prometheus 指标端点 ────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """暴露 Prometheus 格式指标"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
