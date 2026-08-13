"""
文件: chapter10/observability/prober.py

端到端黑盒探针。

功能:
  每 30 秒向 vLLM 和 SGLang 推理端点发送轻量级探测请求，
  验证完整推理链路（网关→框架→模型→响应）的端到端可用性，
  并将结果以 Prometheus 指标形式暴露，供 Grafana Dashboard 展示
  和 AlertManager 告警规则引用。

与框架自身指标的区别:
  框架的 /metrics 端点反映的是内部运行状态（队列长度、缓存使用率等）。
  黑盒探针反映的是从调用方视角观察到的端到端可用性和延迟，
  包括网络链路、Istio 路由、模型推理的全链路健康状态。
  两者互补，缺一不可。

指标说明:
  ai_platform_endpoint_up{endpoint, framework}
    —— 端点可用性（1=可用，0=不可用）
  ai_platform_e2e_latency_seconds{framework}
    —— 端到端响应延迟直方图（包含网络+推理完整耗时）
  ai_platform_probe_errors_total{endpoint, framework, error_type}
    —— 探测错误计数（按错误类型分类：timeout/http_error/connection_error）

启动方式:
  python -m observability.prober
  # 指标端点: http://localhost:9091/metrics
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger("prober")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── 配置 ───────────────────────────────────────────────────────────────────

PROBE_INTERVAL_SECONDS = int(os.getenv("PROBE_INTERVAL_SECONDS", "30"))
PROBE_TIMEOUT_SECONDS = float(os.getenv("PROBE_TIMEOUT_SECONDS", "15.0"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "9091"))

# 推理端点配置：(名称, URL, 框架标识)
ENDPOINTS: list[tuple[str, str, str]] = [
    (
        "vllm-primary",
        os.getenv("VLLM_ENDPOINT", "http://vllm-service:8000") + "/v1/chat/completions",
        "vllm",
    ),
    (
        "sglang-primary",
        os.getenv("SGLANG_ENDPOINT", "http://sglang-service:30000") + "/v1/chat/completions",
        "sglang",
    ),
]

# 探测请求体：单轮对话，max_tokens=10，最小化对推理资源的占用
PROBE_PAYLOAD = {
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 10,
    "stream": False,
}


# ── Prometheus 指标定义 ────────────────────────────────────────────────────

ENDPOINT_UP = Gauge(
    "ai_platform_endpoint_up",
    "Endpoint availability: 1=up, 0=down",
    ["endpoint", "framework"],
)

E2E_LATENCY = Histogram(
    "ai_platform_e2e_latency_seconds",
    "End-to-end probe latency from client perspective (network + inference)",
    ["framework"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0],
)

PROBE_ERRORS = Counter(
    "ai_platform_probe_errors_total",
    "Total probe errors by type",
    ["endpoint", "framework", "error_type"],
)


# ── 探测逻辑 ──────────────────────────────────────────────────────────────

async def probe_endpoint(
    name: str,
    url: str,
    framework: str,
    client: httpx.AsyncClient,
) -> None:
    """
    向单个端点发送探测请求，记录可用性和端到端延迟。

    成功条件: HTTP 200 且响应体包含 choices 字段（确认推理链路完整）。
    失败类型:
      - timeout        : 请求超过 PROBE_TIMEOUT_SECONDS
      - http_error     : 非200状态码（如502、503）
      - connection_error: 网络连接失败（DNS解析失败、连接被拒绝等）
      - invalid_response: HTTP 200 但响应体缺少预期字段（框架内部错误）
    """
    start = time.perf_counter()
    try:
        resp = await client.post(url, json=PROBE_PAYLOAD)
        elapsed = time.perf_counter() - start

        if resp.status_code == 200:
            # 进一步验证响应体结构，确认推理链路端到端正常
            body = resp.json()
            if "choices" in body and len(body["choices"]) > 0:
                ENDPOINT_UP.labels(endpoint=name, framework=framework).set(1)
                E2E_LATENCY.labels(framework=framework).observe(elapsed)
                logger.debug(
                    "Probe OK: endpoint=%s, latency=%.3fs", name, elapsed
                )
            else:
                ENDPOINT_UP.labels(endpoint=name, framework=framework).set(0)
                PROBE_ERRORS.labels(
                    endpoint=name, framework=framework, error_type="invalid_response"
                ).inc()
                logger.warning(
                    "Probe INVALID_RESPONSE: endpoint=%s, body=%s",
                    name, str(body)[:200],
                )
        else:
            ENDPOINT_UP.labels(endpoint=name, framework=framework).set(0)
            PROBE_ERRORS.labels(
                endpoint=name, framework=framework, error_type="http_error"
            ).inc()
            logger.warning(
                "Probe HTTP_ERROR: endpoint=%s, status=%d", name, resp.status_code
            )

    except httpx.TimeoutException:
        ENDPOINT_UP.labels(endpoint=name, framework=framework).set(0)
        PROBE_ERRORS.labels(
            endpoint=name, framework=framework, error_type="timeout"
        ).inc()
        logger.warning("Probe TIMEOUT: endpoint=%s", name)

    except httpx.RequestError as exc:
        ENDPOINT_UP.labels(endpoint=name, framework=framework).set(0)
        PROBE_ERRORS.labels(
            endpoint=name, framework=framework, error_type="connection_error"
        ).inc()
        logger.warning("Probe CONNECTION_ERROR: endpoint=%s, error=%s", name, exc)


async def probe_loop() -> None:
    """主探测循环：并发探测所有端点，每 PROBE_INTERVAL_SECONDS 秒一轮。"""
    logger.info(
        "Prober started. Endpoints=%d, interval=%ds, timeout=%.1fs",
        len(ENDPOINTS), PROBE_INTERVAL_SECONDS, PROBE_TIMEOUT_SECONDS,
    )
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
        while True:
            tasks = [
                probe_endpoint(name, url, framework, client)
                for name, url, framework in ENDPOINTS
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(PROBE_INTERVAL_SECONDS)


# ── 入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 启动 Prometheus 指标 HTTP 服务器
    start_http_server(METRICS_PORT)
    logger.info("Metrics server started on port %d", METRICS_PORT)
    asyncio.run(probe_loop())
