"""
文件: chapter10/failover/degradation.py

降级控制中间件。

功能:
  监控 vLLM 和 SGLang 的负载压力状态，在持续过载时触发服务降级。
  降级期间，低优先级请求（x-priority: low）返回 429 Too Many Requests，
  高优先级请求（x-priority: high）不受影响，正常透传至推理后端。

降级触发条件:
  负载压力分数持续 ≥ 0.9 超过 180 秒（3分钟）。

降级解除条件:
  Redis 中的降级标记自动在 60 秒后过期（TTL），
  或由外部运维操作手动 DEL degradation:{framework} 键强制解除。

并发安全修复（相对原始草稿）:
  原始实现中，"首次超阈值时间记录"存在竞态条件：
  多个进程同时判断 key 不存在时各自写入，导致计时窗口被重置。
  修复方案: 使用 Redis SET NX（仅在键不存在时设置）原子操作，
  确保首次写入的唯一性，后续并发调用不会覆盖已记录的起始时间。

部署方式:
  作为路由服务或独立 FastAPI 服务的中间件层运行。
  Redis 连接地址通过 REDIS_URL 环境变量注入。
"""

from __future__ import annotations

import logging
import os
import time

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("degradation")

# ── 配置 ───────────────────────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://redis-service:6379")

# 负载压力分数触发降级的阈值
DEGRADATION_THRESHOLD: float = float(os.getenv("DEGRADATION_THRESHOLD", "0.9"))

# 负载压力分数持续超过阈值多少秒后触发降级（单位：秒）
DEGRADATION_WINDOW_SECONDS: int = int(os.getenv("DEGRADATION_WINDOW_SECONDS", "180"))

# 降级状态的自动解除时间（Redis TTL，单位：秒）
DEGRADATION_TTL_SECONDS: int = int(os.getenv("DEGRADATION_TTL_SECONDS", "60"))

# Redis键前缀
_DEGRADED_KEY_PREFIX = "degradation:"       # 降级激活标记
_OVER_THRESHOLD_KEY_PREFIX = "over_threshold:"  # 首次超阈值时间戳


# ── Redis 连接 ─────────────────────────────────────────────────────────────

def _make_redis_client() -> aioredis.Redis:
    return aioredis.from_url(REDIS_URL, decode_responses=True)


# 模块级单例，由 FastAPI 生命周期管理
_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = _make_redis_client()
    return _redis


# ── 核心降级状态管理 ───────────────────────────────────────────────────────

async def check_and_set_degradation(framework: str, load_score: float) -> None:
    """
    根据当前负载压力分数更新降级状态。

    并发安全保证：
      使用 SET key value EX ttl NX（仅在键不存在时设置）原子操作记录首次超阈值时间，
      多个并发调用只有第一个会成功写入，后续调用不会覆盖起始时间戳。
      这消除了原始实现中 get→判断→set 三步操作之间的竞态窗口。

    Args:
        framework:  框架名称（"vllm" 或 "sglang"）
        load_score: 当前归一化负载压力分数（0.0-1.0）
    """
    redis = get_redis()
    degraded_key = f"{_DEGRADED_KEY_PREFIX}{framework}"
    over_threshold_key = f"{_OVER_THRESHOLD_KEY_PREFIX}{framework}"

    if load_score >= DEGRADATION_THRESHOLD:
        # 原子写入：仅在键不存在时记录首次超阈值时间
        # NX 保证并发安全：第一个到达的调用写入成功，后续调用直接跳过
        now = str(time.time())
        await redis.set(
            over_threshold_key,
            now,
            ex=DEGRADATION_WINDOW_SECONDS * 3,  # TTL设为窗口的3倍，防止误清除
            nx=True,                             # 关键：仅在键不存在时设置
        )

        # 读取首次超阈值时间，判断是否已持续超过窗口期
        first_over_str = await redis.get(over_threshold_key)
        if first_over_str is not None:
            elapsed = time.time() - float(first_over_str)
            if elapsed >= DEGRADATION_WINDOW_SECONDS:
                # 触发降级：写入降级标记，TTL到期后自动解除
                was_already_degraded = await redis.exists(degraded_key)
                await redis.set(degraded_key, "1", ex=DEGRADATION_TTL_SECONDS)
                if not was_already_degraded:
                    logger.warning(
                        "DEGRADATION ACTIVATED: framework=%s, "
                        "load_score=%.3f, elapsed=%.0fs",
                        framework, load_score, elapsed,
                    )
    else:
        # 负载恢复：清除超阈值计时和降级标记
        was_degraded = await redis.exists(degraded_key)
        await redis.delete(over_threshold_key, degraded_key)
        if was_degraded:
            logger.info(
                "DEGRADATION CLEARED: framework=%s, load_score=%.3f",
                framework, load_score,
            )


async def is_degraded(framework: str) -> bool:
    """查询框架是否处于降级激活状态。"""
    redis = get_redis()
    return await redis.exists(f"{_DEGRADED_KEY_PREFIX}{framework}") > 0


async def force_clear_degradation(framework: str) -> None:
    """
    运维接口：强制解除降级状态（用于人工干预恢复）。
    同时清除超阈值计时，防止立即重新触发降级。
    """
    redis = get_redis()
    await redis.delete(
        f"{_DEGRADED_KEY_PREFIX}{framework}",
        f"{_OVER_THRESHOLD_KEY_PREFIX}{framework}",
    )
    logger.info("DEGRADATION FORCE-CLEARED: framework=%s", framework)


# ── FastAPI 应用与中间件 ───────────────────────────────────────────────────

app = FastAPI(title="Degradation Controller", version="1.0.0")


@app.middleware("http")
async def degradation_middleware(request: Request, call_next):
    """
    降级中间件：在框架过载时对低优先级请求返回 429。

    请求优先级通过 HTTP Header 声明：
      x-priority: high  —— 实时服务请求，不受降级影响，正常转发
      x-priority: low   —— 批处理/异步请求，降级期间返回 429（默认）

    目标框架由请求路径前缀推断：
      /sglang/* → SGLang
      其他      → vLLM
    """
    priority = request.headers.get("x-priority", "low")
    path = request.url.path

    # 健康检查和运维端点不受降级影响
    if path in ("/healthz", "/metrics", "/admin/degradation"):
        return await call_next(request)

    # 高优先级请求无条件透传，保障实时SLA
    if priority == "high":
        return await call_next(request)

    # 推断请求目标框架
    target_framework = "sglang" if path.startswith("/sglang/") else "vllm"

    # 低优先级请求在降级期间返回 429
    if await is_degraded(target_framework):
        logger.debug(
            "Request rejected (degraded): path=%s, framework=%s",
            path, target_framework,
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "service_degraded",
                "message": (
                    f"{target_framework} cluster is overloaded. "
                    f"Please retry after {DEGRADATION_TTL_SECONDS}s."
                ),
                "retry_after_seconds": DEGRADATION_TTL_SECONDS,
            },
            headers={"Retry-After": str(DEGRADATION_TTL_SECONDS)},
        )

    return await call_next(request)


# ── 运维管理接口 ──────────────────────────────────────────────────────────

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


@app.get("/admin/degradation/{framework}")
async def get_degradation_status(framework: str):
    """查询指定框架的降级状态（供运维监控使用）。"""
    degraded = await is_degraded(framework)
    return {"framework": framework, "degraded": degraded}


@app.delete("/admin/degradation/{framework}")
async def clear_degradation(framework: str):
    """强制解除指定框架的降级状态（运维人工恢复接口）。"""
    await force_clear_degradation(framework)
    return {"framework": framework, "degraded": False, "message": "degradation cleared"}


@app.post("/admin/degradation/{framework}")
async def update_degradation(framework: str, load_score: float):
    """
    接收外部上报的负载压力分数，更新降级状态。
    通常由负载均衡控制器（lb/controller.py）定期调用。
    """
    await check_and_set_degradation(framework, load_score)
    degraded = await is_degraded(framework)
    return {
        "framework": framework,
        "load_score": load_score,
        "degraded": degraded,
    }
