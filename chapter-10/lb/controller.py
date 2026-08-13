"""
文件: chapter10/lb/controller.py

跨框架负载均衡控制器。

功能:
  每15秒向 Prometheus 查询 vLLM 和 SGLang 的最新负载指标，
  计算归一化负载压力分数，动态调整 Istio VirtualService 的路由权重。

归一化压力分数（0.0-1.0）:
  vLLM:   0.6 × min(等待队列长度 / 50, 1) + 0.4 × (KV cache使用率% / 100)
  SGLang: 0.6 × min(队列请求数 / 30, 1)  + 0.4 × token使用率(0-1)

路由权重调整策略:
  正常状态：vLLM 100%，SGLang 0%（SGLang只处理专属请求）
  vLLM过载（>0.8）且SGLang有余量（<0.6）：
    将部分通用请求溢出至SGLang（最多40%），线性插值

并发安全说明:
  使用 patch_namespaced_custom_object + JSON Patch 替代 read-modify-write，
  避免多副本控制器实例并发写入时产生的覆盖冲突。
  建议以单副本（replicas: 1）部署，配合 Kubernetes Leader Election 防止分裂脑。

部署方式:
  该控制器以 Deployment 运行在 inference-prod 命名空间，
  需要 ServiceAccount 具备读写 VirtualService 的 RBAC 权限。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx
from kubernetes import client as k8s_client, config as k8s_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lb-controller")

# ── 配置常量 ───────────────────────────────────────────────────────────────

PROMETHEUS_URL = "http://prometheus-server.monitoring:9090"
NAMESPACE = "inference-prod"
VIRTUAL_SERVICE_NAME = "ai-platform-routing"
CONTROL_LOOP_INTERVAL_SECONDS = 15

# 负载压力分数的归一化基准（队列长度超过此值时压力分数达到上限）
VLLM_QUEUE_NORM_BASE = 50      # vLLM等待队列长度的归一化分母
SGLANG_QUEUE_NORM_BASE = 30    # SGLang队列请求数的归一化分母

# 权重调整阈值
VLLM_OVERLOAD_THRESHOLD = 0.8   # vLLM压力分数超过此值时触发溢出路由
SGLANG_CAPACITY_THRESHOLD = 0.6 # SGLang压力分数低于此值时才接受溢出
MAX_OVERFLOW_WEIGHT = 40        # 溢出流量最多占通用请求的40%


# ── 数据模型 ───────────────────────────────────────────────────────────────

@dataclass
class LoadScore:
    """归一化负载压力分数（0.0-1.0，值越高表示负载越重）"""
    vllm_score: float
    sglang_score: float


# ── Prometheus 查询 ────────────────────────────────────────────────────────

async def query_prometheus(client: httpx.AsyncClient, promql: str) -> float:
    """
    查询 Prometheus 即时值，返回所有匹配序列的均值。
    查询失败时返回 0.0（保守策略：不因监控故障触发不必要的路由变更）。
    """
    try:
        resp = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            return 0.0
        # 多个序列取均值
        values = [float(r["value"][1]) for r in results]
        return sum(values) / len(values)
    except Exception as exc:
        logger.warning("Prometheus query failed: %s | query=%s", exc, promql)
        return 0.0


async def compute_load_scores(client: httpx.AsyncClient) -> LoadScore:
    """
    并发查询 vLLM 和 SGLang 的负载指标，计算归一化压力分数。

    vLLM指标（原始名称，含冒号分隔符）:
      vllm:num_requests_waiting  —— 等待队列长度
      vllm:gpu_cache_usage_perc  —— KV cache使用率（0-100）

    SGLang指标:
      sglang:num_queue_reqs      —— 队列中等待的请求数
      sglang:token_usage         —— token使用率（0-1，已归一化）
    """
    # 并发查询4个指标，减少总等待时间
    (
        vllm_queue,
        vllm_cache,
        sglang_queue,
        sglang_token,
    ) = await asyncio.gather(
        query_prometheus(client, "avg(vllm:num_requests_waiting)"),
        query_prometheus(client, "avg(vllm:gpu_cache_usage_perc)"),
        query_prometheus(client, "avg(sglang:num_queue_reqs)"),
        query_prometheus(client, "avg(sglang:token_usage)"),
    )

    vllm_score = (
        min(vllm_queue / VLLM_QUEUE_NORM_BASE, 1.0) * 0.6
        + (vllm_cache / 100.0) * 0.4
    )
    sglang_score = (
        min(sglang_queue / SGLANG_QUEUE_NORM_BASE, 1.0) * 0.6
        + sglang_token * 0.4
    )

    logger.info(
        "Load scores — vLLM: %.3f (queue=%.0f, cache=%.0f%%) | "
        "SGLang: %.3f (queue=%.0f, token=%.2f)",
        vllm_score, vllm_queue, vllm_cache,
        sglang_score, sglang_queue, sglang_token,
    )
    return LoadScore(vllm_score=vllm_score, sglang_score=sglang_score)


# ── 权重计算 ───────────────────────────────────────────────────────────────

def compute_routing_weights(scores: LoadScore) -> tuple[int, int]:
    """
    基于负载压力分数计算 vLLM 和 SGLang 的路由权重。

    Returns:
        (vllm_weight, sglang_overflow_weight): 两者之和为100
    """
    # 正常状态：全部通用请求路由至vLLM
    if scores.vllm_score <= VLLM_OVERLOAD_THRESHOLD:
        return 100, 0

    # vLLM过载但SGLang已饱和：不溢出，维持vLLM承接（接受排队）
    if scores.sglang_score >= SGLANG_CAPACITY_THRESHOLD:
        return 100, 0

    # vLLM过载且SGLang有余量：按过载程度线性插值溢出比例
    # 过载程度 = (当前分数 - 阈值) / (1.0 - 阈值)，值域[0, 1]
    overload_ratio = (scores.vllm_score - VLLM_OVERLOAD_THRESHOLD) / (
        1.0 - VLLM_OVERLOAD_THRESHOLD
    )
    overflow = int(overload_ratio * MAX_OVERFLOW_WEIGHT)
    vllm_w = 100 - overflow

    logger.info(
        "Weight adjustment: vLLM=%d%%, SGLang_overflow=%d%% (overload_ratio=%.2f)",
        vllm_w, overflow, overload_ratio,
    )
    return vllm_w, overflow


# ── Kubernetes VirtualService 更新 ────────────────────────────────────────

def build_weight_patch(vllm_w: int, sglang_w: int) -> list[dict]:
    """
    构建 JSON Patch 操作列表，仅修改通用路由（/v1/*）的权重字段。

    使用 JSON Patch（RFC 6902）而非全量 replace，原因：
    - replace 需要先读取完整对象再写回，存在并发写入覆盖问题（ABA问题）
    - JSON Patch 仅修改指定路径，减少写冲突概率
    - 路径 /spec/http/1/route 对应 04_gateway_config.yaml 中的第二条路由规则（索引1）
      第一条规则（索引0）是 SGLang 专属路径，不参与权重调整
    """
    return [
        {
            "op": "replace",
            "path": "/spec/http/1/route/0/weight",    # vLLM destination weight
            "value": vllm_w,
        },
        {
            "op": "replace",
            "path": "/spec/http/1/route/1/weight",    # SGLang overflow weight
            "value": sglang_w,
        },
    ]


async def update_virtual_service_weights(vllm_w: int, sglang_w: int):
    """
    通过 Kubernetes API 以 JSON Patch 方式更新 VirtualService 路由权重。

    使用 patch 而非 replace 的理由见 build_weight_patch() 的注释。
    即使多个控制器副本并发执行 patch，最坏情况是最后一次写入生效，
    不会像 replace 那样静默丢弃中间状态的完整配置。
    """
    custom_api = k8s_client.CustomObjectsApi()
    patch_body = build_weight_patch(vllm_w, sglang_w)

    try:
        custom_api.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1alpha3",
            namespace=NAMESPACE,
            plural="virtualservices",
            name=VIRTUAL_SERVICE_NAME,
            body=patch_body,
            # content_type 必须声明为 JSON Patch 格式
            _content_type="application/json-patch+json",
        )
        logger.info("VirtualService updated: vLLM=%d%%, SGLang=%d%%", vllm_w, sglang_w)
    except k8s_client.ApiException as exc:
        logger.error("Failed to patch VirtualService: %s", exc)


# ── 主控制循环 ─────────────────────────────────────────────────────────────

async def load_balance_loop():
    """
    负载均衡主循环，每 CONTROL_LOOP_INTERVAL_SECONDS 秒执行一次权重调整。
    """
    # 加载 Kubernetes 配置（Pod内运行使用 in-cluster config）
    try:
        k8s_config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes config")
    except k8s_config.ConfigException:
        # 本地开发环境使用 kubeconfig
        k8s_config.load_kube_config()
        logger.info("Loaded local kubeconfig")

    async with httpx.AsyncClient() as http_client:
        while True:
            try:
                scores = await compute_load_scores(http_client)
                vllm_w, sglang_w = compute_routing_weights(scores)
                await update_virtual_service_weights(vllm_w, sglang_w)
            except Exception as exc:
                logger.error("Control loop iteration failed: %s", exc, exc_info=True)
            finally:
                await asyncio.sleep(CONTROL_LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(load_balance_loop())
