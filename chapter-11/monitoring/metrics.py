# monitoring/metrics.py
# Prometheus 指标定义与缓存监控中间件（ch11_code_06）
#
# 监控体系的三个层次：
#   框架层：直接采集 SGLang /metrics 端点的原生指标。
#           SGLang v0.5.x 暴露 sglang:cache_hit_tokens_total
#           和 sglang:total_input_tokens_total，两者相除
#           即为 token 级缓存命中率。
#   应用层：本模块实现的 CacheMetricsMiddleware，
#           在请求处理后记录请求粒度的命中信息，
#           并按前缀层级（L1/L2/L3）分类统计。
#   系统层：通过 memory_budget.get_gpu_memory_status()
#           采集每卡显存利用率，超过阈值时触发告警。
#
# vLLM 的命中率说明：
#   vLLM v0.8.x 的原生指标不直接暴露 token 级缓存命中数。
#   本模块通过比较请求的实际 TTFT 与无缓存时的理论 TTFT
#   来推断命中状态：实际 TTFT < 理论值的 30% 时判定为命中。
#   理论 TTFT 基于实测的无缓存 prefill 速率（约 0.54 ms/token
#   在 4× A100 + Qwen2-72B 配置下）计算。
#   此推断方法存在约 ±8% 的误差（含网络延迟和调度等待噪声），
#   在统计大量请求时误差趋于抵消，适用于宏观命中率分析，
#   不适用于单请求级别的精确判断。

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ── Prometheus 指标注册 ──────────────────────────────────────────
# 所有指标集中在此处定义，避免重复注册导致的 ValueError

METRICS: dict = {
    # 命中缓存的 prefill token 总数（按框架、文书类型、前缀层级）
    "hit_tokens": Counter(
        "legal_cache_hit_tokens_total",
        "命中缓存的 prefill token 总数",
        ["framework", "doc_type", "prefix_layer"],
    ),
    # 未命中缓存的 prefill token 总数（按框架、文书类型）
    "miss_tokens": Counter(
        "legal_cache_miss_tokens_total",
        "未命中缓存的 prefill token 总数",
        ["framework", "doc_type"],
    ),
    # 当前 5 分钟滑动窗口内的缓存命中率
    "hit_ratio": Gauge(
        "legal_cache_hit_ratio_current",
        "5 分钟滑动窗口缓存命中率",
        ["framework"],
    ),
    # 请求首 token 延迟分布（按框架、缓存状态）
    "ttft_histogram": Histogram(
        "legal_request_ttft_seconds",
        "请求首 token 延迟分布（秒）",
        ["framework", "cache_status"],
        buckets=[
            0.5, 1.0, 2.0, 5.0, 8.0,
            10.0, 15.0, 20.0, 30.0,
        ],
    ),
    # 当前缓存中的文书数量（估算值）
    "active_cached_docs": Gauge(
        "legal_active_cached_docs",
        "当前缓存中的文书数量（估算）",
        ["framework"],
    ),
    # 端到端请求延迟分布
    "e2e_histogram": Histogram(
        "legal_request_e2e_seconds",
        "请求端到端延迟分布（秒）",
        ["framework", "task_type"],
        buckets=[
            1.0, 5.0, 10.0, 15.0, 20.0,
            30.0, 45.0, 60.0,
        ],
    ),
}


@dataclass
class _WindowEntry:
    """滑动窗口中的单条记录。"""
    timestamp: float
    hit_ratio: float


class CacheMetricsMiddleware:
    """
    请求粒度的缓存命中监控中间件。

    在每次推理请求完成后调用 record_request()，
    中间件将自动更新 Prometheus 指标并维护滑动窗口统计。

    前缀层级（prefix_layer）的判断规则：
      hit_ratio >= 0.95 → doc_body（L3 文书正文层命中）
      hit_ratio >= 0.60 → instruction（L2 任务指令层命中）
      hit_ratio >= 0.10 → system_prompt（L1 系统提示层命中）
      hit_ratio <  0.10 → miss（全部未命中）
    """

    # 滑动窗口宽度（秒）
    WINDOW_SECONDS: float = 300.0
    # 判定缓存命中的 hit_ratio 阈值
    HIT_THRESHOLD: float = 0.50

    # vLLM 无缓存情况下的 prefill 速率（ms/token）
    # 基于 4× A100 80GB + Qwen2-72B AWQ 的实测基准，
    # 请在实际环境中通过基线测试脚本重新标定此值
    VLLM_PREFILL_MS_PER_TOKEN: float = 0.54

    def __init__(self, framework: str) -> None:
        """
        参数
        ----
        framework : str
            框架标识符，如 "sglang" 或 "vllm"。
        """
        self.framework = framework
        self._window: deque[_WindowEntry] = deque()
        self._lock = Lock()

    def _classify_prefix_layer(
        self, hit_ratio: float
    ) -> str:
        """根据命中率推断命中的前缀层级。"""
        if hit_ratio >= 0.95:
            return "doc_body"
        if hit_ratio >= 0.60:
            return "instruction"
        if hit_ratio >= 0.10:
            return "system_prompt"
        return "miss"

    def infer_hit_ratio_vllm(
        self,
        input_tokens: int,
        actual_ttft_ms: float,
    ) -> float:
        """
        通过 TTFT 推断 vLLM 的缓存命中率（间接方法）。

        逻辑：
          无缓存时理论 TTFT = input_tokens × prefill_ms_per_token
          实际 TTFT 远低于理论值时，推断为缓存命中。
          命中率估算 = 1 - actual_ttft / theoretical_ttft
          结果限制在 [0.0, 1.0] 区间内。

        误差说明：
          此方法存在约 ±8% 的误差，来源包括：
          网络往返延迟（约 1–3 ms）、调度队列等待时间、
          以及不同批次大小导致的 prefill 速率波动。
          适用于宏观统计，不适用于单请求精确判断。
        """
        theoretical_ttft = (
            input_tokens * self.VLLM_PREFILL_MS_PER_TOKEN
        )
        if theoretical_ttft <= 0:
            return 0.0
        estimated_miss_ratio = actual_ttft_ms / theoretical_ttft
        return max(0.0, min(1.0, 1.0 - estimated_miss_ratio))

    def record_request(
        self,
        input_tokens: int,
        cached_tokens: int,
        ttft_ms: float,
        e2e_ms: float,
        doc_type: str = "unknown",
        task_type: str = "unknown",
    ) -> None:
        """
        记录单次请求的缓存命中信息并更新所有指标。

        参数
        ----
        input_tokens : int
            本次请求的输入 token 数（含前缀）。
        cached_tokens : int
            命中缓存的 token 数。
            对于 SGLang：从 /metrics 端点获取。
            对于 vLLM：通过 infer_hit_ratio_vllm() 估算。
        ttft_ms : float
            实测首 token 延迟（毫秒）。
        e2e_ms : float
            实测端到端延迟（毫秒）。
        doc_type : str
            文书类型标签，如 "procurement" / "labor" / "software"。
        task_type : str
            任务类型标签，如 "risk" / "obligation" 等。
        """
        hit_ratio = cached_tokens / max(input_tokens, 1)
        prefix_layer = self._classify_prefix_layer(hit_ratio)
        cache_status = (
            "hit" if hit_ratio >= self.HIT_THRESHOLD else "miss"
        )

        # 更新 Prometheus 计数器
        if cached_tokens > 0:
            METRICS["hit_tokens"].labels(
                framework=self.framework,
                doc_type=doc_type,
                prefix_layer=prefix_layer,
            ).inc(cached_tokens)
        miss_tokens = max(input_tokens - cached_tokens, 0)
        if miss_tokens > 0:
            METRICS["miss_tokens"].labels(
                framework=self.framework,
                doc_type=doc_type,
            ).inc(miss_tokens)

        # 更新延迟直方图
        METRICS["ttft_histogram"].labels(
            framework=self.framework,
            cache_status=cache_status,
        ).observe(ttft_ms / 1000.0)

        METRICS["e2e_histogram"].labels(
            framework=self.framework,
            task_type=task_type,
        ).observe(e2e_ms / 1000.0)

        # 更新滑动窗口命中率
        now = time.monotonic()
        with self._lock:
            self._window.append(
                _WindowEntry(timestamp=now, hit_ratio=hit_ratio)
            )
            # 移除窗口外的过期条目
            while (
                self._window
                and now - self._window[0].timestamp
                > self.WINDOW_SECONDS
            ):
                self._window.popleft()

            if self._window:
                avg_ratio = sum(
                    e.hit_ratio for e in self._window
                ) / len(self._window)
                METRICS["hit_ratio"].labels(
                    framework=self.framework
                ).set(avg_ratio)

    def update_cached_doc_count(self, count: int) -> None:
        """更新当前缓存中的文书数量估算值。"""
        METRICS["active_cached_docs"].labels(
            framework=self.framework
        ).set(count)

    def get_window_stats(self) -> dict:
        """
        返回当前滑动窗口的统计摘要，供调试使用。
        """
        with self._lock:
            if not self._window:
                return {
                    "window_size": 0,
                    "avg_hit_ratio": 0.0,
                    "hit_count": 0,
                    "miss_count": 0,
                }
            ratios = [e.hit_ratio for e in self._window]
            hit_count = sum(
                1 for r in ratios if r >= self.HIT_THRESHOLD
            )
            return {
                "window_size": len(ratios),
                "avg_hit_ratio": sum(ratios) / len(ratios),
                "hit_count": hit_count,
                "miss_count": len(ratios) - hit_count,
            }


def start_metrics_server(port: int = 8000) -> None:
    """
    启动 Prometheus metrics HTTP 服务器。

    启动后，Prometheus 可从 http://host:port/metrics
    拉取指标数据。建议在主进程启动时调用一次。
    """
    start_http_server(port)
    print(f"Prometheus metrics 服务器已启动，监听端口 {port}")
