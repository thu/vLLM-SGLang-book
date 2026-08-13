# src/video_inference/latency_profiler.py
# 功能: 多模态推理四阶段延迟测量框架
#   阶段划分：帧预处理 / ViT编码 / LLM prefill / LLM decode
#   支持滑动窗口统计 P50/P95/P99 分位数
#
# ⚠ 测量说明：
#   vLLM 在引擎层面将 ViT 编码与 LLM prefill 作为同一请求整体处理，
#   Python 调用层面无法精确分离两者。本框架通过"短序列代理法"近似拆分
#   （见 split_vit_prefill），误差约 ±8%，仅用于优化方向判断。
#   如需 kernel 级精确分析，请改用 torch.profiler 或 nsys profile。
#
# 版本基线: vLLM v0.8.x, Python 3.10+

import time
import numpy as np
from dataclasses import dataclass
from collections import deque


@dataclass
class LatencyBreakdown:
    """
    单次推理四阶段延迟记录（毫秒）。

    frame_preprocess_ms : CPU 端帧解码 + 归一化 + base64 编码 + 消息构建
    vit_encode_ms       : ViT 前向计算（近似值，通过短序列代理法测量）
    llm_prefill_ms      : LLM prefill 阶段（近似值）
    llm_decode_ms       : LLM decode 阶段（自回归生成至 EOS 或 max_tokens）
    """
    frame_preprocess_ms: float = 0.0
    vit_encode_ms: float       = 0.0
    llm_prefill_ms: float      = 0.0
    llm_decode_ms: float       = 0.0

    @property
    def total_ms(self) -> float:
        return (self.frame_preprocess_ms + self.vit_encode_ms
                + self.llm_prefill_ms + self.llm_decode_ms)

    @property
    def bottleneck(self) -> str:
        """返回耗时最长的阶段名称，用于自动化瓶颈定位。"""
        stages = {
            "frame_preprocess": self.frame_preprocess_ms,
            "vit_encode":       self.vit_encode_ms,
            "llm_prefill":      self.llm_prefill_ms,
            "llm_decode":       self.llm_decode_ms,
        }
        return max(stages, key=stages.get)


class LatencyProfiler:
    """
    延迟分析器，维护滑动窗口（默认 200 条记录），支持实时查询分位数统计。

    record(bd)  : 写入单次延迟记录
    summary()   : 返回 P50/P95/P99 及各阶段均值字典
    reset()     : 清空历史记录（切换测试配置时调用）
    """

    def __init__(self, window_size: int = 200):
        self._window: deque = deque(maxlen=window_size)

    def record(self, bd: LatencyBreakdown) -> None:
        self._window.append(bd)

    def summary(self) -> dict:
        if not self._window:
            return {}
        totals = [r.total_ms for r in self._window]
        return {
            "count":              len(totals),
            "p50_ms":             float(np.percentile(totals, 50)),
            "p95_ms":             float(np.percentile(totals, 95)),
            "p99_ms":             float(np.percentile(totals, 99)),
            "mean_preprocess_ms": float(np.mean([r.frame_preprocess_ms for r in self._window])),
            "mean_vit_ms":        float(np.mean([r.vit_encode_ms       for r in self._window])),
            "mean_prefill_ms":    float(np.mean([r.llm_prefill_ms      for r in self._window])),
            "mean_decode_ms":     float(np.mean([r.llm_decode_ms       for r in self._window])),
            "bottleneck":         self._window[-1].bottleneck,
        }

    def reset(self) -> None:
        self._window.clear()


def split_vit_prefill(llm, image_data_uri: str, full_messages: list) -> tuple[float, float]:
    """
    通过"短序列代理法"近似分离 ViT 编码时间与 LLM prefill 时间。

    原理：
      T_vit_proxy    = 处理"图像 + 单 token 文本"的总时间
                       （文本 prefill 计算量极小，约占 0.5%，可忽略）
      T_full_prefill = 处理完整消息（图像 + 完整 prompt）的总时间
      T_llm_prefill  ≈ T_full_prefill - T_vit_proxy

    误差来源：
      1. GPU warm-up 效应（应在预热 10+ 次后调用）
      2. vLLM 调度开销（两次均含，差值中大部分抵消）
      3. ViT 与文本 prefill 的部分流水线重叠（低估 T_vit 约 3~8%）

    建议：每个配置取 20 次调用的中位数再使用。
    """
    from vllm import SamplingParams

    probe_params = SamplingParams(max_tokens=1, temperature=0.0)

    # 代理测量：图像 + 最短文本（单字符）
    probe_messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": "？"},
        ],
    }]
    t0 = time.perf_counter()
    llm.chat(probe_messages, sampling_params=probe_params)
    vit_proxy_ms = (time.perf_counter() - t0) * 1000

    # 完整 prefill 测量（max_tokens=1，只测 prefill，不测 decode）
    t0 = time.perf_counter()
    llm.chat(full_messages, sampling_params=probe_params)
    full_prefill_ms = (time.perf_counter() - t0) * 1000

    vit_encode_ms  = vit_proxy_ms
    llm_prefill_ms = max(0.0, full_prefill_ms - vit_proxy_ms)
    return vit_encode_ms, llm_prefill_ms
