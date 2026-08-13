# 文件名: scripts/common/metrics.py
# 功能: 单请求性能指标的数据类定义，供 vLLM 和 SGLang 测试共用
# 说明: 两个框架的测试脚本均导入此模块，确保指标语义和精度完全一致

import dataclasses
from typing import Dict


@dataclasses.dataclass
class RequestMetrics:
    """
    记录单个请求的完整性能数据。

    时间戳字段（submit_time、first_token_time、completion_time）
    均使用 time.perf_counter() 采集，单位为秒。perf_counter 使用
    操作系统最高精度单调时钟，不受 NTP 校时影响，差值始终非负。

    TTFT 定义为 SSE 流中第一个包含非空 token 文本的数据帧到达时刻
    与请求提交时刻之间的差值，反映模型真正开始产出内容的等待时间。
    """

    # ---- 请求标识 ----
    request_id: int
    scenario: str

    # ---- 输入输出规模 ----
    input_length: int    # 输入 token 数
    output_length: int   # 实际输出 token 数

    # ---- 关键时间戳（perf_counter，单位秒） ----
    submit_time: float        # 请求提交（HTTP POST 发出前）
    first_token_time: float   # 首个非空 token 到达客户端
    completion_time: float    # SSE 流结束（收到 [DONE]）

    # ---- 状态 ----
    success: bool = True
    error_msg: str = ""

    # ------------------------------------------------------------------
    # 计算属性
    # ------------------------------------------------------------------

    @property
    def ttft_ms(self) -> float:
        """
        首 token 延迟（Time To First Token），单位毫秒。

        在交互式应用中，TTFT 直接决定用户感受到的响应速度。
        """
        return (self.first_token_time - self.submit_time) * 1000.0

    @property
    def total_latency_ms(self) -> float:
        """
        端到端总延迟，单位毫秒。

        从请求提交到全部 token 生成完毕的完整耗时。
        """
        return (self.completion_time - self.submit_time) * 1000.0

    @property
    def decode_latency_ms(self) -> float:
        """
        解码阶段延迟，单位毫秒。

        从首 token 到达到全部 token 生成完毕，反映纯解码速度。
        total_latency = TTFT + decode_latency
        """
        return (self.completion_time - self.first_token_time) * 1000.0

    @property
    def throughput_tps(self) -> float:
        """
        单请求输出吞吐量，单位 tokens/s。

        用总延迟除输出 token 数，反映解码速度。
        注意：系统整体吞吐量需在测试框架层统一计算，
        不等于单请求吞吐量的简单累加。
        """
        duration = self.completion_time - self.submit_time
        if duration <= 0 or self.output_length <= 0:
            return 0.0
        return self.output_length / duration

    @property
    def decode_throughput_tps(self) -> float:
        """
        纯解码阶段吞吐量，单位 tokens/s。

        排除预填充时间后的解码速度，更准确地反映解码效率。
        """
        duration = self.completion_time - self.first_token_time
        if duration <= 0 or self.output_length <= 0:
            return 0.0
        return self.output_length / duration

    def to_dict(self) -> Dict:
        """
        将指标转换为可 JSON 序列化的字典，包含全部计算属性。
        """
        d = dataclasses.asdict(self)
        d["ttft_ms"] = self.ttft_ms
        d["total_latency_ms"] = self.total_latency_ms
        d["decode_latency_ms"] = self.decode_latency_ms
        d["throughput_tps"] = self.throughput_tps
        d["decode_throughput_tps"] = self.decode_throughput_tps
        return d
