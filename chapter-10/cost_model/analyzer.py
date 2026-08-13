"""
文件: chapter10/cost_model/analyzer.py

混合架构与单框架方案的成本对比计算工具。

使用说明:
  本模块提供成本对比的计算框架，所有参数均需替换为实际测量值后才有参考意义。
  参数来源建议:
    - throughput_*: 在目标硬件上运行 chapter05 的基准测试脚本获取实测值
    - gpu_hourly_usd_per_card: 查询当前云厂商控制台的 Spot/按需价格
      AWS: https://aws.amazon.com/ec2/spot/pricing/
      GCP: https://cloud.google.com/compute/docs/gpus/gpu-regions-zones
      Azure: https://azure.microsoft.com/en-us/pricing/details/virtual-machines/

价格参数单位说明（重要）:
  gpu_hourly_usd_per_card 的单位是【单张GPU卡的小时成本】，
  而非实例整体价格。从云厂商价格换算方式如下：
    AWS p4d.24xlarge（8×A100 40GB）按需价格 ≈ $32.77/h
    → 单卡成本 = $32.77 / 8 ≈ $4.10/卡/小时
  注意：A100 40GB 与 80GB 规格价格不同，使用 p4de.24xlarge（8×A100 80GB）
  的单卡成本约 $5.20/卡/小时（2025年初 us-east-1 参考值，以实际报价为准）。

Spot折扣说明:
  Spot/Preemptible实例价格历史波动较大，建议查看过去30天的Spot价格历史
  取均值而非使用当前瞬时价格，以获得更稳定的成本估算结果。
  AWS Spot 折扣通常在 50-70%（即按需价格的 30-50%）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── 配置数据类 ────────────────────────────────────────────────────────────

@dataclass
class WorkloadProfile:
    """
    工作负载分布配置。
    三类请求比例之和应为 1.0。
    """
    short_text_ratio: float = 0.60       # 高并发短文本（智能客服、实时问答）
    structured_ratio: float = 0.25       # 结构化生成（合同抽取、报表生成）
    long_context_ratio: float = 0.15     # 长上下文分析（文档摘要、RAG）
    total_rps: float = 100.0             # 峰值总请求量（req/s）

    def __post_init__(self):
        total = self.short_text_ratio + self.structured_ratio + self.long_context_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Request ratios must sum to 1.0, got {total:.4f}. "
                f"Adjust short_text_ratio, structured_ratio, long_context_ratio."
            )


@dataclass
class FrameworkThroughput:
    """
    各框架在不同工作负载类型下的实测吞吐量（req/s/GPU）。

    !! 重要: 所有字段均为占位示例值，必须替换为在目标硬件和模型规模下的实测数据。
    !!        示例值基于 Qwen2.5-7B-Instruct，单卡 A100 80GB，推理长度约128/512 tokens。
    !!        不同模型规模、精度、硬件配置下实测值会有显著差异。
    """
    # vLLM v0.8.x 实测吞吐量参考范围（需在目标环境实测替换）
    vllm_short_rps_per_gpu: float = 80.0    # 短文本（input≈64, output≈128 tokens）
    vllm_structured_rps_per_gpu: float = 18.0  # 结构化（含schema约束，性能低于无约束）
    vllm_long_ctx_rps_per_gpu: float = 4.0  # 长上下文（input≈8192, output≈512 tokens）

    # SGLang v0.5.x 实测吞吐量参考范围
    sglang_short_rps_per_gpu: float = 65.0   # 短文本（略低于vLLM，调度开销稍高）
    sglang_structured_rps_per_gpu: float = 32.0  # 结构化（原生constrained decoding优势）
    sglang_long_ctx_rps_per_gpu: float = 10.0   # 长上下文（RadixAttention前缀缓存复用）


@dataclass
class PricingConfig:
    """
    GPU 实例定价配置。

    单位: 单张GPU卡的小时成本（USD/卡/小时），非实例整体价格。
    换算方式: 实例价格 ÷ 实例内GPU卡数 = 单卡价格。

    示例换算（仅供参考，以实际报价为准）:
      AWS p4de.24xlarge（8×A100 80GB）按需 ≈ $41.61/h → 单卡 ≈ $5.20/卡/h
      AWS p4de.24xlarge Spot 历史均值 ≈ $12.50-$16.00/h → 单卡 ≈ $1.56-$2.00/卡/h
    """
    # 按需实例单卡小时成本（用于生产实时服务层）
    on_demand_usd_per_gpu_hour: float = 5.20   # 替换为实际报价

    # Spot实例单卡小时成本（用于批处理层）
    # 建议取过去30天Spot价格历史均值，而非当前瞬时价格
    spot_usd_per_gpu_hour: float = 1.80         # 替换为Spot历史均值

    hours_per_month: float = 730.0              # 月均小时数（365×24÷12 ≈ 730）


@dataclass
class CostComparisonResult:
    """成本对比计算结果"""
    # 纯 vLLM 方案
    vllm_only_gpu_count: float
    vllm_only_monthly_usd: float

    # 混合架构方案（生产层按需 + 批处理层Spot）
    hybrid_vllm_gpu_count: float
    hybrid_sglang_gpu_count: float
    hybrid_total_gpu_count: float
    hybrid_monthly_usd: float

    # 对比结果
    saving_usd_per_month: float
    saving_pct: float

    # 辅助信息
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  混合架构 vs 纯vLLM 月成本对比（示例计算）",
            "=" * 60,
            "",
            "【纯 vLLM 方案】",
            f"  所需GPU数量: {self.vllm_only_gpu_count:.1f} 卡",
            f"  月成本估算:  ${self.vllm_only_monthly_usd:,.0f}",
            "",
            "【混合架构方案】",
            f"  vLLM GPU数量:   {self.hybrid_vllm_gpu_count:.1f} 卡",
            f"  SGLang GPU数量: {self.hybrid_sglang_gpu_count:.1f} 卡",
            f"  合计GPU数量:    {self.hybrid_total_gpu_count:.1f} 卡",
            f"  月成本估算:     ${self.hybrid_monthly_usd:,.0f}",
            "",
            "【成本节约】",
            f"  节约金额: ${self.saving_usd_per_month:,.0f}/月",
            f"  节约比例: {self.saving_pct:.1f}%",
            "",
        ]
        if self.notes:
            lines.append("【注意事项】")
            for note in self.notes:
                lines.append(f"  * {note}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── 计算函数 ──────────────────────────────────────────────────────────────

def _gpu_count_with_headroom(rps: float, throughput_per_gpu: float,
                              headroom: float = 0.20) -> float:
    """
    计算满足目标吞吐量所需的GPU数量，附加余量。

    Args:
        rps:               目标吞吐量（req/s）
        throughput_per_gpu: 单卡实测吞吐量（req/s/GPU）
        headroom:           资源余量比例（默认20%，应对流量抖动）

    Returns:
        向上取整后的GPU数量
    """
    if throughput_per_gpu <= 0:
        raise ValueError(f"throughput_per_gpu must be positive, got {throughput_per_gpu}")
    import math
    return math.ceil((rps / throughput_per_gpu) * (1 + headroom))


def compute_cost_comparison(
    workload: WorkloadProfile,
    throughput: FrameworkThroughput,
    pricing: PricingConfig,
    batch_workload_ratio: float = 0.30,
    spot_discount_rate: float = 0.35,
) -> CostComparisonResult:
    """
    计算纯vLLM方案与混合架构方案的月成本对比。

    混合架构的成本优势来自两个来源：
      1. 工作负载与框架的精准匹配（SGLang处理结构化和长上下文任务效率更高）
      2. 批处理层使用Spot实例（按需价格的30-50%）

    Args:
        workload:             工作负载分布配置
        throughput:           各框架实测吞吐量
        pricing:              GPU实例定价（单卡小时成本）
        batch_workload_ratio: 批处理任务占总算力需求的比例（用于Spot节约计算）
        spot_discount_rate:   Spot实例相对按需价格的折扣率（如0.35表示35%）

    Returns:
        CostComparisonResult: 包含详细成本数据和节约分析
    """
    # 各类型请求的绝对吞吐量需求（req/s）
    short_rps = workload.total_rps * workload.short_text_ratio
    struct_rps = workload.total_rps * workload.structured_ratio
    long_rps = workload.total_rps * workload.long_context_ratio

    # ── 纯 vLLM 方案：三类请求均由 vLLM 处理 ────────────────────────────
    gpu_vllm_short = _gpu_count_with_headroom(short_rps, throughput.vllm_short_rps_per_gpu)
    gpu_vllm_struct = _gpu_count_with_headroom(struct_rps, throughput.vllm_structured_rps_per_gpu)
    gpu_vllm_long = _gpu_count_with_headroom(long_rps, throughput.vllm_long_ctx_rps_per_gpu)
    total_gpu_vllm_only = gpu_vllm_short + gpu_vllm_struct + gpu_vllm_long

    monthly_vllm_only = (
        total_gpu_vllm_only
        * pricing.on_demand_usd_per_gpu_hour
        * pricing.hours_per_month
    )

    # ── 混合架构方案：短文本→vLLM，结构化+长上下文→SGLang ───────────────
    # vLLM 只处理短文本请求
    gpu_hybrid_vllm = _gpu_count_with_headroom(short_rps, throughput.vllm_short_rps_per_gpu)

    # SGLang 处理结构化生成和长上下文请求（两者在SGLang上的吞吐量更高）
    gpu_hybrid_sglang_struct = _gpu_count_with_headroom(
        struct_rps, throughput.sglang_structured_rps_per_gpu
    )
    gpu_hybrid_sglang_long = _gpu_count_with_headroom(
        long_rps, throughput.sglang_long_ctx_rps_per_gpu
    )
    gpu_hybrid_sglang = gpu_hybrid_sglang_struct + gpu_hybrid_sglang_long
    total_gpu_hybrid = gpu_hybrid_vllm + gpu_hybrid_sglang

    # 成本计算：生产实时服务层使用按需实例，批处理层使用Spot实例
    # 简化假设：批处理工作负载占总GPU需求的 batch_workload_ratio 比例
    on_demand_gpu = total_gpu_hybrid * (1 - batch_workload_ratio)
    spot_gpu = total_gpu_hybrid * batch_workload_ratio

    monthly_hybrid = (
        on_demand_gpu * pricing.on_demand_usd_per_gpu_hour * pricing.hours_per_month
        + spot_gpu * pricing.spot_usd_per_gpu_hour * pricing.hours_per_month
    )

    saving = monthly_vllm_only - monthly_hybrid
    saving_pct = (saving / monthly_vllm_only * 100) if monthly_vllm_only > 0 else 0.0

    notes = [
        "所有吞吐量参数为示例占位值，必须替换为目标环境的实测数据。",
        f"on_demand_usd_per_gpu_hour={pricing.on_demand_usd_per_gpu_hour} 为单张GPU卡的小时成本，"
        "请勿混用实例整体价格。",
        "Spot节约假设批处理任务可容忍中断并实现检查点续跑。",
        "实际节约幅度受Spot可用性、中断率和任务重试成本影响。",
    ]

    return CostComparisonResult(
        vllm_only_gpu_count=total_gpu_vllm_only,
        vllm_only_monthly_usd=monthly_vllm_only,
        hybrid_vllm_gpu_count=gpu_hybrid_vllm,
        hybrid_sglang_gpu_count=gpu_hybrid_sglang,
        hybrid_total_gpu_count=total_gpu_hybrid,
        hybrid_monthly_usd=monthly_hybrid,
        saving_usd_per_month=saving,
        saving_pct=saving_pct,
        notes=notes,
    )


# ── 示例入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 示例：使用占位参数运行计算
    # 实际使用时，将各参数替换为实测值和当前云厂商报价
    workload = WorkloadProfile(
        short_text_ratio=0.60,
        structured_ratio=0.25,
        long_context_ratio=0.15,
        total_rps=100.0,
    )
    throughput = FrameworkThroughput()   # 使用占位示例值
    pricing = PricingConfig(
        on_demand_usd_per_gpu_hour=5.20,  # 单卡按需价格（需替换为实际报价）
        spot_usd_per_gpu_hour=1.80,        # 单卡Spot价格历史均值（需替换）
    )

    result = compute_cost_comparison(
        workload=workload,
        throughput=throughput,
        pricing=pricing,
        batch_workload_ratio=0.30,
        spot_discount_rate=0.35,
    )
    print(result.summary())
