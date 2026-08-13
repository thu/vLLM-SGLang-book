# inference/memory_budget.py
# KV Cache 容量计算与显存预算规划（ch11_code_02）
#
# 本模块提供精确的显存预算计算，用于在部署前确认：
#   1. 给定硬件配置下可缓存的最大文书数量
#   2. 不同文书长度分布下的 OOM 风险边界
#   3. SGLang / vLLM 参数设置的合理性验证
#
# 注意（TP 模式下的 KV Cache 存储方式）：
#   张量并行（TP）下，每张卡仅保存 num_kv_heads/tp_size 个
#   KV 头的数据。以 Qwen2-72B（8 个 KV 头）+ TP=4 为例，
#   每卡保存 2 个头的 KV Cache，4 卡合计覆盖全部 8 个头，
#   逻辑上等价于单一缓存单元，因此：
#
#   "单 token KV Cache 的逻辑大小" = 4 卡合计存储量
#       = num_layers × num_kv_heads × head_dim × 2 × dtype_bytes
#
#   可用于 KV Cache 的逻辑容量 = 4 卡可用显存之和
#   这两者相除即得最大可缓存 token 数。
#
#   此计算与实际框架行为一致：SGLang / vLLM 均按逻辑容量
#   调度缓存，不需要应用层感知每卡的物理存储布局。

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from config.model_config import (
    DEPLOY_CONFIG,
    KV_BYTES_PER_TOKEN,
    MODEL_CONFIG,
)


@dataclass
class MemoryReport:
    """单次内存预算计算结果。"""
    # 单 token KV Cache 逻辑大小（字节，4 卡合计）
    kv_bytes_per_token: int
    # 4 卡可用于 KV Cache 的逻辑总容量（字节）
    total_kv_capacity_bytes: int
    # 不同文书长度下的理论最大并发缓存文书数
    capacity_by_doc_length: dict[int, int]
    # 建议的安全缓存上限（保留 30% 余量）
    recommended_cache_limit: int


def compute_kv_bytes_per_token() -> int:
    """
    计算单 token KV Cache 的逻辑占用字节数（4 卡合计）。

    公式：
        num_layers × num_kv_heads × head_dim × 2（K+V）× dtype_bytes

    对于 Qwen2-72B（TP=4）：
        80 × 8 × 128 × 2 × 2 = 327,680 bytes
    """
    return KV_BYTES_PER_TOKEN


def compute_cache_capacity(
    tp_size: int | None = None,
    total_vram_per_card_gb: float | None = None,
    weight_per_card_gb: float | None = None,
    activation_buffer_per_card_gb: float | None = None,
    sglang_mem_fraction_static: float | None = None,
) -> MemoryReport:
    """
    计算 KV Cache 的逻辑可用容量及不同文书长度下的缓存上限。

    参数（全部可选，默认使用 DEPLOY_CONFIG 中的值）
    --------
    tp_size : int
        张量并行卡数。
    total_vram_per_card_gb : float
        单卡总显存（GB）。
    weight_per_card_gb : float
        单卡权重占用（GB）。
    activation_buffer_per_card_gb : float
        单卡激活值缓冲区预留（GB）。
    sglang_mem_fraction_static : float
        SGLang 的 mem_fraction_static 参数，用于计算静态池大小。

    返回
    ----
    MemoryReport
        完整的内存预算报告。
    """
    tp = tp_size or DEPLOY_CONFIG["tensor_parallel_size"]
    vram = (
        total_vram_per_card_gb
        or DEPLOY_CONFIG["total_vram_per_card_gb"]
    )
    weight = (
        weight_per_card_gb
        or DEPLOY_CONFIG["weight_per_card_gb"]
    )
    activation = (
        activation_buffer_per_card_gb
        or DEPLOY_CONFIG["activation_buffer_per_card_gb"]
    )
    mem_frac = (
        sglang_mem_fraction_static
        or DEPLOY_CONFIG["sglang_mem_fraction_static"]
    )

    # 单卡扣除权重后的剩余显存
    remaining_per_card_gb = vram - weight
    # SGLang 静态 KV Cache 池：mem_fraction_static × 剩余显存
    # 激活值缓冲区从剩余显存中扣除，不计入静态池
    kv_pool_per_card_gb = mem_frac * (
        remaining_per_card_gb - activation
    )
    # 4 卡合计逻辑 KV Cache 容量（字节）
    total_kv_bytes = (
        kv_pool_per_card_gb * tp * (1024 ** 3)
    )

    kv_bpt = compute_kv_bytes_per_token()

    # 计算不同文书长度下的最大并发缓存文书数
    doc_lengths = [14000, 18000, 20000, 25000, 32000]
    capacity_map: dict[int, int] = {}
    for tokens in doc_lengths:
        doc_kv_bytes = tokens * kv_bpt
        capacity_map[tokens] = int(total_kv_bytes / doc_kv_bytes)

    # 以均值文书长度（约 18K tokens）为基准计算建议上限
    # 保留 30% 余量以应对：活跃 decode 序列占用 + 内存碎片
    baseline_tokens = 18000
    theoretical_max = capacity_map[baseline_tokens]
    recommended = int(theoretical_max * 0.70)

    return MemoryReport(
        kv_bytes_per_token=kv_bpt,
        total_kv_capacity_bytes=int(total_kv_bytes),
        capacity_by_doc_length=capacity_map,
        recommended_cache_limit=recommended,
    )


def print_memory_report(report: MemoryReport | None = None) -> None:
    """
    打印格式化的内存预算报告。

    参数
    ----
    report : MemoryReport, optional
        若为 None，则自动调用 compute_cache_capacity() 生成。
    """
    if report is None:
        report = compute_cache_capacity()

    cfg = DEPLOY_CONFIG
    print("=" * 60)
    print("显存预算规划报告")
    print("=" * 60)
    print(
        f"硬件配置  ：{cfg['tensor_parallel_size']}× "
        f"A100 {cfg['total_vram_per_card_gb']}GB"
    )
    print(
        f"权重占用  ：{cfg['weight_per_card_gb']} GB/卡 "
        f"（AWQ-4bit, TP={cfg['tensor_parallel_size']}）"
    )
    print(
        f"KV Cache 逻辑容量：{
            report.total_kv_capacity_bytes / 1024**3:.1f
        } GB（4 卡合计）"
    )
    print(
        f"单 token KV Cache：{report.kv_bytes_per_token:,} bytes "
        f"（{report.kv_bytes_per_token / 1024:.1f} KB）"
    )
    print()
    print("不同文书长度下的理论最大缓存文书数：")
    for tokens, cap in report.capacity_by_doc_length.items():
        doc_gb = tokens * report.kv_bytes_per_token / 1024**3
        print(
            f"  {tokens:,} tokens（{doc_gb:.1f} GB/份）"
            f" → 最多 {cap} 份"
        )
    print()
    print(
        f"建议安全缓存上限：{report.recommended_cache_limit} 份"
        f"（保留 30% 余量）"
    )
    print("=" * 60)


def get_gpu_memory_status() -> list[dict]:
    """
    通过 nvidia-smi 获取每张 GPU 卡的实时显存使用情况。

    返回
    ----
    list[dict]
        每个元素包含：card（卡序号）、used_gb、free_gb、
        total_gb、utilization_ratio、pressure（超过 92% 时为 True）。

    异常
    ----
    RuntimeError
        若 nvidia-smi 不可用则抛出。
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,"
                "memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            f"nvidia-smi 调用失败，请确认 GPU 驱动已安装：{e}"
        ) from e

    cards = []
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split(", ")
        if len(parts) != 4:
            continue
        idx, used, free, total = map(int, parts)
        ratio = used / total if total > 0 else 0.0
        cards.append({
            "card": idx,
            "used_gb": used / 1024,
            "free_gb": free / 1024,
            "total_gb": total / 1024,
            "utilization_ratio": ratio,
            # 超过 92% 触发预警
            "pressure": ratio > 0.92,
        })
    return cards


if __name__ == "__main__":
    print_memory_report()
