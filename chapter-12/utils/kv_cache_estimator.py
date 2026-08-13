#!/usr/bin/env python3
"""
第12章：KV Cache 显存估算与 block_size 调优工具
版本基线：vLLM v0.8.x
用途：在启动服务前预估显存分配，辅助调参决策
"""

import argparse
import math
from dataclasses import dataclass
from typing import Iterator


# ──────────────────────────────────────────────
# DeepSeek V4-Flash 架构常量
# ──────────────────────────────────────────────
@dataclass
class ModelSpec:
    name: str
    num_layers: int
    num_kv_heads: int      # MLA 压缩后等效 KV 头数
    kv_head_dim: int       # 压缩 KV 维度
    hidden_size: int
    # V4 CSA/HCA 压缩因子（相对于标准全注意力）
    # 来源：V4 技术报告，1M token 场景实测 10% KV cache
    kv_compression_ratio: float


V4_FLASH_SPEC = ModelSpec(
    name="DeepSeek-V4-Flash",
    num_layers=62,
    num_kv_heads=8,
    kv_head_dim=512,
    hidden_size=7168,
    kv_compression_ratio=0.10,
)

V4_PRO_SPEC = ModelSpec(
    name="DeepSeek-V4-Pro",
    num_layers=94,
    num_kv_heads=8,
    kv_head_dim=512,
    hidden_size=7168,
    kv_compression_ratio=0.10,
)


# ──────────────────────────────────────────────
# 显存估算核心函数
# ──────────────────────────────────────────────
def estimate_kv_cache_bytes(
    spec: ModelSpec,
    seq_len: int,
    num_sequences: int,
    dtype_bytes: int = 2,   # BF16
) -> int:
    """
    估算给定并发序列数下的 KV cache 总字节数。
    V4 的 CSA+HCA 机制已将实际 KV 量压缩至标准注意力的 10%。
    """
    per_token_bytes = (
        2                           # K 和 V 两个矩阵
        * spec.num_layers
        * spec.num_kv_heads
        * spec.kv_head_dim
        * dtype_bytes
        * spec.kv_compression_ratio
    )
    return int(per_token_bytes * seq_len * num_sequences)


def estimate_weight_bytes(
    total_params_b: float,          # 十亿参数量
    active_params_b: float,         # 实际激活参数量（MoE）
    quantization: str = "awq_int4",
) -> int:
    """
    估算模型权重显存（AWQ INT4 量化）。
    AWQ INT4：约 0.5 bytes/param（含 scale/zero-point 开销）
    """
    bits_map = {
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8": 1.0,
        "awq_int4": 0.55,   # INT4 + AWQ scale/zero 附加开销
        "awq_marlin": 0.52, # Marlin 内核进一步压缩布局
    }
    bytes_per_param = bits_map.get(quantization, 2.0)
    # MoE 模型：专家权重全量存储，推理时仅激活子集
    # 权重显存 = 总参数量（非激活参数量）
    return int(total_params_b * 1e9 * bytes_per_param)


def format_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def blocks_for_seq_len(
    spec: ModelSpec,
    seq_len: int,
    block_size: int,
    dtype_bytes: int = 2,
) -> int:
    """计算单条序列所需的 KV cache 块数量。"""
    per_block_bytes = (
        2 * spec.num_layers * spec.num_kv_heads
        * spec.kv_head_dim * block_size
        * dtype_bytes * spec.kv_compression_ratio
    )
    total_bytes = (
        2 * spec.num_layers * spec.num_kv_heads
        * spec.kv_head_dim * seq_len
        * dtype_bytes * spec.kv_compression_ratio
    )
    return math.ceil(total_bytes / per_block_bytes)


def scan_block_sizes(
    spec: ModelSpec,
    total_kv_budget_gb: float,
    seq_len: int,
    block_sizes: list[int],
    dtype_bytes: int = 2,
) -> Iterator[dict]:
    """
    扫描不同 block_size 下的最大并发序列数，
    用于辅助选择最优 block_size。
    """
    budget_bytes = int(total_kv_budget_gb * 1024 ** 3)
    for bs in block_sizes:
        per_seq_bytes = estimate_kv_cache_bytes(
            spec, seq_len, 1, dtype_bytes
        )
        # 块对齐导致的碎片率
        blocks_per_seq = math.ceil(seq_len / bs)
        aligned_bytes = blocks_per_seq * bs * (
            2 * spec.num_layers * spec.num_kv_heads
            * spec.kv_head_dim * dtype_bytes
            * spec.kv_compression_ratio
        )
        fragmentation = (aligned_bytes - per_seq_bytes) / aligned_bytes
        max_concurrent = int(budget_bytes / aligned_bytes)
        yield {
            "block_size": bs,
            "per_seq_mb": aligned_bytes / 1024 ** 2,
            "fragmentation_pct": fragmentation * 100,
            "max_concurrent_seqs": max_concurrent,
        }


def print_summary(
    spec: ModelSpec,
    seq_len: int,
    tp: int,
    gpu_mem_total_gb: float,
    gpu_mem_util: float,
    quantization: str,
    total_params_b: float,
    block_sizes: list[int],
) -> None:
    """打印完整的显存分配摘要报告。"""
    weight_bytes = estimate_weight_bytes(total_params_b, 0, quantization)
    weight_gb = weight_bytes / 1024 ** 3 / tp  # 每卡权重显存

    total_budget_gb = gpu_mem_total_gb * tp * gpu_mem_util
    kv_budget_gb = total_budget_gb - weight_gb * tp
    if kv_budget_gb <= 0:
        print(f"[警告] 权重显存 ({weight_gb*tp:.1f} GB) 超过总预算，"
              f"请降低 max-model-len 或增大 TP。")
        return

    print("=" * 62)
    print(f"  {spec.name} 显存分配分析")
    print(f"  上下文长度: {seq_len:,} tokens | TP={tp} | {quantization}")
    print("=" * 62)
    print(f"  GPU 总显存        : {gpu_mem_total_gb * tp:.0f} GB "
          f"({tp}×{gpu_mem_total_gb:.0f}G)")
    print(f"  可用显存预算      : {total_budget_gb:.1f} GB "
          f"(利用率 {gpu_mem_util:.0%})")
    print(f"  模型权重占用      : {weight_gb * tp:.1f} GB（{quantization}）")
    print(f"  KV cache 可用     : {kv_budget_gb:.1f} GB")
    print()
    print(f"  {'block_size':>12} | {'单序显存(MB)':>14} | "
          f"{'碎片率':>8} | {'最大并发数':>10}")
    print("  " + "-" * 56)
    for row in scan_block_sizes(
        spec, kv_budget_gb, seq_len, block_sizes
    ):
        print(
            f"  {row['block_size']:>12} | "
            f"{row['per_seq_mb']:>14.1f} | "
            f"{row['fragmentation_pct']:>7.1f}% | "
            f"{row['max_concurrent_seqs']:>10}"
        )
    print("=" * 62)
    print("  建议：并发优先选小 block_size(16/32)；")
    print("        延迟优先选大 block_size(64/128)。")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek V4 KV cache 显存估算工具"
    )
    parser.add_argument(
        "--model", choices=["v4-flash", "v4-pro"], default="v4-flash"
    )
    parser.add_argument(
        "--seq-len", type=int, default=131072,
        help="最大上下文长度（tokens）"
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument(
        "--gpu-mem-gb", type=float, default=80.0,
        help="单卡显存（GB），A100 80G 填 80"
    )
    parser.add_argument(
        "--gpu-mem-util", type=float, default=0.90
    )
    parser.add_argument(
        "--quantization", default="awq_marlin",
        choices=["fp16", "bf16", "fp8", "awq_int4", "awq_marlin"]
    )
    args = parser.parse_args()

    spec = V4_FLASH_SPEC if args.model == "v4-flash" else V4_PRO_SPEC
    # V4-Flash：284B 总参数（AWQ 量化存储全量专家）
    total_params_b = 284.0 if args.model == "v4-flash" else 1600.0

    print_summary(
        spec=spec,
        seq_len=args.seq_len,
        tp=args.tp,
        gpu_mem_total_gb=args.gpu_mem_gb,
        gpu_mem_util=args.gpu_mem_util,
        quantization=args.quantization,
        total_params_b=total_params_b,
        block_sizes=[16, 32, 64, 128],
    )


if __name__ == "__main__":
    main()
