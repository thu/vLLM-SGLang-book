#!/usr/bin/env python3
"""
第12章：DeepSeek V4-Flash vLLM 推理服务启动脚本
版本基线：vLLM v0.8.x，DeepSeek-V4-Flash AWQ
硬件环境：4×NVIDIA A100 80G，NVLink 互联
"""

import argparse
import subprocess
import sys
import os


# ──────────────────────────────────────────────
# 默认服务配置：针对 V4-Flash 长上下文优化
# ──────────────────────────────────────────────
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-AWQ"
DEFAULT_PORT = 8000
DEFAULT_TP = 4                    # 张量并行度，匹配 4×A100
DEFAULT_MAX_MODEL_LEN = 131072    # 默认 128K；可调至 1M
DEFAULT_GPU_MEM_UTIL = 0.90       # 留 10% 余量给 CUDA 运行时
DEFAULT_BLOCK_SIZE = 32           # KV cache 块大小（tokens/块）
DEFAULT_CHUNK_SIZE = 4096         # chunked prefill 每块 token 数
DEFAULT_QUANTIZATION = "awq_marlin"  # V4-Flash AWQ 量化后端


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    """
    构建 vllm serve 命令行参数列表。
    返回可直接传入 subprocess 的参数列表。
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--block-size", str(args.block_size),
        "--quantization", args.quantization,
        "--dtype", "auto",
        "--trust-remote-code",
        "--served-model-name", "deepseek-v4-flash",
        # 启用 chunked prefill，控制长序列 prefill 显存峰值
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(args.chunk_size),
        # 禁用前缀缓存（长上下文场景命中率低，避免额外开销）
        "--enable-prefix-caching", "False",
        # 日志级别
        "--uvicorn-log-level", "info",
    ]

    # 若指定了 chat template，追加参数
    if args.chat_template:
        cmd += ["--chat-template", args.chat_template]

    return cmd


def print_config(args: argparse.Namespace) -> None:
    """打印启动配置摘要，便于运维确认。"""
    kv_cache_gb = estimate_kv_cache_gb(
        seq_len=args.max_model_len,
        num_layers=62,          # V4-Flash 层数
        num_heads=8,            # V4-Flash MLA 压缩 KV 头数
        head_dim=512,           # V4-Flash KV 压缩维度
        tp=args.tp,
    )
    print("=" * 60)
    print("  DeepSeek V4-Flash vLLM 服务配置")
    print("=" * 60)
    print(f"  模型路径        : {args.model}")
    print(f"  张量并行度      : {args.tp}×A100")
    print(f"  最大上下文长度  : {args.max_model_len:,} tokens")
    print(f"  GPU 显存利用率  : {args.gpu_mem_util:.0%}")
    print(f"  KV cache 块大小 : {args.block_size} tokens/块")
    print(f"  Chunked prefill : {args.chunk_size} tokens/chunk")
    print(f"  量化方式        : {args.quantization}")
    print(f"  预估 KV 显存    : {kv_cache_gb:.1f} GB（{args.tp} 卡合计）")
    print(f"  服务端口        : {args.port}")
    print("=" * 60)


def estimate_kv_cache_gb(
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    tp: int,
    dtype_bytes: int = 2,  # FP16/BF16
) -> float:
    """
    估算 KV cache 显存占用（全部 GPU 合计）。
    公式：2 × layers × heads × head_dim × seq_len × dtype_bytes / TP
    注：V4 的 CSA/HCA 机制已大幅压缩 KV，此处按压缩后参数计算。
    """
    # V4 CSA 层压缩率约为 1/4，HCA 层压缩率约为 1/16；
    # 此处取等效压缩因子 0.10（对应官方报告的 10% KV cache）
    compression_factor = 0.10
    total_bytes = (
        2 * num_layers * num_heads * head_dim
        * seq_len * dtype_bytes * compression_factor
    )
    return total_bytes / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="启动 DeepSeek V4-Flash vLLM 推理服务"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tp", type=int, default=DEFAULT_TP)
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN
    )
    parser.add_argument(
        "--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL
    )
    parser.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE
    )
    parser.add_argument(
        "--quantization", default=DEFAULT_QUANTIZATION
    )
    parser.add_argument("--chat-template", default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印命令，不实际启动服务"
    )
    args = parser.parse_args()

    print_config(args)
    cmd = build_vllm_command(args)

    if args.dry_run:
        print("\n[dry-run] 启动命令：")
        print(" ".join(cmd))
        return

    # 设置 CUDA 可见设备（使用所有可用 GPU）
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(args.tp)))

    print("\n正在启动 vLLM 服务，按 Ctrl+C 终止...\n")
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n服务已手动终止。")
    except subprocess.CalledProcessError as e:
        print(f"\n服务异常退出，返回码：{e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
