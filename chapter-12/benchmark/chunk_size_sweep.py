#!/usr/bin/env python3
"""
第12章：Chunked Prefill chunk_size 参数扫描工具
通过测量不同 chunk_size 下的 TTFT 和 GPU 显存峰值，
辅助选择最优 max_num_batched_tokens 配置。
版本基线：vLLM v0.8.x
"""

import argparse
import asyncio
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp


VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "deepseek-v4-flash"

# chunked prefill 测试的 chunk_size 候选值
CHUNK_SIZES = [512, 1024, 2048, 4096, 8192, 16384]

# 测试用的长提示词长度（模拟真实长文档场景）
TEST_CONTEXT_LEN = 32768
OUTPUT_TOKENS = 128
REPEAT = 3


@dataclass
class ChunkSweepRecord:
    chunk_size: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    decode_throughput_tps: float
    # GPU 显存峰值（通过 nvidia-smi 采样，单位 MiB）
    gpu_mem_peak_mib: int


def get_gpu_memory_mib() -> int:
    """通过 nvidia-smi 获取当前 GPU 0 的显存占用（MiB）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def make_prompt(token_count: int) -> str:
    """生成约 token_count 个 token 的合成提示词。"""
    unit = (
        "请分析以下技术文档的核心要点，重点关注架构创新和性能数据。"
        "DeepSeek V4 的混合注意力机制将长上下文推理效率提升至前所未有的水平。"
    )
    repeats = max(1, token_count // 40)
    return (unit * repeats)[:token_count * 3]


async def measure_ttft_stream(
    session: aiohttp.ClientSession,
    context_len: int,
) -> tuple[float, float]:
    """
    发送流式请求，返回 (ttft_ms, tpot_ms)。
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": make_prompt(context_len)}],
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first = None
    completion_tokens = 0

    async with session.post(
        f"{VLLM_BASE_URL}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if t_first is None:
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    t_first = time.perf_counter()

            if "usage" in chunk and chunk["usage"]:
                completion_tokens = chunk["usage"].get(
                    "completion_tokens", 0
                )

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft_ms = (t_first - t_start) * 1000 if t_first else total * 1000
    decode_s = total - ttft_ms / 1000
    tpot_ms = (
        decode_s * 1000 / max(completion_tokens - 1, 1)
        if completion_tokens > 1 else 0.0
    )
    return ttft_ms, tpot_ms


async def sweep_single_chunk_size(
    chunk_size: int,
    repeat: int,
) -> ChunkSweepRecord:
    """
    针对单个 chunk_size 执行 repeat 次测试，返回聚合结果。
    注意：此函数假设 vLLM 服务已以对应 chunk_size 启动。
    """
    ttfts, tpots = [], []
    mem_samples = []

    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(repeat):
            mem_before = get_gpu_memory_mib()
            ttft, tpot = await measure_ttft_stream(session, TEST_CONTEXT_LEN)
            mem_after = get_gpu_memory_mib()
            ttfts.append(ttft)
            tpots.append(tpot)
            mem_samples.append(max(mem_before, mem_after))

    ttfts_sorted = sorted(ttfts)
    p50_idx = len(ttfts_sorted) // 2
    p95_idx = int(len(ttfts_sorted) * 0.95)

    return ChunkSweepRecord(
        chunk_size=chunk_size,
        ttft_p50_ms=ttfts_sorted[p50_idx],
        ttft_p95_ms=ttfts_sorted[min(p95_idx, len(ttfts_sorted) - 1)],
        decode_throughput_tps=(
            OUTPUT_TOKENS / (sum(tpots) / len(tpots) / 1000)
            if tpots else 0.0
        ),
        gpu_mem_peak_mib=max(mem_samples) if mem_samples else -1,
    )


def print_sweep_table(records: list[ChunkSweepRecord]) -> None:
    """打印 chunk_size 扫描结果对比表。"""
    print("\n" + "=" * 68)
    print("  Chunked Prefill chunk_size 参数扫描结果")
    print(f"  测试上下文长度：{TEST_CONTEXT_LEN // 1024}K tokens")
    print("=" * 68)
    print(
        f"  {'chunk_size':>12} | {'TTFT-P50':>10} | "
        f"{'TTFT-P95':>10} | {'Decode-TPS':>11} | {'GPU-Mem':>9}"
    )
    print("  " + "-" * 66)
    for r in records:
        mem_str = f"{r.gpu_mem_peak_mib} MiB" if r.gpu_mem_peak_mib > 0 else "N/A"
        print(
            f"  {r.chunk_size:>12} | "
            f"{r.ttft_p50_ms:>8.0f}ms | "
            f"{r.ttft_p95_ms:>8.0f}ms | "
            f"{r.decode_throughput_tps:>10.1f} | "
            f"{mem_str:>9}"
        )
    print("=" * 68)

    # 给出建议
    if records:
        best_ttft = min(records, key=lambda r: r.ttft_p50_ms)
        best_tps = max(records, key=lambda r: r.decode_throughput_tps)
        print(f"\n  延迟最优 chunk_size : {best_ttft.chunk_size} "
              f"(TTFT-P50={best_ttft.ttft_p50_ms:.0f}ms)")
        print(f"  吞吐最优 chunk_size : {best_tps.chunk_size} "
              f"(Decode={best_tps.decode_throughput_tps:.1f} tps)")


async def run_sweep(
    chunk_sizes: list[int],
    repeat: int,
) -> None:
    """
    执行完整的 chunk_size 扫描。
    注意：每个 chunk_size 需要重启 vLLM 服务，此脚本仅负责测量，
    请在外部脚本中配合 launch_v4_flash.py 完成服务重启。
    """
    records = []
    for cs in chunk_sizes:
        print(f"测试 chunk_size={cs} ...", end=" ", flush=True)
        r = await sweep_single_chunk_size(cs, repeat)
        records.append(r)
        print(f"TTFT-P50={r.ttft_p50_ms:.0f}ms")
    print_sweep_table(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked Prefill chunk_size 扫描工具"
    )
    parser.add_argument(
        "--chunk-sizes", nargs="+", type=int, default=CHUNK_SIZES
    )
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--vllm-url", default=VLLM_BASE_URL)
    args = parser.parse_args()

    global VLLM_BASE_URL
    VLLM_BASE_URL = args.vllm_url

    asyncio.run(run_sweep(args.chunk_sizes, args.repeat))


if __name__ == "__main__":
    main()
