#!/usr/bin/env python3
"""
第12章：TTFT / TPOT / 吞吐量基准测试脚本
测量不同上下文长度下 DeepSeek V4-Flash 的推理性能
版本基线：vLLM v0.8.x
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import aiohttp


# ──────────────────────────────────────────────
# 测试配置
# ──────────────────────────────────────────────
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "deepseek-v4-flash"

# 测试的上下文长度梯度（tokens）
CONTEXT_LENGTHS = [4_096, 16_384, 32_768, 65_536, 131_072]

# 每个长度的并发数配置
CONCURRENCY_LEVELS = [1, 4, 8, 16]

# 每组配置的重复测试次数
REPEAT = 5

# 固定输出长度（隔离 decode 延迟变量）
OUTPUT_TOKENS = 256


@dataclass
class LatencyRecord:
    """单次请求的性能记录。"""
    context_len: int
    concurrency: int
    request_id: str
    ttft_ms: float          # 首 token 延迟（ms）
    tpot_ms: float          # 每输出 token 平均延迟（ms）
    total_latency_s: float
    prompt_tokens: int
    completion_tokens: int
    throughput_tps: float   # tokens/s（completion 维度）
    error: str = ""


@dataclass
class BenchmarkResult:
    """一组（context_len, concurrency）配置的聚合结果。"""
    context_len: int
    concurrency: int
    n_samples: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    throughput_mean_tps: float
    error_rate: float


def percentile(data: list[float], p: float) -> float:
    """计算百分位数（线性插值）。"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def generate_synthetic_prompt(target_tokens: int) -> str:
    """
    生成指定 token 数的合成提示词。
    用 Lorem Ipsum 风格的重复文本填充，
    确保 tokenizer 近似估算准确（中文约 1.5 chars/token）。
    """
    unit = (
        "在大规模语言模型推理优化领域，长上下文处理始终是核心挑战之一。"
        "DeepSeek V4 通过混合注意力机制将 KV cache 压缩至标准注意力的十分之一，"
        "使百万 token 上下文在生产环境中成为可能。"
        "vLLM 的 PagedAttention 机制则从显存管理角度提供了互补的工程优化。"
    )
    # 约 80 tokens/unit，重复至目标长度
    repeats = max(1, target_tokens // 80)
    text = unit * repeats
    return text[:target_tokens * 2]   # 粗略截断


def make_stream_payload(
    prompt: str,
    max_tokens: int = OUTPUT_TOKENS,
) -> dict:
    """构造 streaming chat completions 请求体（用于测量 TTFT）。"""
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,         # 流式模式才能准确测 TTFT
        "stream_options": {"include_usage": True},
    }


async def single_stream_request(
    session: aiohttp.ClientSession,
    context_len: int,
    concurrency: int,
) -> LatencyRecord:
    """
    执行单次流式请求，记录 TTFT、TPOT 和总延迟。
    """
    prompt = generate_synthetic_prompt(context_len)
    payload = make_stream_payload(prompt, max_tokens=OUTPUT_TOKENS)
    rid = str(uuid.uuid4())[:8]

    t_start = time.perf_counter()
    t_first_token = None
    completion_tokens = 0
    prompt_tokens = 0
    error_msg = ""

    try:
        async with session.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # 首 token 时间戳
                if t_first_token is None:
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        t_first_token = time.perf_counter()

                # 从最后一个 chunk 读取 usage
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

    except Exception as e:
        error_msg = str(e)

    t_end = time.perf_counter()

    total_latency = t_end - t_start
    ttft_ms = (
        (t_first_token - t_start) * 1000
        if t_first_token else total_latency * 1000
    )
    # TPOT = (总延迟 - TTFT) / (completion_tokens - 1)
    decode_time = total_latency - (ttft_ms / 1000)
    tpot_ms = (
        decode_time * 1000 / max(completion_tokens - 1, 1)
        if completion_tokens > 1 else 0.0
    )
    throughput_tps = completion_tokens / total_latency if total_latency > 0 else 0

    return LatencyRecord(
        context_len=context_len,
        concurrency=concurrency,
        request_id=rid,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_s=total_latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        throughput_tps=throughput_tps,
        error=error_msg,
    )


async def run_concurrency_group(
    context_len: int,
    concurrency: int,
    repeat: int,
) -> list[LatencyRecord]:
    """
    以指定并发数发送 repeat 轮请求，每轮并发 concurrency 个。
    """
    records: list[LatencyRecord] = []
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(repeat):
            batch = [
                single_stream_request(session, context_len, concurrency)
                for _ in range(concurrency)
            ]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, LatencyRecord):
                    records.append(r)
                else:
                    # 将异常记录为错误条目
                    records.append(LatencyRecord(
                        context_len=context_len,
                        concurrency=concurrency,
                        request_id="error",
                        ttft_ms=0, tpot_ms=0,
                        total_latency_s=0,
                        prompt_tokens=0, completion_tokens=0,
                        throughput_tps=0,
                        error=str(r),
                    ))
    return records


def aggregate(
    records: list[LatencyRecord],
    context_len: int,
    concurrency: int,
) -> BenchmarkResult:
    """将原始记录聚合为百分位统计结果。"""
    valid = [r for r in records if not r.error]
    n_error = len(records) - len(valid)

    ttfts = [r.ttft_ms for r in valid]
    tpots = [r.tpot_ms for r in valid]
    tpss = [r.throughput_tps for r in valid]

    return BenchmarkResult(
        context_len=context_len,
        concurrency=concurrency,
        n_samples=len(records),
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p95_ms=percentile(ttfts, 95),
        ttft_p99_ms=percentile(ttfts, 99),
        tpot_p50_ms=percentile(tpots, 50),
        tpot_p95_ms=percentile(tpots, 95),
        throughput_mean_tps=statistics.mean(tpss) if tpss else 0,
        error_rate=n_error / len(records) if records else 0,
    )


def save_results(
    results: list[BenchmarkResult],
    output_path: Path,
) -> None:
    """将聚合结果保存为 CSV，便于后续绘图。"""
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    print(f"结果已保存至 {output_path}")


def print_results(results: list[BenchmarkResult]) -> None:
    """在终端打印格式化结果表格。"""
    header = (
        f"{'CTX(K)':>8} | {'并发':>5} | "
        f"{'TTFT-P50':>10} | {'TTFT-P95':>10} | "
        f"{'TPOT-P50':>10} | {'吞吐(tps)':>10} | {'错误率':>7}"
    )
    print("\n" + "=" * 75)
    print("  DeepSeek V4-Flash 推理延迟基准测试结果")
    print("=" * 75)
    print("  " + header)
    print("  " + "-" * 73)
    for r in results:
        print(
            f"  {r.context_len//1024:>6}K | "
            f"{r.concurrency:>5} | "
            f"{r.ttft_p50_ms:>8.0f}ms | "
            f"{r.ttft_p95_ms:>8.0f}ms | "
            f"{r.tpot_p50_ms:>8.1f}ms | "
            f"{r.throughput_mean_tps:>9.1f} | "
            f"{r.error_rate:>6.1%}"
        )
    print("=" * 75)


async def run_benchmark(
    context_lengths: list[int],
    concurrency_levels: list[int],
    repeat: int,
    output_path: Path,
) -> None:
    all_results: list[BenchmarkResult] = []
    total = len(context_lengths) * len(concurrency_levels)
    idx = 0
    for ctx_len in context_lengths:
        for concurrency in concurrency_levels:
            idx += 1
            print(
                f"[{idx}/{total}] 测试 ctx={ctx_len//1024}K "
                f"并发={concurrency} ...",
                end=" ", flush=True,
            )
            records = await run_concurrency_group(ctx_len, concurrency, repeat)
            result = aggregate(records, ctx_len, concurrency)
            all_results.append(result)
            print(
                f"TTFT-P50={result.ttft_p50_ms:.0f}ms "
                f"吞吐={result.throughput_mean_tps:.1f}tps"
            )

    print_results(all_results)
    save_results(all_results, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek V4-Flash 推理基准测试"
    )
    parser.add_argument(
        "--context-lengths", nargs="+", type=int,
        default=CONTEXT_LENGTHS,
        help="测试的上下文长度列表（tokens）"
    )
    parser.add_argument(
        "--concurrency", nargs="+", type=int,
        default=CONCURRENCY_LEVELS,
    )
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark_results.csv"),
    )
    parser.add_argument(
        "--vllm-url", default=VLLM_BASE_URL
    )
    args = parser.parse_args()

    global VLLM_BASE_URL
    VLLM_BASE_URL = args.vllm_url

    asyncio.run(run_benchmark(
        context_lengths=args.context_lengths,
        concurrency_levels=args.concurrency,
        repeat=args.repeat,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
