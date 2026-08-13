# scripts/run_benchmark.py
# 完整性能对比测试：启用缓存 vs 基线，SGLang vs vLLM（ch11_code_11）
#
# 测试流程：
#   1. 构建合成数据集（可指定文书数量）
#   2. 验证前缀一致性
#   3. 可选：运行预热阶段
#   4. 运行两阶段流水线（文书内串行 + 文书间并行）
#   5. 统计延迟分布（按缓存命中状态分组）
#   6. 输出汇总报告
#
# 使用方式：
#   # 完整测试（SGLang，180 份文书，含预热）
#   python scripts/run_benchmark.py \
#       --framework sglang --docs 180 --warmup
#
#   # 快速验证（30 份文书，无预热）
#   python scripts/run_benchmark.py \
#       --framework sglang --docs 30 --no-warmup
#
#   # 两框架对比
#   python scripts/run_benchmark.py --compare

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass

from config.prompt_config import TASK_TYPES
from data.dataset_builder import build_synthetic_dataset
from inference.sglang_server import get_sglang_base_url
from inference.vllm_server import get_vllm_base_url
from pipeline.scheduler import (
    DocumentTaskResult,
    PipelineStats,
    run_full_pipeline,
)
from utils.prefix_validator import verify_prefix_consistency


# ── 延迟分析（ch11_code_11）────────────────────────────────────
# 判定缓存命中的 hit_ratio 阈值：
# 命中请求的实际 TTFT 通常低于无缓存时的 5% 以下，
# 此处以 0.5 作为保守阈值（命中 token 数超过输入 token 数
# 50% 时判定为命中）。对于 SGLang 可从 /metrics 端点获取
# 精确的命中 token 数；对于 vLLM 使用 TTFT 推断（见
# monitoring/metrics.py 中的 infer_hit_ratio_vllm()）。
HIT_TTFT_THRESHOLD_MS = 3000.0  # 毫秒，低于此值推断为命中


def analyze_latency_by_cache_status(
    results: list[DocumentTaskResult],
) -> dict:
    """
    按缓存命中状态分组分析延迟分布。

    命中判断规则：
      TTFT < HIT_TTFT_THRESHOLD_MS 时视为缓存命中。
      此规则基于以下观察：
        命中请求的 TTFT ≈ 0.5–2 秒（仅计算查询层的少量 token）
        未命中请求的 TTFT ≈ 8–15 秒（需要完整 prefill 文书正文）
      阈值 3 秒在两类请求之间留有充裕的分离带，
      误判率在正常工况下低于 3%。

    参数
    ----
    results : list[DocumentTaskResult]
        流水线输出的所有任务结果。

    返回
    ----
    dict
        包含 cache_hit / cache_miss 两组的延迟统计，
        以及整体命中率。
    """
    hit_records = [
        r for r in results
        if r.ttft_ms < HIT_TTFT_THRESHOLD_MS
        and not r.error
    ]
    miss_records = [
        r for r in results
        if r.ttft_ms >= HIT_TTFT_THRESHOLD_MS
        and not r.error
    ]
    error_records = [r for r in results if r.error]

    def pct(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = min(int(len(s) * p / 100), len(s) - 1)
        return s[idx]

    def group_stats(group: list[DocumentTaskResult]) -> dict:
        if not group:
            return {
                "count": 0,
                "p50_ttft_ms": 0.0,
                "p95_ttft_ms": 0.0,
                "p99_ttft_ms": 0.0,
                "mean_ttft_ms": 0.0,
                "stdev_ttft_ms": 0.0,
                "p50_e2e_ms": 0.0,
                "p95_e2e_ms": 0.0,
            }
        ttfts = [r.ttft_ms for r in group]
        e2es = [r.e2e_ms for r in group]
        return {
            "count": len(group),
            "p50_ttft_ms": pct(ttfts, 50),
            "p95_ttft_ms": pct(ttfts, 95),
            "p99_ttft_ms": pct(ttfts, 99),
            "mean_ttft_ms": statistics.mean(ttfts),
            "stdev_ttft_ms": (
                statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
            ),
            "p50_e2e_ms": pct(e2es, 50),
            "p95_e2e_ms": pct(e2es, 95),
        }

    total_valid = len(hit_records) + len(miss_records)
    return {
        "cache_hit": group_stats(hit_records),
        "cache_miss": group_stats(miss_records),
        "errors": len(error_records),
        "overall_hit_ratio": (
            len(hit_records) / max(total_valid, 1)
        ),
        "ttft_speedup": (
            group_stats(miss_records)["mean_ttft_ms"]
            / max(group_stats(hit_records)["mean_ttft_ms"], 1.0)
        ),
    }


def print_benchmark_report(
    framework: str,
    stats: PipelineStats,
    latency_analysis: dict,
) -> None:
    """格式化打印完整性能对比报告。"""
    hit = latency_analysis["cache_hit"]
    miss = latency_analysis["cache_miss"]
    print(f"\n{'=' * 60}")
    print(f"性能报告：{framework.upper()}")
    print(f"{'─' * 60}")
    print(
        f"  文书总数   ：{stats.total_docs} 份\n"
        f"  任务总数   ：{stats.total_tasks} 次\n"
        f"  总耗时     ：{stats.total_wall_time_s:.1f} 秒\n"
        f"    其中预热 ：{stats.warmup_time_s:.1f} 秒\n"
        f"    其中批处理："
        f"{stats.total_wall_time_s - stats.warmup_time_s:.1f} 秒\n"
        f"{'─' * 60}\n"
        f"  整体命中率 ："
        f"{latency_analysis['overall_hit_ratio']:.1%}\n"
        f"  TTFT 加速比（未命中/命中）："
        f"{latency_analysis['ttft_speedup']:.1f}×\n"
        f"  错误请求数 ：{latency_analysis['errors']}\n"
        f"{'─' * 60}\n"
        f"  【缓存命中请求 {hit['count']} 次】\n"
        f"    TTFT P50 ：{hit['p50_ttft_ms']:.0f} ms\n"
        f"    TTFT P95 ：{hit['p95_ttft_ms']:.0f} ms\n"
        f"    TTFT 均值：{hit['mean_ttft_ms']:.0f} ms\n"
        f"    E2E  P95 ：{hit['p95_e2e_ms']:.0f} ms\n"
        f"{'─' * 60}\n"
        f"  【缓存未命中请求 {miss['count']} 次】\n"
        f"    TTFT P50 ：{miss['p50_ttft_ms']:.0f} ms\n"
        f"    TTFT P95 ：{miss['p95_ttft_ms']:.0f} ms\n"
        f"    TTFT 均值：{miss['mean_ttft_ms']:.0f} ms\n"
        f"{'=' * 60}"
    )


async def run_single_framework(
    framework: str,
    n_docs: int,
    enable_warmup: bool,
    max_concurrent_docs: int,
) -> tuple[PipelineStats, dict]:
    """运行单个框架的完整测试流程。"""
    # 构建数据集
    n_each = n_docs // 3
    docs = build_synthetic_dataset(
        n_procurement=n_each,
        n_labor=n_each,
        n_software=n_each,
    )

    # 前缀一致性验证
    sample_prompts = [
        __import__(
            "config.prompt_config",
            fromlist=["build_stable_prefix"]
        ).build_stable_prefix("risk", doc.text)
        for doc in docs[:10]
    ]
    report = verify_prefix_consistency(sample_prompts)
    if not report.is_consistent:
        print(
            f"[警告] 前缀一致性验证失败！\n"
            f"{report.summary()}"
        )
    else:
        print(
            f"前缀一致性验证通过（抽样 {len(sample_prompts)} 份）"
        )

    # 确定服务 URL
    base_url = (
        get_sglang_base_url()
        if framework == "sglang"
        else get_vllm_base_url()
    )

    # 运行两阶段流水线
    stats = await run_full_pipeline(
        docs=docs,
        base_url=base_url,
        enable_warmup=enable_warmup,
        warmup_concurrency=16,
        max_concurrent_docs=max_concurrent_docs,
        task_types=TASK_TYPES,
        max_tokens_per_task=512,
    )

    latency_analysis = analyze_latency_by_cache_status(
        stats.results
    )
    return stats, latency_analysis


async def main_async(args: argparse.Namespace) -> None:
    if args.compare:
        # 两框架对比模式
        for fw in ["sglang", "vllm"]:
            stats, analysis = await run_single_framework(
                fw, args.docs,
                not args.no_warmup,
                args.concurrency,
            )
            print_benchmark_report(fw, stats, analysis)

            out = f"benchmark_{fw}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "framework": fw,
                        "total_wall_time_s": stats.total_wall_time_s,
                        "warmup_time_s": stats.warmup_time_s,
                        "latency_analysis": analysis,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"结果已保存至 {out}")
    else:
        stats, analysis = await run_single_framework(
            args.framework, args.docs,
            not args.no_warmup,
            args.concurrency,
        )
        print_benchmark_report(args.framework, stats, analysis)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="前缀缓存性能对比测试"
    )
    parser.add_argument(
        "--framework",
        choices=["sglang", "vllm"],
        default="sglang",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=30,
        help="测试文书数量（须为 3 的倍数，默认 30）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="批处理并发文书数（默认 8）",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过预热阶段（测试冷启动性能）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="同时测试 SGLang 和 vLLM 并输出对比报告",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
