#!/usr/bin/env python3
# scripts/benchmark_throughput.py
# 吞吐量基准测试
# 使用方法：
#   python scripts/benchmark_throughput.py --input data/extracted/ --n 200
#   python scripts/benchmark_throughput.py --input data/extracted/ --n 200 --concurrency 10 20 30
#
# 测试逻辑：
# 遍历多个并发数，绘制吞吐量-并发数曲线，找到边际收益开始下降的拐点。
# 结果保存至 results/benchmark_throughput.json。

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.sglang_extractor import extract_resume
from scripts.verify_radix_attention import generate_test_resumes


async def run_single_benchmark(
    texts: list[str],
    max_concurrent: int,
) -> dict:
    """
    在指定并发数下测量吞吐量。

    Args:
        texts:          测试简历文本列表
        max_concurrent: 最大并发请求数

    Returns:
        包含并发数、成功数、耗时、吞吐量的字典
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def one(text: str):
        async with sem:
            return await extract_resume(text, session)

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[one(t) for t in texts],
            return_exceptions=True,
        )
    elapsed = time.perf_counter() - t0

    success    = sum(1 for r in results if not isinstance(r, Exception))
    failed     = len(results) - success
    throughput = success / elapsed if elapsed > 0 else 0.0
    per_hour   = throughput * 3600

    return {
        "concurrency":        max_concurrent,
        "n_success":          success,
        "n_failed":           failed,
        "n_total":            len(texts),
        "elapsed_s":          round(elapsed, 2),
        "throughput_rps":     round(throughput, 3),
        "throughput_per_h":   round(per_hour, 0),
    }


async def sweep_concurrency(
    texts: list[str],
    concurrency_levels: list[int],
) -> list[dict]:
    """
    遍历多个并发数，收集吞吐量曲线数据。

    Args:
        texts:               测试文本列表
        concurrency_levels:  要测试的并发数列表

    Returns:
        每个并发数对应的测试结果列表
    """
    results = []
    for c in concurrency_levels:
        print(f"\n  测试并发数 {c}（共 {len(texts)} 份简历）...")
        r = await run_single_benchmark(texts, max_concurrent=c)
        results.append(r)
        print(
            f"  并发={c:>3d} | 成功={r['n_success']}/{r['n_total']} | "
            f"耗时={r['elapsed_s']:.1f}s | "
            f"吞吐={r['throughput_per_h']:.0f} 份/小时"
        )
    return results


def print_summary(results: list[dict]) -> None:
    """打印汇总表格和拐点分析"""
    print("\n" + "=" * 60)
    print("吞吐量-并发数曲线汇总")
    print("=" * 60)
    print(f"  {'并发数':>6}  {'吞吐(份/小时)':>14}  {'成功率':>8}  {'建议'}")
    print("  " + "-" * 50)

    max_tp     = max(r["throughput_per_h"] for r in results) if results else 0
    prev_tp    = 0.0
    plateau_c  = None   # 吞吐量趋于平坦的并发数

    for r in results:
        c      = r["concurrency"]
        tp     = r["throughput_per_h"]
        rate   = r["n_success"] / r["n_total"] if r["n_total"] > 0 else 0.0
        growth = (tp - prev_tp) / prev_tp * 100 if prev_tp > 0 else 0.0

        # 增幅低于 5% 视为进入平坦区
        is_plateau = prev_tp > 0 and growth < 5.0
        flag       = "← 推荐（边际收益开始下降）" if is_plateau and plateau_c is None else ""
        if is_plateau and plateau_c is None:
            plateau_c = c

        bar_len = int(tp / max_tp * 20) if max_tp > 0 else 0
        bar     = "█" * bar_len

        print(f"  {c:>6}  {tp:>14.0f}  {rate:>8.1%}  {bar} {flag}")
        prev_tp = tp

    if plateau_c:
        print(f"\n  推荐并发数：{plateau_c}（吞吐量已趋于饱和，继续增加不显著提升）")
    else:
        print("\n  未检测到明显拐点，可继续增大并发数测试。")

    print("=" * 60)


def load_texts_from_dir(input_dir: Path, n: int) -> list[str]:
    """从目录加载提取好的文本文件，不足则用合成简历补充"""
    texts = []
    for txt_file in sorted(input_dir.glob("*.txt"))[:n]:
        try:
            texts.append(txt_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if len(texts) < n:
        needed = n - len(texts)
        print(f"  目录中文本文件不足 {n} 份，用 {needed} 份合成简历补充")
        texts += generate_test_resumes(needed)

    return texts[:n]


def main():
    parser = argparse.ArgumentParser(description="吞吐量基准测试")
    parser.add_argument(
        "--input", type=str, default=None,
        help="提取好的 .txt 文件目录（不指定则使用合成简历）"
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="测试简历数量（默认 200）"
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[5, 10, 20, 30, 50],
        help="要测试的并发数列表（默认 5 10 20 30 50）"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/benchmark_throughput.json",
        help="结果保存路径"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("吞吐量基准测试")
    print("=" * 60)
    print(f"  简历数量：{args.n}")
    print(f"  并发数列表：{args.concurrency}")

    # 加载或生成测试文本
    if args.input:
        input_dir = Path(args.input)
        print(f"  数据来源：{input_dir}")
        texts = load_texts_from_dir(input_dir, args.n)
    else:
        print("  数据来源：合成简历（未指定 --input）")
        texts = generate_test_resumes(args.n)
    print(f"  实际测试数量：{len(texts)}")

    # 执行测试
    results = asyncio.run(sweep_concurrency(texts, args.concurrency))

    # 打印汇总
    print_summary(results)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": vars(args), "results": results},
            f, ensure_ascii=False, indent=2
        )
    print(f"\n结果已保存：{output_path}")


if __name__ == "__main__":
    main()
