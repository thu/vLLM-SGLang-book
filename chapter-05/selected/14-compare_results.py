# 文件名: scripts/common/compare_results.py
# 功能: 加载 vLLM 和 SGLang 的最新测试结果，逐指标对比并生成报告
# 用法: python compare_results.py [--scenario SCENARIO]

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def load_latest_stats(framework: str, scenario: str) -> Optional[dict]:
    """
    在 results/<framework>/ 目录下查找 <scenario>_stats_*.json，
    按文件修改时间排序，返回最新文件的内容。
    """
    results_dir = (
        Path.home() / "llm-benchmark" / "results" / framework
    )
    if not results_dir.exists():
        return None

    files = sorted(
        results_dir.glob(f"{scenario}_stats_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return None

    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def compare_scenario(scenario: str) -> Dict:
    """
    对单个场景执行对比分析。

    改进幅度的定义统一为"正值 = SGLang 更优"：
    - 延迟类指标（越小越好）: diff = (vLLM - SGLang) / vLLM × 100%
    - 吞吐量指标（越大越好）: diff = (SGLang - vLLM) / vLLM × 100%
    """
    vllm_stats = load_latest_stats("vllm", scenario)
    sglang_stats = load_latest_stats("sglang", scenario)

    if not vllm_stats:
        print(f"  ⚠ 未找到 vLLM 场景 {scenario} 的结果")
        return {}
    if not sglang_stats:
        print(f"  ⚠ 未找到 SGLang 场景 {scenario} 的结果")
        return {}

    # 要对比的指标：(键名, 显示标签, 是否越小越好)
    metrics = [
        ("ttft_p50_ms",            "TTFT P50 (ms)",        True),
        ("ttft_p95_ms",            "TTFT P95 (ms)",        True),
        ("ttft_p99_ms",            "TTFT P99 (ms)",        True),
        ("latency_p50_ms",         "总延迟 P50 (ms)",      True),
        ("latency_p95_ms",         "总延迟 P95 (ms)",      True),
        ("throughput_mean_tps",    "平均吞吐量 (tok/s)",   False),
        ("system_throughput_tps",  "系统吞吐量 (tok/s)",   False),
    ]

    comparison = {"scenario": scenario, "metrics": {}}

    print(f"\n{'='*70}")
    print(f"场景对比: {scenario}")
    print(f"{'='*70}")
    header = (f"{'指标':<25} {'vLLM':>12} {'SGLang':>12} "
              f"{'SGLang 优势':>14}")
    print(header)
    print("-" * 70)

    for key, label, lower_is_better in metrics:
        v = vllm_stats.get(key, 0.0)
        s = sglang_stats.get(key, 0.0)

        if v > 0:
            if lower_is_better:
                diff_pct = (v - s) / v * 100.0
            else:
                diff_pct = (s - v) / v * 100.0
        else:
            diff_pct = 0.0

        sign = "+" if diff_pct >= 0 else ""
        if diff_pct > 2:
            marker = "✓"
        elif diff_pct < -2:
            marker = "✗"
        else:
            marker = "="

        print(
            f"{label:<25} {v:>12.1f} {s:>12.1f} "
            f"{marker} {sign}{diff_pct:.1f}%"
        )

        comparison["metrics"][key] = {
            "label": label,
            "vllm": v,
            "sglang": s,
            "diff_pct": diff_pct,
            "lower_is_better": lower_is_better,
        }

    # 成功率对比
    v_sr = vllm_stats.get("success_rate", 0)
    s_sr = sglang_stats.get("success_rate", 0)
    print(f"\n{'成功率':<25} {v_sr*100:>11.1f}% {s_sr*100:>11.1f}%")

    comparison["vllm_success_rate"] = v_sr
    comparison["sglang_success_rate"] = s_sr

    # 写出对比文件
    out_dir = (
        Path.home() / "llm-benchmark" / "results" / "comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{scenario}_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n对比结果: {out_file}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="vLLM vs SGLang 结果对比分析"
    )
    parser.add_argument(
        "--scenario",
        help="指定场景名称；留空则对比全部场景",
    )
    args = parser.parse_args()

    all_scenarios = [
        "short_short", "short_long",
        "long_short", "long_long",
        "single_turn", "multi_turn",
    ]

    targets = [args.scenario] if args.scenario else all_scenarios

    all_comparisons = {}
    for scenario in targets:
        result = compare_scenario(scenario)
        if result:
            all_comparisons[scenario] = result

    # 写出汇总文件
    if all_comparisons:
        summary_file = (
            Path.home() / "llm-benchmark"
            / "results" / "comparison" / "summary.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                all_comparisons, f, indent=2, ensure_ascii=False
            )
        print(f"\n汇总结果: {summary_file}")

        # 全局摘要
        print(f"\n{'='*70}")
        print("全局摘要")
        print(f"{'='*70}")
        for sc, comp in all_comparisons.items():
            ms = comp.get("metrics", {})
            ttft = ms.get("ttft_p50_ms", {}).get("diff_pct", 0)
            lat = ms.get("latency_p50_ms", {}).get("diff_pct", 0)
            tput = ms.get("throughput_mean_tps", {}).get("diff_pct", 0)
            print(f"  {sc:<20} TTFT: {ttft:+.1f}%  "
                  f"延迟: {lat:+.1f}%  吞吐: {tput:+.1f}%")


if __name__ == "__main__":
    main()
