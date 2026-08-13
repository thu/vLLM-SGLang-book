#!/usr/bin/env python3
# scripts/monitor_cache.py
# 前缀缓存命中率实时监测
# 使用方法：python scripts/monitor_cache.py --duration 300
#
# 在批量任务运行期间，后台持续采样 /metrics 端点，
# 记录缓存命中率随时间的变化轨迹，用于分析缓存效果。
#
# 典型用法：在另一个终端运行批量处理任务，同时启动本脚本监测。

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

SGLANG_BASE = "http://127.0.0.1:30000"

# 指标名称以实际 /metrics 输出为准（SGLang v0.4 常见形式）
WATCHED_METRICS = [
    "sglang:cache_hit_rate",
    "sglang:num_running_reqs",
    "sglang:token_usage",
]


def parse_prometheus_line(line: str) -> tuple[str, float] | None:
    """
    解析 Prometheus 格式的单行指标。
    格式：metric_name{labels} value timestamp
    只处理无标签的简单格式和有标签的格式。
    返回 (metric_name_without_labels, float_value) 或 None。
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # 去除标签部分 {k="v",...}
    import re
    clean = re.sub(r"\{[^}]*\}", "", line)
    parts = clean.split()

    if len(parts) < 2:
        return None

    try:
        name  = parts[0]
        value = float(parts[-1])
        return name, value
    except ValueError:
        return None


def sample_metrics() -> dict[str, float]:
    """
    从 /metrics 端点采样一次，返回所有数值指标的字典。
    """
    try:
        text = requests.get(f"{SGLANG_BASE}/metrics", timeout=5).text
    except Exception as e:
        print(f"  采样失败：{e}")
        return {}

    result = {}
    for line in text.splitlines():
        parsed = parse_prometheus_line(line)
        if parsed:
            name, value = parsed
            result[name] = value

    return result


async def monitor(duration_s: int, interval_s: int) -> list[dict]:
    """
    持续采样指标，返回时间序列数据。

    Args:
        duration_s: 总监测时长（秒）
        interval_s: 采样间隔（秒）

    Returns:
        时间序列记录列表，每条包含时间戳和关键指标值
    """
    records   = []
    start_ts  = time.time()
    end_ts    = start_ts + duration_s
    sample_no = 0

    print(f"开始监测（持续 {duration_s}s，每 {interval_s}s 采样一次）...")
    print(f"指标：{WATCHED_METRICS}")
    print("-" * 60)
    print(f"{'时间(s)':>8}  {'命中率':>10}  {'运行请求':>10}  {'Token使用':>10}")
    print("-" * 60)

    while time.time() < end_ts:
        all_metrics = sample_metrics()
        elapsed     = time.time() - start_ts
        sample_no  += 1

        record = {
            "sample_no": sample_no,
            "elapsed_s": round(elapsed, 1),
            "timestamp": round(time.time(), 3),
        }

        # 提取关注的指标
        for metric_name in WATCHED_METRICS:
            short_key = metric_name.split(":")[-1]
            value     = all_metrics.get(metric_name, -1.0)
            record[short_key] = value

        records.append(record)

        # 控制台输出
        hit_rate     = record.get("cache_hit_rate", -1.0)
        running_reqs = record.get("num_running_reqs", -1.0)
        token_usage  = record.get("token_usage", -1.0)

        hit_str     = f"{hit_rate:.1%}"     if hit_rate >= 0     else "N/A"
        req_str     = f"{running_reqs:.0f}" if running_reqs >= 0 else "N/A"
        token_str   = f"{token_usage:.3f}"  if token_usage >= 0  else "N/A"

        print(f"{elapsed:>8.1f}  {hit_str:>10}  {req_str:>10}  {token_str:>10}")

        await asyncio.sleep(interval_s)

    return records


def analyze_records(records: list[dict]) -> None:
    """对采样数据做简单统计分析"""
    if not records:
        print("无数据可分析")
        return

    hit_rates = [r["cache_hit_rate"] for r in records if r.get("cache_hit_rate", -1) >= 0]

    if not hit_rates:
        print("未采集到 cache_hit_rate 数据，请检查指标名称")
        print("提示：执行 python -c \"import requests; print(requests.get('http://127.0.0.1:30000/metrics').text[:500])\"")
        return

    print("\n── 缓存命中率统计 ──")
    print(f"  采样次数：{len(hit_rates)}")
    print(f"  最小值：{min(hit_rates):.1%}")
    print(f"  最大值：{max(hit_rates):.1%}")
    print(f"  平均值：{sum(hit_rates)/len(hit_rates):.1%}")

    # 检测典型的"首次请求命中率为0，后续迅速升高"模式
    if len(hit_rates) >= 3:
        early  = hit_rates[:2]
        late   = hit_rates[-3:]
        early_avg = sum(early) / len(early)
        late_avg  = sum(late)  / len(late)

        if late_avg > early_avg + 0.3:
            print("\n  ✓ 检测到典型前缀缓存生效模式：")
            print(f"    早期命中率（前2次）：{early_avg:.1%}")
            print(f"    后期命中率（最后3次）：{late_avg:.1%}")
        elif late_avg < 0.1:
            print("\n  ⚠️  命中率持续偏低，可能原因：")
            print("    1. 系统提示在不同请求间存在差异（如含时间戳）")
            print("    2. 缓存容量不足，被频繁驱逐")
            print("    3. SGLang 的前缀缓存功能被禁用")


async def main():
    parser = argparse.ArgumentParser(description="前缀缓存命中率监测")
    parser.add_argument("--duration",  type=int, default=120,  help="监测时长（秒，默认 120）")
    parser.add_argument("--interval",  type=int, default=5,    help="采样间隔（秒，默认 5）")
    parser.add_argument("--output",    type=str, default="results/cache_monitor.json",
                        help="结果保存路径")
    args = parser.parse_args()

    print("=" * 60)
    print("前缀缓存命中率监测")
    print("=" * 60)

    records = await monitor(args.duration, args.interval)

    analyze_records(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n时间序列数据已保存：{output_path}")


if __name__ == "__main__":
    asyncio.run(main())
