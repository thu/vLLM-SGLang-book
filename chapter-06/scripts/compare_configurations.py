#!/usr/bin/env python3
# ch06/scripts/compare_configurations.py
# 对比基线配置与最终优化配置的性能差异
#
# 前置条件：
#   1. 已完成 6.4 节的参数扫描，确定最优的 max_num_seqs 和 gpu_memory_utilization
#   2. 将最优参数填入下方 OPTIMIZED_CONFIG
#
# 运行：python ch06/scripts/compare_configurations.py
# 输出：打印对比表 + 保存 results/comparison_report.json

import json
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import httpx

# ---------- 待填入：根据 6.4 节扫描结果确定 ----------
OPTIMIZED_CONFIG = {
    "max_num_seqs": 256,           # 从 tune_max_num_seqs.py 结果填入
    "gpu_memory_utilization": 0.90,  # 从 6.4.3 节测试结果填入
}
# ----------------------------------------------------

BASELINE_CONFIG = {
    "max_num_seqs": 256,
    "gpu_memory_utilization": 0.90,
}

TEST_DURATION_SECONDS = 600   # 正式对比压测使用 10 分钟
LOCUST_USERS = 500
RESULTS_DIR = Path("results")
VLLM_PORT = 8000
GATEWAY_PORT = 8001


@dataclass
class BenchResult:
    label: str
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    failure_rate: float


def _restart_vllm(config: dict, label: str) -> bool:
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(5)

    cmd = [
        "vllm", "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq",
        "--dtype", "float16",
        "--tensor-parallel-size", "2",
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-model-len", "4096",
        "--port", str(VLLM_PORT),
        "--host", "0.0.0.0",
        "--enable-prefix-caching",
        "--disable-log-requests",
        "--served-model-name", "Qwen2.5-7B-Instruct-AWQ",
    ]
    log_path = RESULTS_DIR / f"vllm_{label}.log"
    with open(log_path, "w") as logf:
        subprocess.Popen(cmd, stdout=logf, stderr=logf)

    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _run_and_parse(label: str, config: dict) -> BenchResult | None:
    import csv

    print(f"\n--- {label} ---")
    print(f"  配置: max_num_seqs={config['max_num_seqs']}, "
          f"gpu_memory_utilization={config['gpu_memory_utilization']}")

    if not _restart_vllm(config, label):
        print("  错误：vLLM 启动失败")
        return None

    # 预热
    time.sleep(10)
    try:
        httpx.post(
            f"http://localhost:{GATEWAY_PORT}/v1/chat/completions",
            json={"model": "Qwen2.5-7B-Instruct-AWQ",
                  "messages": [{"role": "user", "content": "你好"}],
                  "max_tokens": 5},
            timeout=60,
        )
    except Exception:
        pass
    time.sleep(5)

    csv_prefix = str(RESULTS_DIR / f"compare_{label}")
    cmd = [
        "locust", "-f", "ch06/load_test/locustfile.py",
        "--host", f"http://localhost:{GATEWAY_PORT}",
        "--headless",
        "-u", str(LOCUST_USERS),
        "-r", str(LOCUST_USERS // 10),
        "-t", f"{TEST_DURATION_SECONDS}s",
        "--csv", csv_prefix,
        "--only-summary",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Locust 运行失败: {result.stderr[:200]}")
        return None

    stats_file = f"{csv_prefix}_stats.csv"
    try:
        with open(stats_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Name") == "Aggregated":
                    total = int(row.get("Request Count", 1))
                    failures = int(row.get("Failure Count", 0))
                    return BenchResult(
                        label=label,
                        qps=float(row.get("Requests/s", 0)),
                        p50_ms=float(row.get("50%", 0)),
                        p95_ms=float(row.get("95%", 0)),
                        p99_ms=float(row.get("99%", 0)),
                        failure_rate=failures / total,
                    )
    except Exception as e:
        print(f"  CSV 解析失败: {e}")
    return None


def print_comparison(baseline: BenchResult, optimized: BenchResult):
    print("\n" + "=" * 65)
    print(f"{'指标':<20} {'基线':>14} {'优化后':>14} {'变化幅度':>12}")
    print("-" * 65)

    def row(label, b_val, o_val, higher_is_better=True, unit=""):
        change = (o_val - b_val) / b_val * 100 if b_val != 0 else 0
        sign = "+" if change >= 0 else ""
        symbol = "✓" if (higher_is_better and change > 0) or (not higher_is_better and change < 0) else "·"
        print(f"{label:<20} {b_val:>12.1f}{unit} {o_val:>12.1f}{unit} {sign}{change:.1f}% {symbol}")

    row("QPS",       baseline.qps,       optimized.qps,       higher_is_better=True)
    row("P50 延迟",   baseline.p50_ms,    optimized.p50_ms,    higher_is_better=False, unit="ms")
    row("P95 延迟",   baseline.p95_ms,    optimized.p95_ms,    higher_is_better=False, unit="ms")
    row("P99 延迟",   baseline.p99_ms,    optimized.p99_ms,    higher_is_better=False, unit="ms")

    print()
    print(f"{'目标达成情况':}")
    print(f"  QPS ≥ 2000:      {'✓' if optimized.qps >= 2000 else '✗'} ({optimized.qps:.0f})")
    print(f"  P95 ≤ 500ms:     {'✓' if optimized.p95_ms <= 500 else '✗'} ({optimized.p95_ms:.0f}ms)")
    print(f"  P99 ≤ 1000ms:    {'✓' if optimized.p99_ms <= 1000 else '✗'} ({optimized.p99_ms:.0f}ms)")
    print(f"  失败率 ≤ 0.1%:   {'✓' if optimized.failure_rate <= 0.001 else '✗'} ({optimized.failure_rate:.3%})")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("配置对比压测")
    print(f"压测时长: {TEST_DURATION_SECONDS}s / 组")
    print("=" * 60)

    baseline = _run_and_parse("baseline", BASELINE_CONFIG)
    optimized = _run_and_parse("optimized", OPTIMIZED_CONFIG)

    if baseline is None or optimized is None:
        print("错误：部分测试失败，请检查日志")
        return

    print_comparison(baseline, optimized)

    output = RESULTS_DIR / "comparison_report.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {"baseline": asdict(baseline), "optimized": asdict(optimized)},
            f, indent=2, ensure_ascii=False
        )
    print(f"\n完整结果已保存: {output}")


if __name__ == "__main__":
    main()
