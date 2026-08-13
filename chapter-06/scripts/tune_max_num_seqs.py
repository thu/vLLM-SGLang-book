#!/usr/bin/env python3
# ch06/scripts/tune_max_num_seqs.py
# 自动化扫描 max_num_seqs 参数的最优值
#
# 原理：max_num_seqs 控制 vLLM 单次调度周期内的最大并行序列数。
#       设置过低：GPU 批处理规模不足，算力空转；
#       设置过高：请求排队时间增加，P99 延迟上升。
#       本脚本对候选值逐一测试，找到 QPS 与 P99 之间的帕累托最优点。
#
# 前置条件：vLLM 可以通过 start_vllm.sh 启动，Locust 已安装
# 运行：python ch06/scripts/tune_max_num_seqs.py
# 结果：打印对比表，同时保存到 results/tune_max_num_seqs.json

import json
import subprocess
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import httpx

# 候选参数值
CANDIDATE_VALUES = [128, 192, 256, 320, 384]

# 每组测试时长（秒），建议不低于 180s 以获得稳定的分位数
TEST_DURATION_SECONDS = 180

# 压测并发用户数
LOCUST_USERS = 500

VLLM_PORT = 8000
GATEWAY_PORT = 8001
RESULTS_DIR = Path("results")


@dataclass
class TuneResult:
    max_num_seqs: int
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    failure_rate: float
    note: str = ""


def wait_for_vllm(timeout: int = 120) -> bool:
    """等待 vLLM 健康检查通过"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def restart_vllm(max_num_seqs: int) -> bool:
    """停止现有 vLLM 进程，以新参数重启"""
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(5)

    cmd = [
        "vllm", "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq",
        "--dtype", "float16",
        "--tensor-parallel-size", "2",
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "4096",
        "--port", str(VLLM_PORT),
        "--host", "0.0.0.0",
        "--enable-prefix-caching",
        "--disable-log-requests",
        "--served-model-name", "Qwen2.5-7B-Instruct-AWQ",
    ]

    log_path = RESULTS_DIR / f"vllm_seqs{max_num_seqs}.log"
    with open(log_path, "w") as logf:
        subprocess.Popen(cmd, stdout=logf, stderr=logf)

    print(f"  等待 vLLM 启动（max_num_seqs={max_num_seqs}）...")
    if not wait_for_vllm(timeout=120):
        print("  错误：vLLM 未能在 120s 内启动")
        return False
    print("  vLLM 就绪")
    return True


def run_locust_headless(duration: int, users: int, csv_prefix: str) -> bool:
    """无头模式运行 Locust，结果写入 CSV"""
    cmd = [
        "locust",
        "-f", "ch06/load_test/locustfile.py",
        "--host", f"http://localhost:{GATEWAY_PORT}",
        "--headless",
        "-u", str(users),
        "-r", str(users // 10),      # 10 秒内升至目标并发
        "-t", f"{duration}s",
        "--csv", csv_prefix,
        "--csv-full-history",
        "--only-summary",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def parse_locust_csv(csv_prefix: str) -> dict | None:
    """从 Locust CSV 统计文件提取关键指标"""
    import csv
    stats_file = f"{csv_prefix}_stats.csv"
    try:
        with open(stats_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Name") == "Aggregated":
                    total = int(row.get("Request Count", 0))
                    failures = int(row.get("Failure Count", 0))
                    return {
                        "qps": float(row.get("Requests/s", 0)),
                        "p50_ms": float(row.get("50%", 0)),
                        "p95_ms": float(row.get("95%", 0)),
                        "p99_ms": float(row.get("99%", 0)),
                        "failure_rate": failures / total if total > 0 else 0.0,
                    }
    except Exception as e:
        print(f"  解析 CSV 失败: {e}")
    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[TuneResult] = []

    print("=" * 60)
    print("max_num_seqs 参数扫描")
    print(f"候选值: {CANDIDATE_VALUES}")
    print(f"每组测试时长: {TEST_DURATION_SECONDS}s")
    print("=" * 60)

    for value in CANDIDATE_VALUES:
        print(f"\n--- 测试 max_num_seqs={value} ---")

        if not restart_vllm(value):
            results.append(TuneResult(value, 0, 0, 0, 0, 1.0, "vLLM 启动失败"))
            continue

        # 预热：发送一次推理请求确认 CUDA kernel 已编译
        try:
            httpx.post(
                f"http://localhost:{GATEWAY_PORT}/v1/chat/completions",
                json={
                    "model": "Qwen2.5-7B-Instruct-AWQ",
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 5,
                },
                timeout=60,
            )
        except Exception:
            pass
        time.sleep(5)

        csv_prefix = str(RESULTS_DIR / f"tune_seqs{value}")
        print(f"  运行 Locust 压测，时长 {TEST_DURATION_SECONDS}s...")
        success = run_locust_headless(TEST_DURATION_SECONDS, LOCUST_USERS, csv_prefix)
        if not success:
            results.append(TuneResult(value, 0, 0, 0, 0, 1.0, "Locust 运行失败"))
            continue

        metrics = parse_locust_csv(csv_prefix)
        if metrics is None:
            results.append(TuneResult(value, 0, 0, 0, 0, 1.0, "CSV 解析失败"))
            continue

        r = TuneResult(
            max_num_seqs=value,
            qps=metrics["qps"],
            p50_ms=metrics["p50_ms"],
            p95_ms=metrics["p95_ms"],
            p99_ms=metrics["p99_ms"],
            failure_rate=metrics["failure_rate"],
        )
        results.append(r)
        print(
            f"  QPS={r.qps:.1f}  P50={r.p50_ms:.0f}ms  "
            f"P95={r.p95_ms:.0f}ms  P99={r.p99_ms:.0f}ms  "
            f"失败率={r.failure_rate:.2%}"
        )

    # 汇总输出
    print("\n" + "=" * 70)
    print(f"{'max_num_seqs':>14} {'QPS':>8} {'P50(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} {'失败率':>7}")
    print("-" * 70)
    for r in results:
        marker = " ←" if r.note == "" and r.p99_ms <= 1000 and r.qps == max((x.qps for x in results if x.note == ""), default=0) else ""
        print(
            f"{r.max_num_seqs:>14} {r.qps:>8.1f} {r.p50_ms:>9.0f} "
            f"{r.p95_ms:>9.0f} {r.p99_ms:>9.0f} {r.failure_rate:>7.2%}{marker}"
        )

    # 保存结果
    output = RESULTS_DIR / "tune_max_num_seqs.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output}")


if __name__ == "__main__":
    main()
