# scripts/run_baseline.py
# 基线性能测试：无缓存场景（ch11_code_03）
#
# 用途：
#   在启用前缀缓存之前建立性能基准，量化：
#     1. 不同并发数下的吞吐量和延迟
#     2. GPU 计算资源的饱和点（约并发数=12）
#     3. 单次请求的 TTFT 与端到端延迟基准
#
# 使用方式：
#   # SGLang 无缓存基线（单并发）
#   python scripts/run_baseline.py \
#       --framework sglang --concurrency 1
#
#   # vLLM 无缓存基线（并发数=8）
#   python scripts/run_baseline.py \
#       --framework vllm --concurrency 8
#
#   # 扫描多个并发数，绘制性能曲线
#   python scripts/run_baseline.py \
#       --framework sglang --sweep

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from openai import AsyncOpenAI

from config.model_config import DEPLOY_CONFIG, MODEL_CONFIG
from config.prompt_config import build_stable_prefix
from data.dataset_builder import build_synthetic_dataset


def _make_client(framework: str) -> AsyncOpenAI:
    port = (
        DEPLOY_CONFIG["sglang_port"]
        if framework == "sglang"
        else DEPLOY_CONFIG["vllm_port"]
    )
    return AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY",
    )


async def measure_single_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    request_id: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    测量单次请求的延迟指标。

    使用流式接口精确捕获首 token 时间戳（TTFT），
    这是衡量 prefill 阶段耗时的核心指标。
    """
    async with semaphore:
        start = time.perf_counter()
        first_token_time: float | None = None
        token_count = 0
        error_msg = ""

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=384,
                temperature=0.1,
                stream=True,
            )
            async for chunk in stream:
                delta = (
                    chunk.choices[0].delta.content
                    if chunk.choices
                    else None
                )
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += len(delta)
        except Exception as exc:
            error_msg = str(exc)

        end = time.perf_counter()
        ft = first_token_time or end
        return {
            "request_id": request_id,
            "ttft_ms": (ft - start) * 1000,
            "e2e_ms": (end - start) * 1000,
            # decode 阶段吞吐（仅统计生成 token 的速率）
            "decode_tps": (
                token_count / max((end - ft), 0.001)
            ),
            "output_tokens": token_count,
            "error": error_msg,
        }


async def run_baseline_benchmark(
    framework: str,
    prompts: list[str],
    concurrency: int,
) -> dict:
    """
    在指定并发数下运行完整基线测试。

    参数
    ----
    framework : str
        "sglang" 或 "vllm"。
    prompts : list[str]
        测试请求内容列表。
    concurrency : int
        最大并发请求数。

    返回
    ----
    dict
        包含 P50 / P95 / P99 TTFT、吞吐量等统计指标。
    """
    client = _make_client(framework)
    model = MODEL_CONFIG["model_path"].split("/")[-1]
    semaphore = asyncio.Semaphore(concurrency)

    print(
        f"[基线测试] 框架={framework}，"
        f"并发数={concurrency}，"
        f"请求总数={len(prompts)}"
    )
    wall_start = time.perf_counter()

    tasks = [
        measure_single_request(
            client, model, prompt, i, semaphore
        )
        for i, prompt in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - wall_start
    valid = [r for r in results if not r["error"]]

    if not valid:
        return {"error": "所有请求均失败", "concurrency": concurrency}

    ttfts = sorted(r["ttft_ms"] for r in valid)
    e2es = sorted(r["e2e_ms"] for r in valid)
    n = len(ttfts)

    def pct(data: list[float], p: float) -> float:
        idx = min(int(n * p / 100), n - 1)
        return data[idx]

    return {
        "framework": framework,
        "concurrency": concurrency,
        "total_requests": len(prompts),
        "success_count": len(valid),
        "wall_time_s": wall_time,
        "throughput_rps": len(valid) / wall_time,
        "ttft_p50_ms": pct(ttfts, 50),
        "ttft_p95_ms": pct(ttfts, 95),
        "ttft_p99_ms": pct(ttfts, 99),
        "ttft_mean_ms": statistics.mean(ttfts),
        "e2e_p50_ms": pct(e2es, 50),
        "e2e_p95_ms": pct(e2es, 95),
        "decode_tps_mean": statistics.mean(
            r["decode_tps"] for r in valid
        ),
    }


def print_result(result: dict) -> None:
    """格式化打印单次测试结果。"""
    print(
        f"\n{'=' * 56}\n"
        f"框架：{result.get('framework', 'N/A')}  "
        f"并发数：{result.get('concurrency', 'N/A')}\n"
        f"{'─' * 56}\n"
        f"  成功请求   ：{result.get('success_count', 0)}"
        f"/{result.get('total_requests', 0)}\n"
        f"  总耗时     ：{result.get('wall_time_s', 0):.1f} 秒\n"
        f"  吞吐量     ：{result.get('throughput_rps', 0):.2f} req/s\n"
        f"  TTFT P50   ：{result.get('ttft_p50_ms', 0):.0f} ms\n"
        f"  TTFT P95   ：{result.get('ttft_p95_ms', 0):.0f} ms\n"
        f"  TTFT P99   ：{result.get('ttft_p99_ms', 0):.0f} ms\n"
        f"  TTFT 均值  ：{result.get('ttft_mean_ms', 0):.0f} ms\n"
        f"  E2E P50    ：{result.get('e2e_p50_ms', 0):.0f} ms\n"
        f"  Decode TPS ：{result.get('decode_tps_mean', 0):.1f} t/s\n"
        f"{'=' * 56}"
    )


async def main_async(args: argparse.Namespace) -> None:
    # 构建测试数据集（使用前 30 份文书以控制测试时长）
    print("构建测试数据集...")
    docs = build_synthetic_dataset(
        n_procurement=10, n_labor=10, n_software=10
    )

    # 每份文书取一个任务（risk），生成测试提示词列表
    prompts = [
        build_stable_prefix("risk", doc.text) for doc in docs
    ]

    if args.sweep:
        # 扫描模式：依次测试多个并发数
        sweep_results = []
        for c in [1, 2, 4, 8, 12, 16]:
            result = await run_baseline_benchmark(
                args.framework, prompts, c
            )
            print_result(result)
            sweep_results.append(result)

        output_path = (
            f"baseline_sweep_{args.framework}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, ensure_ascii=False, indent=2)
        print(f"\n扫描结果已保存至 {output_path}")
    else:
        result = await run_baseline_benchmark(
            args.framework, prompts, args.concurrency
        )
        print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="无缓存基线性能测试"
    )
    parser.add_argument(
        "--framework",
        choices=["sglang", "vllm"],
        default="sglang",
        help="推理框架（默认：sglang）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发请求数（默认：1）",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="扫描多个并发数（1/2/4/8/12/16）",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
