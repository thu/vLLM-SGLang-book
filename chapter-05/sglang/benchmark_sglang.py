# 文件名: scripts/sglang/benchmark_sglang.py
# 功能: SGLang 基准测试入口脚本
# 说明: 测试逻辑全部委托给 common/benchmark_engine.py，
#       本脚本仅提供框架特定的配置和 CLI 入口，
#       与 benchmark_vllm.py 完全对称
# 用法: python benchmark_sglang.py --scenario long_short
#        python benchmark_sglang.py --scenario short_short --concurrency 10

import asyncio
import argparse
import sys
from pathlib import Path

# 将 common 目录加入模块搜索路径
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "common")
)

from benchmark_engine import (
    run_benchmark,
    calculate_statistics,
    load_prompts,
    save_results,
    print_statistics,
    warmup,
    get_tokenizer,
    DEFAULT_MODEL_PATH,
)
from resource_monitor import ResourceMonitor


# ---- 框架特定配置 ----
FRAMEWORK = "sglang"
DEFAULT_PORT = 30000


async def main_async(args):
    """异步主流程：加载数据 → 预热 → 测试 → 保存结果。"""

    base_url = f"http://127.0.0.1:{args.port}"
    tokenizer = get_tokenizer(DEFAULT_MODEL_PATH)

    # 加载测试数据
    prompts, max_tokens_list = load_prompts(
        args.scenario, args.num_samples
    )

    print(f"框架     : SGLang")
    print(f"目标服务 : {base_url}")
    print(f"测试场景 : {args.scenario}")
    print(f"样本数   : {len(prompts)}")
    print(f"并发数   : {args.concurrency}")

    # 启动资源监控（可选）
    monitor = None
    if args.monitor:
        monitor_file = (
            Path.home() / "llm-benchmark" / "results" / "sglang"
            / f"{args.scenario}_resources.json"
        )
        monitor = ResourceMonitor(
            gpu_id=0,
            sample_interval=1.0,
            output_file=str(monitor_file),
        )
        monitor.start()
        print(f"资源监控 : 已启动 → {monitor_file}")

    try:
        # 预热
        await warmup(base_url, num_requests=5, tokenizer=tokenizer)

        # 正式测试
        results = await run_benchmark(
            prompts=prompts,
            max_tokens_list=max_tokens_list,
            scenario=args.scenario,
            base_url=base_url,
            concurrency=args.concurrency,
            tokenizer=tokenizer,
        )

        # 统计与输出
        stats = calculate_statistics(results, framework=FRAMEWORK)
        print_statistics(stats, framework=FRAMEWORK)

        # 附加资源监控统计
        if monitor:
            res_stats = monitor.get_statistics()
            stats["resource_stats"] = res_stats
            print(f"\nGPU 平均利用率 : "
                  f"{res_stats.get('gpu_util_mean_pct', 0):.1f}%")
            print(f"显存峰值占用   : "
                  f"{res_stats.get('gpu_mem_peak_gb', 0):.1f} GB")

        # 保存结果
        save_results(results, stats, FRAMEWORK, args.scenario)

    finally:
        if monitor:
            monitor.stop()


def main():
    parser = argparse.ArgumentParser(
        description="SGLang 基准测试客户端"
    )
    parser.add_argument(
        "--scenario", required=True,
        choices=[
            "short_short", "short_long",
            "long_short", "long_long",
            "single_turn", "multi_turn",
        ],
        help="测试场景名称",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"SGLang 服务器端口（默认 {DEFAULT_PORT}）",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="测试样本数量（默认 100）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="并发请求数（1 = 单请求延迟测试）",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="启用 GPU/CPU 资源监控",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
