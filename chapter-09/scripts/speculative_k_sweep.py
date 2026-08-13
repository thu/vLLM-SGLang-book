# scripts/speculative_k_sweep.py
# 功能: 扫描 num_speculative_tokens（k 值）对端到端延迟的影响
#   确定 Qwen2.5-VL-7B + Qwen2.5-1.5B 组合的最优 k 值
#   每个 k 值 50 帧 × 3 次取中位数
#   输出 results/speculative_k_sweep.json
#
# 理论背景（Leviathan et al., 2023）：
#   speedup(k) = (k·α(k) + 1) / (1 + T_draft(k) / T_target)
#   α(k) 随 k 增大单调递减，T_draft(k) 与 k 近似线性，
#   因此存在使 speedup 最大化的最优 k*。
#
# 运行方式:
#   python scripts/speculative_k_sweep.py --video data/test_video.mp4

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vllm import LLM, SamplingParams
from video_inference.video_pipeline import extract_frames, build_vllm_messages
from video_inference.latency_profiler import LatencyProfiler, LatencyBreakdown

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 10]


def run_k_sweep(video_path: str, target_path: str, draft_path: str,
                num_frames: int, num_runs: int, output_path: str) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    params = SamplingParams(temperature=0.1, max_tokens=128)
    all_results = {}

    for k in K_VALUES:
        print(f"\n测试 k={k}...")
        llm = LLM(
            model=target_path, speculative_model=draft_path,
            num_speculative_tokens=k,
            dtype="bfloat16", max_model_len=1024,
            gpu_memory_utilization=0.90, enforce_eager=False,
            enable_prefix_caching=True,
            mm_processor_kwargs={"min_pixels": 50_176, "max_pixels": 200_704},
        )

        for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=10):
            llm.chat(build_vllm_messages(image), sampling_params=params)

        runs = []
        for i in range(num_runs):
            profiler = LatencyProfiler()
            for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=num_frames):
                t0 = time.perf_counter()
                llm.chat(build_vllm_messages(image), sampling_params=params)
                profiler.record(LatencyBreakdown(
                    llm_decode_ms=(time.perf_counter() - t0) * 1000
                ))
            s = profiler.summary()
            runs.append(s)
            print(f"  Run {i+1}/{num_runs}: P50={s['p50_ms']:.1f}ms")

        final = {k2: float(np.median([r[k2] for r in runs]))
                 for k2 in runs[0] if isinstance(runs[0][k2], float)}
        final["k"] = k
        all_results[f"k_{k}"] = final
        del llm

    best = min(all_results.values(), key=lambda r: r["p50_ms"])
    all_results["best_k"] = best["k"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    base_p50 = all_results["k_1"]["p50_ms"]
    print(f"\n{'k值':<8} {'P50(ms)':<12} {'P95(ms)':<12} {'相对k=1加速比'}")
    print("-" * 48)
    for label, r in all_results.items():
        if label == "best_k":
            continue
        speedup = base_p50 / r["p50_ms"]
        marker = " ← 最优" if r["k"] == best["k"] else ""
        print(f"{r['k']:<8} {r['p50_ms']:<12.1f} {r['p95_ms']:<12.1f} {speedup:.2f}×{marker}")

    print(f"\n最优 k 值: k={best['k']} (P50={best['p50_ms']:.1f}ms)")
    print(f"结果写入: {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--target", default="./models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--draft",  default="./models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--output", default="./results/speculative_k_sweep.json")
    args = parser.parse_args()
    run_k_sweep(args.video, args.target, args.draft, args.frames, args.runs, args.output)
