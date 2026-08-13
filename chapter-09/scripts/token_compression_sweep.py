# scripts/token_compression_sweep.py
# 功能: 扫描五个 max_pixels 配置下的延迟权衡
#   每个配置 50 帧 × 3 次取中位数
#   输出 results/compression_pareto.json
#
# 运行方式:
#   python scripts/token_compression_sweep.py --video data/test_video.mp4

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

# (max_pixels, 标签, token数近似)
# token 数 ≈ max_pixels / 196（patch_size=14，每 patch 196 像素），受 max_grid 约束后取整
CONFIGS = [
    (1_003_520, "default_1024t",  "~1024 token"),
    (602_112,   "light_614t",     "~614 token"),
    (200_704,   "medium_256t",    "~256 token"),   # 推荐
    (100_352,   "heavy_128t",     "~128 token"),
    (50_176,    "extreme_64t",    "~64 token"),
]


def run_sweep(video_path: str, model_path: str, num_frames: int,
              num_runs: int, output_path: str) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    params = SamplingParams(temperature=0.1, max_tokens=80)
    all_results = {}

    for max_pixels, label, token_hint in CONFIGS:
        print(f"\n测试: {label} ({token_hint}), max_pixels={max_pixels}")
        llm = LLM(
            model=model_path, dtype="bfloat16", max_model_len=2048,
            gpu_memory_utilization=0.85, enforce_eager=False,
            enable_prefix_caching=True,
            mm_processor_kwargs={"min_pixels": 50_176, "max_pixels": max_pixels},
        )

        for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=5):
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

        final = {k: float(np.median([r[k] for r in runs]))
                 for k in runs[0] if isinstance(runs[0][k], float)}
        final.update({"token_hint": token_hint, "max_pixels": max_pixels})
        all_results[label] = final
        del llm

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'配置':<20} {'P50(ms)':<12} {'P95(ms)':<12} {'Token数'}")
    print("-" * 55)
    for label, r in all_results.items():
        print(f"{label:<20} {r['p50_ms']:<12.1f} {r['p95_ms']:<12.1f} {r['token_hint']}")
    print(f"\n结果写入: {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--model",  default="./models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--output", default="./results/compression_pareto.json")
    args = parser.parse_args()
    run_sweep(args.video, args.model, args.frames, args.runs, args.output)
