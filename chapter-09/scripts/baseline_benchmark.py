# scripts/baseline_benchmark.py
# 功能: 标准 vLLM 配置基线测试
#   enforce_eager=True，无 prefix cache，无投机解码，默认 max_pixels
#   运行 3 次取中位数，结果写入 results/baseline_results.json
#
# 运行方式（项目根目录下）:
#   python scripts/baseline_benchmark.py --video data/test_video.mp4

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vllm import LLM, SamplingParams
from video_inference.video_pipeline import extract_frames, build_vllm_messages
from video_inference.latency_profiler import (
    LatencyProfiler, LatencyBreakdown, split_vit_prefill,
)


def run_baseline(
    video_path: str,
    model_path: str = "./models/Qwen2.5-VL-7B-Instruct",
    num_frames: int = 100,
    num_runs: int = 3,
    output_path: str = "./results/baseline_results.json",
) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("初始化基线引擎（enforce_eager=True）...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.1, max_tokens=128)

    # 预热
    print("预热中（10 帧）...")
    for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=10):
        llm.chat(build_vllm_messages(image, include_system_prefix=False), sampling_params=params)

    all_runs = []
    for run_idx in range(num_runs):
        profiler = LatencyProfiler()
        for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=num_frames):
            t_pre = time.perf_counter()
            msgs = build_vllm_messages(image, include_system_prefix=False)
            preprocess_ms = (time.perf_counter() - t_pre) * 1000

            from video_inference.video_pipeline import pil_to_data_uri
            data_uri = pil_to_data_uri(image)
            vit_ms, prefill_ms = split_vit_prefill(llm, data_uri, msgs)

            t0 = time.perf_counter()
            llm.chat(msgs, sampling_params=params)
            total_ms = (time.perf_counter() - t0) * 1000
            decode_ms = max(0.0, total_ms - prefill_ms - vit_ms)

            profiler.record(LatencyBreakdown(
                frame_preprocess_ms=preprocess_ms,
                vit_encode_ms=vit_ms,
                llm_prefill_ms=prefill_ms,
                llm_decode_ms=decode_ms,
            ))

        s = profiler.summary()
        all_runs.append(s)
        print(f"Run {run_idx+1}/{num_runs}: P50={s['p50_ms']:.1f}ms  P95={s['p95_ms']:.1f}ms")

    final = {k: float(np.median([r[k] for r in all_runs]))
             for k in all_runs[0] if isinstance(all_runs[0][k], float)}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n基线结果（{num_runs} 次中位数）:")
    print(f"  P50={final['p50_ms']:.1f}ms  P95={final['p95_ms']:.1f}ms  P99={final['p99_ms']:.1f}ms")
    print(f"  ViT均值={final['mean_vit_ms']:.1f}ms  "
          f"Prefill均值={final['mean_prefill_ms']:.1f}ms  "
          f"Decode均值={final['mean_decode_ms']:.1f}ms")
    print(f"结果写入: {output_path}")
    del llm
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--model",  default="./models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--output", default="./results/baseline_results.json")
    args = parser.parse_args()
    run_baseline(args.video, args.model, args.frames, args.runs, args.output)
