# scripts/ablation_study.py
# 功能: 消融实验，逐层叠加五个优化配置（C0→C4）
#   每层配置 100 帧 × 3 次取中位数
#   输出 results/ablation_table.json
#
# 运行方式:
#   python scripts/ablation_study.py --video data/test_video.mp4

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

TARGET_PATH = "./models/Qwen2.5-VL-7B-Instruct"
DRAFT_PATH  = "./models/Qwen2.5-1.5B-Instruct"

ABLATION_CONFIGS = [
    {
        "name": "C0_baseline_eager",
        "desc": "基线：enforce_eager=True，无任何额外优化",
        "kwargs": {
            "model": TARGET_PATH, "dtype": "bfloat16",
            "max_model_len": 2048, "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
        },
    },
    {
        "name": "C1_flash_attn",
        "desc": "叠加 FlashAttention-2（关闭 enforce_eager）",
        "kwargs": {
            "model": TARGET_PATH, "dtype": "bfloat16",
            "max_model_len": 2048, "gpu_memory_utilization": 0.85,
            "enforce_eager": False,
        },
    },
    {
        "name": "C2_token_compression",
        "desc": "叠加 Visual Token 压缩（max_pixels=200_704，约 256 token）",
        "kwargs": {
            "model": TARGET_PATH, "dtype": "bfloat16",
            "max_model_len": 1024, "gpu_memory_utilization": 0.85,
            "enforce_eager": False,
            "mm_processor_kwargs": {"min_pixels": 50_176, "max_pixels": 200_704},
        },
    },
    {
        "name": "C3_prefix_cache",
        "desc": "叠加跨帧 KV Cache 前缀复用",
        "kwargs": {
            "model": TARGET_PATH, "dtype": "bfloat16",
            "max_model_len": 1024, "gpu_memory_utilization": 0.85,
            "enforce_eager": False, "enable_prefix_caching": True,
            "mm_processor_kwargs": {"min_pixels": 50_176, "max_pixels": 200_704},
        },
    },
    {
        "name": "C4_speculative",
        "desc": "叠加投机解码（Qwen2.5-1.5B，k=5）",
        "kwargs": {
            "model": TARGET_PATH, "speculative_model": DRAFT_PATH,
            "num_speculative_tokens": 5,
            "dtype": "bfloat16", "max_model_len": 1024,
            "gpu_memory_utilization": 0.90,
            "enforce_eager": False, "enable_prefix_caching": True,
            "mm_processor_kwargs": {"min_pixels": 50_176, "max_pixels": 200_704},
        },
    },
]


def run_one(cfg: dict, video_path: str, num_frames: int) -> dict:
    llm = LLM(**cfg["kwargs"])
    params = SamplingParams(temperature=0.1, max_tokens=128)
    profiler = LatencyProfiler()

    for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=10):
        llm.chat(build_vllm_messages(image), sampling_params=params)  # 预热

    for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=num_frames):
        msgs = build_vllm_messages(image)
        t0 = time.perf_counter()
        llm.chat(msgs, sampling_params=params)
        profiler.record(LatencyBreakdown(llm_decode_ms=(time.perf_counter() - t0) * 1000))

    del llm
    return profiler.summary()


def run_ablation(video_path: str, num_frames: int, num_runs: int, output_path: str) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for cfg in ABLATION_CONFIGS:
        print(f"\n{'='*60}\n配置: {cfg['name']}\n描述: {cfg['desc']}")
        runs = []
        for i in range(num_runs):
            s = run_one(cfg, video_path, num_frames)
            runs.append(s)
            print(f"  Run {i+1}/{num_runs}: P50={s['p50_ms']:.1f}ms  P95={s['p95_ms']:.1f}ms")

        final = {k: float(np.median([r[k] for r in runs]))
                 for k in runs[0] if isinstance(runs[0][k], float)}
        final["desc"] = cfg["desc"]
        all_results[cfg["name"]] = final

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n消融实验汇总:")
    print(f"{'配置':<30} {'P50(ms)':<12} {'P95(ms)':<12} {'P50降幅'}")
    print("-" * 65)
    prev_p50 = None
    for name, r in all_results.items():
        delta = f"-{(prev_p50 - r['p50_ms']) / prev_p50 * 100:.1f}%" if prev_p50 else "（基线）"
        print(f"{name:<30} {r['p50_ms']:<12.1f} {r['p95_ms']:<12.1f} {delta}")
        prev_p50 = r["p50_ms"]

    print(f"\n结果写入: {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--output", default="./results/ablation_table.json")
    args = parser.parse_args()
    run_ablation(args.video, args.frames, args.runs, args.output)
