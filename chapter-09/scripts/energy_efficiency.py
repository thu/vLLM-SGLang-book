# scripts/energy_efficiency.py
# 功能: GPU 能耗效率分析（Tokens/Watt）
#   对比基线（C0）与全优化（C4）两个配置
#   以 100ms 为间隔后台采样 GPU 功耗（pynvml），与 token 生成速率对比
#   输出 results/energy_results.json
#
# 依赖: pip install pynvml
# 运行方式:
#   python scripts/energy_efficiency.py --video data/test_video.mp4

import sys
import time
import json
import argparse
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vllm import LLM, SamplingParams
from video_inference.video_pipeline import extract_frames, build_vllm_messages

try:
    import pynvml
    PYNVML_OK = True
except ImportError:
    PYNVML_OK = False
    print("警告: pynvml 未安装，功耗将记录为 0。运行: pip install pynvml")


class PowerMonitor:
    """后台线程每 100ms 采样一次 GPU 瞬时功耗（毫瓦→瓦特）。"""

    def __init__(self, device_index: int = 0, interval_ms: int = 100):
        self._interval = interval_ms / 1000.0
        self._device_index = device_index
        self._readings: list[float] = []
        self._running = False
        self._thread: threading.Thread = None
        self._handle = None

    def start(self) -> None:
        if not PYNVML_OK:
            return
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        self._running = True
        self._readings.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                self._readings.append(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0)
            except Exception:
                pass
            time.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if PYNVML_OK and self._handle:
            pynvml.nvmlShutdown()

    @property
    def average_watt(self) -> float:
        if not self._readings:
            return 0.0
        n = len(self._readings)
        trim = max(1, n // 10)
        trimmed = sorted(self._readings)[trim:-trim]
        return float(np.mean(trimmed)) if trimmed else float(np.mean(self._readings))

    @property
    def peak_watt(self) -> float:
        return max(self._readings) if self._readings else 0.0


def measure_one(video_path: str, config_name: str, llm_kwargs: dict,
                num_warmup: int = 100, num_measure: int = 200) -> dict:
    print(f"\n测量: {config_name}")
    llm = LLM(**llm_kwargs)
    params = SamplingParams(temperature=0.1, max_tokens=128)
    monitor = PowerMonitor()

    print(f"  预热（{num_warmup} 帧）...")
    for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=num_warmup):
        llm.chat(build_vllm_messages(image), sampling_params=params)

    print(f"  正式测量（{num_measure} 帧）...")
    total_tokens = 0
    monitor.start()
    t_start = time.perf_counter()

    for image, _ in extract_frames(video_path, target_fps=5.0, max_frames=num_measure):
        outputs = llm.chat(build_vllm_messages(image), sampling_params=params)
        total_tokens += len(outputs[0].outputs[0].token_ids)

    elapsed = time.perf_counter() - t_start
    monitor.stop()

    tps = total_tokens / elapsed
    tpw = tps / monitor.average_watt if monitor.average_watt > 0 else 0.0

    result = {
        "config_name":      config_name,
        "total_tokens":     total_tokens,
        "elapsed_sec":      round(elapsed, 2),
        "tokens_per_sec":   round(tps, 2),
        "avg_power_watt":   round(monitor.average_watt, 1),
        "peak_power_watt":  round(monitor.peak_watt, 1),
        "tokens_per_watt":  round(tpw, 4),
    }
    print(f"  {tps:.1f} tok/s | {monitor.average_watt:.1f}W | {tpw:.4f} tok/W")
    del llm
    return result


def run_energy_comparison(video_path: str, target_path: str, draft_path: str,
                           output_path: str) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    configs = [
        ("C0_baseline_eager", {
            "model": target_path, "dtype": "bfloat16",
            "max_model_len": 2048, "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
        }),
        ("C4_full_optimized", {
            "model": target_path, "speculative_model": draft_path,
            "num_speculative_tokens": 5,
            "dtype": "bfloat16", "max_model_len": 1024,
            "gpu_memory_utilization": 0.90,
            "enforce_eager": False, "enable_prefix_caching": True,
            "max_num_seqs": 8,
            "mm_processor_kwargs": {"min_pixels": 50_176, "max_pixels": 200_704},
        }),
    ]

    results = {}
    for name, kwargs in configs:
        results[name] = measure_one(video_path, name, kwargs)

    if "C0_baseline_eager" in results and "C4_full_optimized" in results:
        base = results["C0_baseline_eager"]["tokens_per_watt"]
        opt  = results["C4_full_optimized"]["tokens_per_watt"]
        results["efficiency_improvement_x"] = round(opt / base, 2) if base > 0 else None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n能耗效率对比:")
    for name, r in results.items():
        if isinstance(r, dict) and "tokens_per_watt" in r:
            print(f"  {name}: {r['tokens_per_watt']:.4f} tok/W "
                  f"({r['tokens_per_sec']:.1f} tok/s @ {r['avg_power_watt']:.1f}W)")
    if "efficiency_improvement_x" in results:
        print(f"  全优化相对基线能耗效率提升: {results['efficiency_improvement_x']}×")
    print(f"结果写入: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--target", default="./models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--draft",  default="./models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="./results/energy_results.json")
    args = parser.parse_args()
    run_energy_comparison(args.video, args.target, args.draft, args.output)
