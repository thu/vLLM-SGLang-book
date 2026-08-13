# 文件名: scripts/common/benchmark_engine.py
# 功能: 双框架共用的统一测试引擎，包含 SSE 流式请求、并发控制、
#       统计计算和数据加载等全部公共逻辑
# 说明: vLLM 和 SGLang 的测试脚本各自仅需提供服务器配置和 main()
#       入口，测试逻辑全部委托给本模块，从架构上保证对比公平性

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

from metrics import RequestMetrics


# ------------------------------------------------------------------
# 全局 tokenizer（避免在每个请求中重复加载）
# ------------------------------------------------------------------

_tokenizer_cache = {}


def get_tokenizer(model_path: str) -> AutoTokenizer:
    """
    获取 tokenizer 实例，使用缓存避免重复加载。

    AutoTokenizer.from_pretrained() 首次调用需要约 200–500ms
    读取磁盘文件并构建分词器数据结构。将其提到请求循环之外，
    确保计时路径上没有不必要的开销。
    """
    if model_path not in _tokenizer_cache:
        _tokenizer_cache[model_path] = (
            AutoTokenizer.from_pretrained(model_path)
        )
    return _tokenizer_cache[model_path]


# ------------------------------------------------------------------
# 默认配置
# ------------------------------------------------------------------

DEFAULT_MODEL_PATH = str(
    Path.home() / "llm-benchmark" / "models" / "llama-3.1-8b-instruct"
)
DEFAULT_MODEL_NAME = "test-model"


# ------------------------------------------------------------------
# 核心请求函数
# ------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    scenario: str,
    prompt: str,
    max_tokens: int,
    base_url: str,
    tokenizer: AutoTokenizer,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.0,
) -> RequestMetrics:
    """
    向 OpenAI 兼容端点发送单个流式请求，精确记录 TTFT。

    函数在发送请求前记录 submit_time；在 SSE 流中接收到第一个
    非空 token 时记录 first_token_time；在流结束后记录
    completion_time。

    参数:
        session: aiohttp 会话，由调用方管理生命周期
        request_id: 请求序号，用于结果追踪
        scenario: 测试场景名称
        prompt: 输入文本
        max_tokens: 最大输出 token 数
        base_url: 服务器地址（如 http://127.0.0.1:8000）
        tokenizer: 预加载的 tokenizer，避免在计时路径上加载
        model_name: 模型名称，需与服务器 --served-model-name 一致
        temperature: 采样温度，0.0 为贪心解码（确保可重复性）
    """
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    input_length = len(tokenizer.encode(prompt))
    first_token_time = None
    output_text = ""
    success = True
    error_msg = ""

    # 记录请求提交时刻
    submit_time = time.perf_counter()

    try:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            # 逐行解析 Server-Sent Events 流
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                # 跳过空行和非数据行（SSE 协议分隔符）
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                token_text = (
                    data.get("choices", [{}])[0]
                    .get("text", "")
                )
                if token_text:
                    # 记录第一个非空 token 到达的时刻
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    output_text += token_text

    except Exception as exc:
        success = False
        error_msg = str(exc)

    completion_time = time.perf_counter()

    # 若流中未产生任何 token（如请求失败），以完成时间填充
    if first_token_time is None:
        first_token_time = completion_time

    output_length = (
        len(tokenizer.encode(output_text)) if output_text else 0
    )

    return RequestMetrics(
        request_id=request_id,
        scenario=scenario,
        input_length=input_length,
        output_length=output_length,
        submit_time=submit_time,
        first_token_time=first_token_time,
        completion_time=completion_time,
        success=success,
        error_msg=error_msg,
    )


# ------------------------------------------------------------------
# 批量测试执行
# ------------------------------------------------------------------

async def run_benchmark(
    prompts: List[str],
    max_tokens_list: List[int],
    scenario: str,
    base_url: str,
    concurrency: int = 1,
    tokenizer: Optional[AutoTokenizer] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout_seconds: int = 300,
) -> List[RequestMetrics]:
    """
    并发执行全部测试请求。

    concurrency 控制同时在途（in-flight）的请求数：
    - concurrency=1: 单请求延迟测试，独占服务器全部计算资源
    - concurrency>1: 吞吐量压测，模拟多用户同时请求

    Semaphore 限制并发窗口，确保客户端侧不成为瓶颈。
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(DEFAULT_MODEL_PATH)

    sem = asyncio.Semaphore(concurrency)
    results: List[RequestMetrics] = []

    async def _bounded(req_id, prompt, max_tokens, sess):
        # 通过信号量限制并发窗口
        async with sem:
            return await send_request(
                sess, req_id, scenario, prompt,
                max_tokens, base_url, tokenizer, model_name,
            )

    # 连接池大小略大于并发数，为 HTTP 保活留余量
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout
    ) as session:
        tasks = [
            _bounded(i, p, m, session)
            for i, (p, m) in enumerate(zip(prompts, max_tokens_list))
        ]
        # tqdm 显示实时进度
        for coro in async_tqdm.as_completed(
            tasks, desc=f"[{scenario}]", total=len(tasks)
        ):
            result = await coro
            results.append(result)

    return results


# ------------------------------------------------------------------
# 统计计算
# ------------------------------------------------------------------

def calculate_statistics(
    results: List[RequestMetrics],
    framework: str = "",
) -> dict:
    """
    从请求指标列表中计算汇总统计数据。

    仅使用成功完成的请求计算延迟与吞吐量指标，同时报告整体
    成功率以反映服务稳定性。

    报告 P50、P95、P99 三个分位数：
    - P50（中位数）: 代表"典型请求"的体验
    - P95: 反映服务稳定性，SLA 常用指标
    - P99: 捕捉极端尾部延迟（100样本下参考价值有限）
    """
    if not results:
        return {}

    successful = [r for r in results if r.success]
    success_rate = len(successful) / len(results)

    if not successful:
        return {
            "framework": framework,
            "success_rate": 0.0,
            "num_total": len(results),
            "num_success": 0,
        }

    ttfts     = [r.ttft_ms          for r in successful]
    latencies = [r.total_latency_ms for r in successful]
    tputs     = [r.throughput_tps   for r in successful]
    dec_tputs = [r.decode_throughput_tps for r in successful]
    in_lens   = [r.input_length     for r in successful]
    out_lens  = [r.output_length    for r in successful]

    def pct(arr, q):
        return float(np.percentile(arr, q))

    # 计算系统整体吞吐量（总输出 token 数 / 总耗时）
    total_output_tokens = sum(r.output_length for r in successful)
    wall_time = max(r.completion_time for r in successful) - min(
        r.submit_time for r in successful
    )
    system_throughput = (
        total_output_tokens / wall_time if wall_time > 0 else 0.0
    )

    return {
        "framework": framework,
        "scenario": results[0].scenario,
        "num_total": len(results),
        "num_success": len(successful),
        "success_rate": success_rate,
        # 输入输出统计
        "avg_input_length":  float(np.mean(in_lens)),
        "avg_output_length": float(np.mean(out_lens)),
        # TTFT 分位数
        "ttft_mean_ms": float(np.mean(ttfts)),
        "ttft_p50_ms":  pct(ttfts, 50),
        "ttft_p95_ms":  pct(ttfts, 95),
        "ttft_p99_ms":  pct(ttfts, 99),
        "ttft_std_ms":  float(np.std(ttfts)),
        # 总延迟分位数
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms":  pct(latencies, 50),
        "latency_p95_ms":  pct(latencies, 95),
        "latency_p99_ms":  pct(latencies, 99),
        "latency_std_ms":  float(np.std(latencies)),
        # 单请求吞吐量
        "throughput_mean_tps":   float(np.mean(tputs)),
        "throughput_median_tps": float(np.median(tputs)),
        "throughput_std_tps":    float(np.std(tputs)),
        # 纯解码吞吐量
        "decode_throughput_mean_tps": float(np.mean(dec_tputs)),
        # 系统整体吞吐量
        "system_throughput_tps": system_throughput,
    }


# ------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------

def load_prompts(
    scenario: str,
    num_samples: int = 100,
    datasets_dir: Optional[Path] = None,
) -> Tuple[List[str], List[int]]:
    """
    根据场景名称加载预处理后的测试数据。

    支持两种数据格式：
    - ShareGPT: List[List[Dict]]，取每条对话第一轮的 user 字段
    - 合成数据: List[Dict]，包含 input 和 target_output_tokens 字段

    返回 (prompts, max_tokens_list) 元组。
    """
    if datasets_dir is None:
        datasets_dir = (
            Path.home() / "llm-benchmark" / "datasets" / "processed"
        )

    dataset_map = {
        "short_short": "synthetic_short_input_short_output.json",
        "short_long":  "synthetic_short_input_long_output.json",
        "long_short":  "synthetic_long_input_short_output.json",
        "long_long":   "synthetic_long_input_long_output.json",
        "single_turn": "sharegpt_single_turn.json",
        "multi_turn":  "sharegpt_multi_turn_5.json",
    }

    if scenario not in dataset_map:
        raise ValueError(
            f"未知场景 '{scenario}'，"
            f"可选: {list(dataset_map.keys())}"
        )

    dataset_path = datasets_dir / dataset_map[scenario]
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    prompts, max_tokens_list = [], []

    for item in data:
        if isinstance(item, list):
            # ShareGPT 格式：取第一轮 user 消息
            if item:
                prompts.append(item[0].get("user", ""))
                max_tokens_list.append(200)
        elif isinstance(item, dict):
            if "input" in item:
                # 合成数据格式
                prompts.append(item["input"])
                max_tokens_list.append(
                    item.get("target_output_tokens", 200)
                )

    # 截取指定数量的样本
    prompts = prompts[:num_samples]
    max_tokens_list = max_tokens_list[:num_samples]

    return prompts, max_tokens_list


# ------------------------------------------------------------------
# 结果保存
# ------------------------------------------------------------------

def save_results(
    results: List[RequestMetrics],
    stats: dict,
    framework: str,
    scenario: str,
    results_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    保存原始请求数据和统计结果到 JSON 文件。

    文件名包含时间戳，避免覆盖历史结果。返回 (raw_file, stats_file)。
    """
    if results_dir is None:
        results_dir = (
            Path.home() / "llm-benchmark" / "results" / framework
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = results_dir / f"{scenario}_raw_{ts}.json"
    stats_file = results_dir / f"{scenario}_stats_{ts}.json"

    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"原始数据: {raw_file}")
    print(f"统计结果: {stats_file}")

    return raw_file, stats_file


# ------------------------------------------------------------------
# 预热
# ------------------------------------------------------------------

async def warmup(
    base_url: str,
    num_requests: int = 5,
    tokenizer: Optional[AutoTokenizer] = None,
) -> None:
    """
    向服务器发送若干短请求以触发 CUDA 图捕获和 JIT 编译。

    预热完成后服务器进入稳态，后续正式测试的延迟将不包含
    一次性初始化开销。
    """
    print(f"预热中（{num_requests} 条短请求）...")
    await run_benchmark(
        prompts=["Hello!"] * num_requests,
        max_tokens_list=[10] * num_requests,
        scenario="warmup",
        base_url=base_url,
        concurrency=1,
        tokenizer=tokenizer,
    )
    print("预热完成")


# ------------------------------------------------------------------
# 结果打印
# ------------------------------------------------------------------

def print_statistics(stats: dict, framework: str = "") -> None:
    """格式化打印测试统计结果。"""
    label = framework.upper() if framework else "测试"
    scenario = stats.get("scenario", "unknown")

    print(f"\n{'='*60}")
    print(f"{label} 测试结果：{scenario}")
    print(f"{'='*60}")
    print(f"成功率             : "
          f"{stats.get('success_rate', 0)*100:.1f}%")
    print(f"平均输入长度       : "
          f"{stats.get('avg_input_length', 0):.0f} tokens")
    print(f"平均输出长度       : "
          f"{stats.get('avg_output_length', 0):.0f} tokens")
    print(f"TTFT P50           : "
          f"{stats.get('ttft_p50_ms', 0):.1f} ms")
    print(f"TTFT P95           : "
          f"{stats.get('ttft_p95_ms', 0):.1f} ms")
    print(f"TTFT P99           : "
          f"{stats.get('ttft_p99_ms', 0):.1f} ms")
    print(f"总延迟 P50         : "
          f"{stats.get('latency_p50_ms', 0):.1f} ms")
    print(f"总延迟 P95         : "
          f"{stats.get('latency_p95_ms', 0):.1f} ms")
    print(f"单请求吞吐量(均值) : "
          f"{stats.get('throughput_mean_tps', 0):.1f} tokens/s")
    print(f"系统整体吞吐量     : "
          f"{stats.get('system_throughput_tps', 0):.1f} tokens/s")
