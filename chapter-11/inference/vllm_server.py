# inference/vllm_server.py
# vLLM v0.8.x 服务启动与管理（ch11_code_05）
#
# vLLM v0.8.x 的推荐部署方式同样是通过命令行启动独立服务进程，
# 并通过 OpenAI 兼容的 HTTP 接口交互。
#
# 关于 block_size 参数：
#   block_size 是服务启动时的命令行参数（--block-size），
#   不是 Python EngineArgs 的构造参数。
#   在 vLLM v0.8.x 中直接实例化 AsyncLLMEngine 已不是
#   推荐做法，本模块统一采用命令行启动方式。
#
# 关于 max_num_batched_tokens：
#   对于 20K tokens 的长文本，建议设为 131072 以避免
#   批处理 token 限制导致的额外调度延迟。

from __future__ import annotations

import subprocess
import time
from typing import IO

import httpx

from config.model_config import DEPLOY_CONFIG, MODEL_CONFIG


def build_vllm_command(
    enable_prefix_caching: bool = True,
    block_size: int | None = None,
    max_num_batched_tokens: int = 131072,
    extra_args: list[str] | None = None,
) -> list[str]:
    """
    构建 vLLM 服务启动命令。

    参数
    ----
    enable_prefix_caching : bool
        是否启用块级前缀缓存，默认 True。
        vLLM v0.8.x 中 enable_prefix_caching 默认为 False，
        须显式传入 --enable-prefix-caching 开启。
    block_size : int, optional
        物理块大小（tokens/块），默认使用 DEPLOY_CONFIG 中的值（32）。
        须为 2 的幂次，推荐选项：16（默认）、32、64。
        32 是长文本场景的推荐值：兼顾对齐损失和哈希表规模。
    max_num_batched_tokens : int
        单次调度的最大 token 数，默认 131072。
        对于 20K tokens 长文本，此值应大于单次 prefill 的 token 数。
    extra_args : list[str], optional
        额外的命令行参数。

    返回
    ----
    list[str]
        可直接传入 subprocess.Popen 的命令列表。
    """
    bs = block_size or DEPLOY_CONFIG["vllm_block_size"]

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_CONFIG["model_path"],
        "--quantization", MODEL_CONFIG["quantization"],
        "--tensor-parallel-size",
        str(DEPLOY_CONFIG["tensor_parallel_size"]),
        "--max-model-len",
        str(MODEL_CONFIG["max_context_len"]),
        "--gpu-memory-utilization",
        str(DEPLOY_CONFIG["vllm_gpu_memory_utilization"]),
        # block_size 是命令行参数，不是 Python 构造参数
        "--block-size", str(bs),
        "--max-num-seqs",
        str(DEPLOY_CONFIG["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        # 禁用 CPU swap，避免延迟不确定性
        "--swap-space", "0",
        "--host", DEPLOY_CONFIG["vllm_host"],
        "--port", str(DEPLOY_CONFIG["vllm_port"]),
    ]

    if enable_prefix_caching:
        # vLLM v0.8.x 中需显式开启前缀缓存
        cmd.append("--enable-prefix-caching")

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def start_vllm_server(
    enable_prefix_caching: bool = True,
    startup_timeout_seconds: int = 180,
    log_file: IO | None = None,
) -> subprocess.Popen:
    """
    启动 vLLM 服务进程并等待其就绪。

    参数
    ----
    enable_prefix_caching : bool
        是否启用块级前缀缓存，默认 True。
    startup_timeout_seconds : int
        等待服务就绪的最大时间（秒），默认 180。
    log_file : IO, optional
        服务日志输出目标。

    返回
    ----
    subprocess.Popen
        服务进程对象。

    异常
    ----
    RuntimeError
        若超时则抛出。
    """
    cmd = build_vllm_command(
        enable_prefix_caching=enable_prefix_caching
    )
    stdout_target = log_file or subprocess.PIPE

    process = subprocess.Popen(
        cmd,
        stdout=stdout_target,
        stderr=subprocess.STDOUT,
    )

    health_url = (
        f"http://localhost:{DEPLOY_CONFIG['vllm_port']}/health"
    )

    for elapsed in range(startup_timeout_seconds):
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM 服务进程意外退出，退出码：{process.returncode}。"
                "请检查模型路径和显存是否充足。"
            )
        try:
            resp = httpx.get(health_url, timeout=3.0)
            if resp.status_code == 200:
                print(
                    f"vLLM 服务已就绪（等待 {elapsed} 秒），"
                    f"监听 :{DEPLOY_CONFIG['vllm_port']}，"
                    f"前缀缓存={'启用' if enable_prefix_caching else '禁用'}"
                )
                return process
        except httpx.ConnectError:
            pass
        time.sleep(1)

    process.terminate()
    raise RuntimeError(
        f"vLLM 服务在 {startup_timeout_seconds} 秒内未就绪。"
    )


def stop_vllm_server(process: subprocess.Popen) -> None:
    """优雅地终止 vLLM 服务进程。"""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    print("vLLM 服务已停止。")


def get_vllm_base_url() -> str:
    """返回 vLLM 服务的 OpenAI 兼容接口基础 URL。"""
    return (
        f"http://localhost:{DEPLOY_CONFIG['vllm_port']}/v1"
    )
