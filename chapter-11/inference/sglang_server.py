# inference/sglang_server.py
# SGLang v0.5.x 服务启动与管理（ch11_code_04）
#
# SGLang v0.5.x 的标准部署方式是通过命令行启动独立服务进程，
# 并通过 OpenAI 兼容的 HTTP 接口（/v1/chat/completions）
# 与之交互。相比直接实例化 Python 对象，命令行方式在进程
# 隔离性和资源管理上更为可靠，是生产环境的推荐做法。
#
# RadixAttention 在 SGLang v0.5.x 中默认启用，
# 无需额外参数。禁用方式：添加 --disable-radix-cache。

from __future__ import annotations

import subprocess
import time
from typing import IO

import httpx

from config.model_config import DEPLOY_CONFIG, MODEL_CONFIG


def build_sglang_command(
    enable_radix_cache: bool = True,
    log_level: str = "info",
    extra_args: list[str] | None = None,
) -> list[str]:
    """
    构建 SGLang 服务启动命令。

    参数
    ----
    enable_radix_cache : bool
        是否启用 RadixAttention 前缀缓存，默认 True。
        设为 False 时追加 --disable-radix-cache（用于基线测试）。
    log_level : str
        日志级别，可选 debug / info / warning / error。
    extra_args : list[str], optional
        额外的命令行参数，用于实验性配置。

    返回
    ----
    list[str]
        可直接传入 subprocess.Popen 的命令列表。
    """
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL_CONFIG["model_path"],
        "--quantization", MODEL_CONFIG["quantization"],
        "--tensor-parallel-size",
        str(DEPLOY_CONFIG["tensor_parallel_size"]),
        "--context-length",
        str(MODEL_CONFIG["max_context_len"]),
        # 静态 KV Cache 预分配比例（0.88 适合长文本场景）
        "--mem-fraction-static",
        str(DEPLOY_CONFIG["sglang_mem_fraction_static"]),
        "--max-running-reqs",
        str(DEPLOY_CONFIG["max_num_seqs"]),
        "--host", DEPLOY_CONFIG["sglang_host"],
        "--port", str(DEPLOY_CONFIG["sglang_port"]),
        "--log-level", log_level,
    ]

    if not enable_radix_cache:
        # 显式禁用 RadixAttention，仅用于基线对比测试
        cmd.append("--disable-radix-cache")

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def start_sglang_server(
    enable_radix_cache: bool = True,
    startup_timeout_seconds: int = 180,
    log_file: IO | None = None,
) -> subprocess.Popen:
    """
    启动 SGLang 服务进程并等待其就绪。

    参数
    ----
    enable_radix_cache : bool
        是否启用 RadixAttention，默认 True。
    startup_timeout_seconds : int
        等待服务就绪的最大时间（秒），默认 180。
        Qwen2-72B AWQ 在首次加载时通常需要 60—120 秒。
    log_file : IO, optional
        服务日志的输出目标，默认为 PIPE（不写文件）。

    返回
    ----
    subprocess.Popen
        服务进程对象，调用方负责在适当时机调用 process.terminate()。

    异常
    ----
    RuntimeError
        若在超时时间内服务未就绪则抛出。
    """
    cmd = build_sglang_command(
        enable_radix_cache=enable_radix_cache
    )
    stdout_target = log_file or subprocess.PIPE

    process = subprocess.Popen(
        cmd,
        stdout=stdout_target,
        stderr=subprocess.STDOUT,
    )

    health_url = (
        f"http://localhost:{DEPLOY_CONFIG['sglang_port']}/health"
    )

    for elapsed in range(startup_timeout_seconds):
        # 检查进程是否意外退出
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang 服务进程意外退出，退出码：{process.returncode}。"
                "请检查模型路径和显存是否充足。"
            )
        try:
            resp = httpx.get(health_url, timeout=3.0)
            if resp.status_code == 200:
                print(
                    f"SGLang 服务已就绪（等待 {elapsed} 秒），"
                    f"监听 :{DEPLOY_CONFIG['sglang_port']}，"
                    f"RadixAttention={'启用' if enable_radix_cache else '禁用'}"
                )
                return process
        except httpx.ConnectError:
            pass
        time.sleep(1)

    process.terminate()
    raise RuntimeError(
        f"SGLang 服务在 {startup_timeout_seconds} 秒内未就绪，"
        "已终止进程。"
    )


def stop_sglang_server(process: subprocess.Popen) -> None:
    """
    优雅地终止 SGLang 服务进程。

    先发送 SIGTERM 等待 10 秒，超时后发送 SIGKILL。
    """
    if process.poll() is not None:
        # 进程已退出，无需操作
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    print("SGLang 服务已停止。")


def get_sglang_base_url() -> str:
    """返回 SGLang 服务的 OpenAI 兼容接口基础 URL。"""
    return (
        f"http://localhost:{DEPLOY_CONFIG['sglang_port']}/v1"
    )
