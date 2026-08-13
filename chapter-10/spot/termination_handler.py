"""
文件: chapter10/spot/termination_handler.py

Spot 实例终止通知处理器。

部署方式:
  作为 Sidecar 容器运行于批处理 Pod（见 k8s/08_spot_job.yaml）。
  Pod 须声明 shareProcessNamespace: true，使 sidecar 能够
  通过 /proc 文件系统扫描并向主容器进程发送信号。

终止通知机制（AWS EC2 Spot）:
  AWS 在实例实际终止前约 2 分钟，通过 Instance Metadata Service（IMDS）
  在以下路径暴露终止通知：
    http://169.254.169.254/latest/meta-data/spot/termination-time
  正常运行时该路径返回 HTTP 404，收到终止通知后返回 HTTP 200，
  响应体为 RFC 3339 格式的终止时间戳（如 2025-01-15T10:30:00Z）。

信号发送机制（父进程PID问题的修复说明）:
  原始草稿使用 os.kill(os.getppid(), signal.SIGTERM) 向父进程发送信号。
  此方案在容器环境中存在以下问题：
    1. 若主进程通过 shell 脚本启动，os.getppid() 返回 shell 的 PID 而非业务进程 PID；
    2. 在 PID 命名空间隔离场景下，PID 1 可能是 init/pause 而非目标进程。

  修复方案：通过 shareProcessNamespace=true 共享进程命名空间后，
  扫描 /proc 文件系统查找目标容器名称对应的主进程 PID（通过 /proc/N/cmdline 匹配），
  精确向目标进程发送 SIGTERM，而非依赖父子进程关系推断。

  替代方案（更简单，适合单一主进程场景）：
  将中断处理逻辑以后台 asyncio Task 的形式内嵌到主进程中，
  完全避免跨进程通信（见 _inline_termination_watcher() 函数）。

GCP Preemptible 实例说明:
  GCP 通过 Metadata Server 提供抢占通知（路径不同，提前约 30 秒），
  修改 IMDS_TERMINATION_URL 为 GCP 端点即可复用本模块逻辑：
    http://metadata.google.internal/computeMetadata/v1/instance/preempted
  注意 GCP 端点需要携带 Metadata-Flavor: Google 请求头。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

import httpx

logger = logging.getLogger("spot-termination-handler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── 配置 ───────────────────────────────────────────────────────────────────

# AWS IMDS Spot 终止通知端点（仅在 AWS EC2 实例上可访问）
IMDS_TERMINATION_URL = (
    "http://169.254.169.254/latest/meta-data/spot/termination-time"
)

# 检查点目录（由环境变量注入，与主容器共享同一 PVC 挂载路径）
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/checkpoints"))

# 目标容器名称（用于 /proc 扫描，匹配 08_spot_job.yaml 中的 container name）
SIGNAL_TARGET_PROCESS = os.getenv("SIGNAL_TARGET", "batch-inference")

# IMDS 轮询间隔（秒）
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))


# ── 检查点保存 ─────────────────────────────────────────────────────────────

def save_termination_checkpoint(termination_time: str) -> Path:
    """
    将终止事件信息写入检查点文件。
    主容器下次启动时读取此文件，跳过已完成的任务批次。

    Args:
        termination_time: AWS IMDS 返回的终止时间戳字符串

    Returns:
        检查点文件路径
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "event": "spot_termination",
        "termination_time": termination_time,
        "detected_at": time.time(),
        "message": "Instance will be terminated. Main process notified via SIGTERM.",
    }
    checkpoint_file = CHECKPOINT_DIR / "termination_checkpoint.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info("Checkpoint saved: %s", checkpoint_file)
    return checkpoint_file


# ── 目标进程查找（修复父进程PID问题）────────────────────────────────────

def find_target_pid(target_name: str) -> int | None:
    """
    通过扫描 /proc 文件系统查找目标进程 PID。

    在 shareProcessNamespace=true 的 Pod 中，所有容器的进程共享同一 PID 命名空间，
    可以通过 /proc/{pid}/cmdline 读取进程启动命令，匹配目标容器名称。

    Args:
        target_name: 目标进程的 cmdline 中应包含的字符串

    Returns:
        找到的 PID（整数），未找到时返回 None
    """
    proc_path = Path("/proc")
    if not proc_path.exists():
        # 非 Linux 环境（本地测试），跳过扫描
        return None

    for pid_dir in proc_path.iterdir():
        if not pid_dir.name.isdigit():
            continue
        cmdline_file = pid_dir / "cmdline"
        try:
            cmdline = cmdline_file.read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="ignore"
            )
            if target_name in cmdline:
                return int(pid_dir.name)
        except (OSError, ValueError):
            # 进程可能在扫描过程中退出，忽略并继续
            continue
    return None


def send_termination_signal(target_name: str) -> bool:
    """
    向目标容器主进程发送 SIGTERM。

    优先通过 /proc 扫描精确定位目标 PID；
    若共享进程命名空间未启用（/proc 不可扫描），回退到发送给当前进程的父进程。

    Returns:
        True 表示信号发送成功，False 表示目标进程未找到
    """
    target_pid = find_target_pid(target_name)
    if target_pid is not None:
        logger.info("Sending SIGTERM to target process: pid=%d, name=%s",
                    target_pid, target_name)
        os.kill(target_pid, signal.SIGTERM)
        return True

    # 回退：发送给父进程（适用于 entrypoint 直接启动主进程的简单场景）
    parent_pid = os.getppid()
    logger.warning(
        "Target process '%s' not found via /proc scan. "
        "Falling back to parent PID=%d. "
        "Ensure shareProcessNamespace=true is set in the Pod spec.",
        target_name, parent_pid,
    )
    if parent_pid > 1:  # PID 1 是 init，不发送
        os.kill(parent_pid, signal.SIGTERM)
        return True
    return False


# ── IMDS 轮询主逻辑 ───────────────────────────────────────────────────────

async def poll_spot_termination():
    """
    轮询 AWS IMDS 终止通知端点，收到通知后执行：
      1. 保存检查点文件（主容器重启时从中断位置续跑）
      2. 向主容器进程发送 SIGTERM（触发优雅退出）
    """
    logger.info(
        "Spot termination handler started. "
        "Polling IMDS every %ds. Target process: %s",
        POLL_INTERVAL_SECONDS, SIGNAL_TARGET_PROCESS,
    )

    # 设置 IMDS 专用超时（2秒），防止网络不通时阻塞轮询循环
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            try:
                resp = await client.get(IMDS_TERMINATION_URL)
                if resp.status_code == 200:
                    # 收到终止通知，立即处理
                    termination_time = resp.text.strip()
                    logger.warning(
                        "SPOT TERMINATION NOTICE received! "
                        "Scheduled termination: %s",
                        termination_time,
                    )
                    save_termination_checkpoint(termination_time)
                    send_termination_signal(SIGNAL_TARGET_PROCESS)
                    # 通知发送后退出轮询循环（自身也即将随 Pod 终止）
                    return
                # HTTP 404 是正常状态（未收到终止通知），继续轮询
            except httpx.TimeoutException:
                # IMDS 超时：可能是非 AWS 环境或网络抖动，继续轮询
                pass
            except httpx.RequestError as exc:
                # 网络错误：IMDS 不可达，继续轮询（不应因此中断监控）
                logger.debug("IMDS request error (non-fatal): %s", exc)

            await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ── 内嵌主进程模式（替代 sidecar 部署的简单方案）────────────────────────

async def inline_termination_watcher(on_termination_callback):
    """
    内嵌式终止监听器，适合将中断处理逻辑集成到主进程中的场景。

    在主进程的 asyncio 事件循环中以后台 Task 运行，
    收到终止通知后调用 on_termination_callback 触发优雅退出，
    无需跨进程信号通信，完全避免 PID 查找问题。

    使用示例（在主进程入口处）:
        async def main():
            asyncio.create_task(
                inline_termination_watcher(on_termination_callback=save_and_exit)
            )
            await run_batch_inference()

    Args:
        on_termination_callback: 收到终止通知后调用的异步协程函数
    """
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            try:
                resp = await client.get(IMDS_TERMINATION_URL)
                if resp.status_code == 200:
                    termination_time = resp.text.strip()
                    logger.warning("Inline watcher: termination at %s", termination_time)
                    save_termination_checkpoint(termination_time)
                    await on_termination_callback()
                    return
            except (httpx.TimeoutException, httpx.RequestError):
                pass
            await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ── 入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(poll_spot_termination())
