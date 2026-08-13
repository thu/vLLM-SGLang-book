# 文件名: scripts/common/resource_monitor.py
# 功能: GPU 与 CPU 资源实时监控，以后台线程采样
# 依赖: pynvml（nvidia-ml-py 包）、psutil
# 用法:
#   monitor = ResourceMonitor(gpu_id=0, sample_interval=1.0)
#   monitor.start()
#   ... 执行测试 ...
#   monitor.stop()
#   stats = monitor.get_statistics()

import threading
import time
import json
import dataclasses
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetPowerUsage,
    NVML_TEMPERATURE_GPU,
    NVMLError,
)


@dataclasses.dataclass
class ResourceSnapshot:
    """
    单次资源采样数据。

    timestamp 使用 time.time()（Unix 纪元时间），以便与服务器日志
    和系统日志对齐。GPU 指标通过 NVML API 直接获取，CPU 指标通过
    psutil 获取。
    """

    timestamp: float

    # GPU 指标
    gpu_util_pct: float        # GPU 核心利用率（%）
    gpu_mem_used_gb: float     # 显存已用量（GB）
    gpu_mem_total_gb: float    # 显存总量（GB）
    gpu_temp_celsius: float    # GPU 温度（℃）
    gpu_power_watt: float      # GPU 功耗（W）

    # CPU / 系统指标
    cpu_util_pct: float        # CPU 平均利用率（%）
    cpu_mem_used_gb: float     # 系统内存已用量（GB）
    cpu_mem_total_gb: float    # 系统总内存（GB）

    @property
    def gpu_mem_util_pct(self) -> float:
        """显存利用率（%）"""
        if self.gpu_mem_total_gb <= 0:
            return 0.0
        return self.gpu_mem_used_gb / self.gpu_mem_total_gb * 100.0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["gpu_mem_util_pct"] = self.gpu_mem_util_pct
        return d


class ResourceMonitor:
    """
    后台线程资源监控器。

    调用 start() 启动后台采样线程，调用 stop() 停止并可选地
    将全部快照写入 JSON 文件。采样间隔默认为 1 秒，足以捕捉
    显存占用趋势且不会对测试产生可观测的性能干扰。

    使用 pynvml 而非 nvidia-smi 命令行工具，避免每次采样启动
    子进程的 50-100ms 开销。
    """

    def __init__(
        self,
        gpu_id: int = 0,
        sample_interval: float = 1.0,
        output_file: Optional[str] = None,
    ):
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        self.output_file = output_file

        self.snapshots: List[ResourceSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_initialized = False
        self._handle = None

    def start(self) -> None:
        """启动后台采样线程。"""
        # 初始化 NVML
        nvmlInit()
        self._nvml_initialized = True
        self._handle = nvmlDeviceGetHandleByIndex(self.gpu_id)

        self._stop_event.clear()
        self.snapshots.clear()
        self._thread = threading.Thread(
            target=self._sampling_loop, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止采样线程，等待退出后可选地写出结果文件。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if self.output_file and self.snapshots:
            self._save()

        if self._nvml_initialized:
            try:
                nvmlShutdown()
            except NVMLError:
                pass
            self._nvml_initialized = False

    def _sampling_loop(self) -> None:
        """采样主循环，每隔 sample_interval 秒取一次快照。"""
        while not self._stop_event.is_set():
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
            except NVMLError as exc:
                # NVML 偶发错误不中断监控
                print(f"资源采样异常（已跳过）: {exc}")
            self._stop_event.wait(timeout=self.sample_interval)

    def _take_snapshot(self) -> ResourceSnapshot:
        """采集并返回当前时刻的资源快照。"""
        util = nvmlDeviceGetUtilizationRates(self._handle)
        mem = nvmlDeviceGetMemoryInfo(self._handle)
        temp = nvmlDeviceGetTemperature(
            self._handle, NVML_TEMPERATURE_GPU
        )
        # nvmlDeviceGetPowerUsage 返回值单位为毫瓦
        power = nvmlDeviceGetPowerUsage(self._handle)

        sys_mem = psutil.virtual_memory()

        return ResourceSnapshot(
            timestamp=time.time(),
            gpu_util_pct=float(util.gpu),
            gpu_mem_used_gb=mem.used / 1024**3,
            gpu_mem_total_gb=mem.total / 1024**3,
            gpu_temp_celsius=float(temp),
            gpu_power_watt=power / 1000.0,
            cpu_util_pct=psutil.cpu_percent(interval=None),
            cpu_mem_used_gb=sys_mem.used / 1024**3,
            cpu_mem_total_gb=sys_mem.total / 1024**3,
        )

    def get_statistics(self) -> dict:
        """
        汇总所有快照，返回各指标的均值和峰值。

        关键指标：
        - gpu_mem_peak_gb: 显存峰值，对应表5.1中显存占用验证目标
        - gpu_util_mean_pct: GPU 平均利用率，用于诊断瓶颈位置
        """
        if not self.snapshots:
            return {}

        gpu_utils  = [s.gpu_util_pct     for s in self.snapshots]
        gpu_mems   = [s.gpu_mem_used_gb  for s in self.snapshots]
        gpu_temps  = [s.gpu_temp_celsius for s in self.snapshots]
        gpu_powers = [s.gpu_power_watt   for s in self.snapshots]
        cpu_utils  = [s.cpu_util_pct     for s in self.snapshots]
        cpu_mems   = [s.cpu_mem_used_gb  for s in self.snapshots]

        duration = (
            self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        )

        return {
            "duration_s":            duration,
            "num_samples":           len(self.snapshots),
            # GPU 指标
            "gpu_util_mean_pct":     float(np.mean(gpu_utils)),
            "gpu_util_max_pct":      float(np.max(gpu_utils)),
            "gpu_util_std_pct":      float(np.std(gpu_utils)),
            "gpu_mem_mean_gb":       float(np.mean(gpu_mems)),
            "gpu_mem_peak_gb":       float(np.max(gpu_mems)),
            "gpu_mem_total_gb":      self.snapshots[0].gpu_mem_total_gb,
            "gpu_temp_mean_celsius": float(np.mean(gpu_temps)),
            "gpu_temp_max_celsius":  float(np.max(gpu_temps)),
            "gpu_power_mean_watt":   float(np.mean(gpu_powers)),
            "gpu_power_max_watt":    float(np.max(gpu_powers)),
            # CPU 指标
            "cpu_util_mean_pct":     float(np.mean(cpu_utils)),
            "cpu_util_max_pct":      float(np.max(cpu_utils)),
            "cpu_mem_mean_gb":       float(np.mean(cpu_mems)),
            "cpu_mem_peak_gb":       float(np.max(cpu_mems)),
        }

    def _save(self) -> None:
        """将快照列表和汇总统计写入 JSON 文件。"""
        out_path = Path(self.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshots":  [s.to_dict() for s in self.snapshots],
            "statistics": self.get_statistics(),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"资源监控数据已写入 {out_path}")

    def __enter__(self):
        """支持 with 语句使用。"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
