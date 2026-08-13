# src/video_inference/adaptive_serving.py
# 功能: 基于 AsyncLLMEngine 的自适应并发视频推理服务
#
# 设计要点：
#   AdaptiveConcurrencyController
#     - 监控 P95 延迟滑动窗口，动态调整并发许可数
#     - 扩容：对同一 Semaphore 对象调用 release()，不替换对象
#       （替换 Semaphore 会导致已等待协程永久挂起，产生 deadlock）
#     - 收缩：设置 _shrink_pending 标志，下次 release() 时消耗许可
#
#   AsyncVideoInferenceServer
#     - 封装 AsyncLLMEngine + 并发控制器
#     - 支持 async context manager（async with）
#
# 版本基线: vLLM v0.8.x (AsyncLLMEngine API), Python 3.10+

import asyncio
import time
import logging
import numpy as np
from collections import deque
from PIL import Image
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
from video_inference.video_pipeline import build_vllm_messages

logger = logging.getLogger(__name__)


class AdaptiveConcurrencyController:
    """
    自适应并发度控制器。

    在 P95 延迟 SLO 约束下动态调整 asyncio.Semaphore 的许可数，
    最大化 GPU 利用率的同时保证延迟目标。

    参数：
      initial  : 初始并发许可数（建议从保守值 4 开始）
      maximum  : 最大并发许可数上限
      slo_p95  : P95 延迟 SLO 上限（ms），超过时停止扩容并尝试收缩
      safe_p95 : P95 延迟安全线（ms），低于时允许尝试扩容
    """

    def __init__(
        self,
        initial: int = 4,
        maximum: int = 16,
        slo_p95: float = 450.0,
        safe_p95: float = 300.0,
        window_size: int = 100,
    ):
        self.current = initial
        self.maximum = maximum
        self.slo_p95 = slo_p95
        self.safe_p95 = safe_p95
        self._semaphore = asyncio.Semaphore(initial)
        self._window: deque = deque(maxlen=window_size)
        self._shrink_pending: int = 0
        self._lock = asyncio.Lock()

    def record_latency(self, latency_ms: float) -> None:
        self._window.append(latency_ms)

    @property
    def p95_latency(self) -> float:
        if len(self._window) < 20:
            return 0.0
        return float(np.percentile(list(self._window), 95))

    async def acquire(self) -> None:
        """
        获取并发许可，并根据 P95 状态决定扩容或收缩。

        扩容：对现有 Semaphore 调用 release()，增加可用许可数，
              不创建新 Semaphore 对象（保证已等待协程的引用有效）。
        收缩：设置 _shrink_pending，在下次 release() 中消耗一个许可。
        """
        async with self._lock:
            p95 = self.p95_latency
            if p95 > 0 and p95 < self.safe_p95 and self.current < self.maximum:
                self._semaphore.release()   # 增加许可（对同一对象！）
                self.current += 1
                logger.debug(f"并发度扩容至 {self.current}，P95={p95:.1f}ms")
            elif p95 > self.slo_p95 and self.current > 1:
                self._shrink_pending += 1
                self.current -= 1
                logger.debug(f"并发度收缩至 {self.current}，P95={p95:.1f}ms 超 SLO")

        await self._semaphore.acquire()

    def release(self) -> None:
        """
        释放并发许可。若有待消耗的收缩许可，本次不调用 semaphore.release()，
        实现净许可数减少 1（收缩效果）。
        """
        if self._shrink_pending > 0:
            self._shrink_pending -= 1
        else:
            self._semaphore.release()

    @property
    def stats(self) -> dict:
        return {
            "current_concurrency": self.current,
            "p95_latency_ms":      round(self.p95_latency, 1),
            "sample_count":        len(self._window),
        }


class AsyncVideoInferenceServer:
    """
    异步视频推理服务（全四层优化）。

    使用示例：
        async with AsyncVideoInferenceServer(target_path, draft_path) as server:
            text, latency = await server.infer(image)
    """

    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        num_speculative_tokens: int = 5,
        max_pixels: int = 200_704,
        gpu_memory_utilization: float = 0.90,
        initial_concurrency: int = 4,
        max_concurrency: int = 16,
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.num_speculative_tokens = num_speculative_tokens
        self.max_pixels = max_pixels
        self.gpu_memory_utilization = gpu_memory_utilization
        self._engine: AsyncLLMEngine = None
        self._controller = AdaptiveConcurrencyController(
            initial=initial_concurrency,
            maximum=max_concurrency,
        )
        self._default_params = SamplingParams(temperature=0.1, max_tokens=128)

    async def start(self) -> None:
        """初始化 AsyncLLMEngine（全四层优化）。"""
        max_model_len = 1024 if self.max_pixels <= 200_704 else 2048
        args = AsyncEngineArgs(
            model=self.target_model_path,
            speculative_model=self.draft_model_path,
            num_speculative_tokens=self.num_speculative_tokens,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=False,
            enable_prefix_caching=True,
            max_num_seqs=self._controller.maximum,
            max_num_batched_tokens=8192,
            mm_processor_kwargs={
                "min_pixels": 50_176,
                "max_pixels": self.max_pixels,
            },
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)
        logger.info("AsyncVideoInferenceServer 启动完成")

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.abort("shutdown")
            self._engine = None

    async def infer(
        self,
        image: Image.Image,
        user_prompt: str = "描述这个视频帧中的内容。",
        sampling_params: SamplingParams = None,
    ) -> tuple[str, float]:
        """异步推理单帧，返回 (生成文本, 端到端延迟毫秒)。"""
        if sampling_params is None:
            sampling_params = self._default_params
        messages = build_vllm_messages(image, user_prompt=user_prompt)
        request_id = random_uuid()

        await self._controller.acquire()
        t0 = time.perf_counter()
        try:
            full_text = ""
            async for output in self._engine.generate(messages, sampling_params, request_id):
                if output.outputs:
                    full_text = output.outputs[0].text
            latency_ms = (time.perf_counter() - t0) * 1000
            self._controller.record_latency(latency_ms)
            return full_text, latency_ms
        except Exception as e:
            logger.error(f"推理失败 (request_id={request_id}): {e}")
            raise
        finally:
            self._controller.release()

    @property
    def concurrency_stats(self) -> dict:
        return self._controller.stats

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
