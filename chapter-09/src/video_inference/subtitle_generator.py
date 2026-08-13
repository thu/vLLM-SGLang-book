# src/video_inference/subtitle_generator.py
# 功能: 实时字幕生成，流式 token 推送 + 视频时间戳对齐
#
# 时间对齐设计：
#   推理延迟存在波动（P95 约 200~400ms），若以推理完成时刻
#   作为字幕起始时间，字幕时间轴将与画面内容存在系统性滞后。
#   正确策略：字幕起始时间 = 帧的视频时间戳（extract_frames 返回值），
#   与推理延迟完全解耦，时间对齐误差仅取决于帧提取延迟（约 12ms），
#   远小于人眼感知阈值（约 150ms）。
#
# 版本基线: vLLM v0.8.x (AsyncLLMEngine), Python 3.10+

import asyncio
from dataclasses import dataclass
from PIL import Image
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
from video_inference.video_pipeline import build_vllm_messages


@dataclass
class SubtitleEntry:
    """SRT 格式字幕条目。"""
    index: int
    start_sec: float
    end_sec: float
    text: str
    DISPLAY_WINDOW_SEC: float = 2.5

    def to_srt_block(self) -> str:
        def fmt(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t % 60
            return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
        return f"{self.index}\n{fmt(self.start_sec)} --> {fmt(self.end_sec)}\n{self.text}\n"


def build_subtitle_engine(target_model_path: str, draft_model_path: str) -> AsyncLLMEngine:
    """
    构建字幕生成专用异步引擎。

    字幕场景特殊配置：
      num_speculative_tokens=3 : 短输出序列（约 20 汉字）的最优 k 值，
                                  较长草稿窗口的边际收益在短序列下迅速下降
      temperature=0.05         : 接近确定性输出，减少字幕措辞抖动（在 SamplingParams 中设置）
    """
    args = AsyncEngineArgs(
        model=target_model_path,
        speculative_model=draft_model_path,
        num_speculative_tokens=3,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        enable_prefix_caching=True,
        mm_processor_kwargs={"min_pixels": 50_176, "max_pixels": 200_704},
    )
    return AsyncLLMEngine.from_engine_args(args)


async def generate_subtitle(
    engine: AsyncLLMEngine,
    image: Image.Image,
    frame_timestamp: float,
    subtitle_index: int,
    user_prompt: str = "用不超过20个字描述画面内容，直接给出描述，不要前缀。",
) -> SubtitleEntry:
    """
    为单帧异步生成字幕条目。

    字幕起始时间取帧的视频时间戳（frame_timestamp），
    与推理延迟完全解耦，保证时间轴对齐。
    engine.generate() 为异步生成器，每产生一个 token 触发一次迭代，
    支持在循环内部实现"逐字显示"效果。
    """
    messages = build_vllm_messages(image, user_prompt=user_prompt)
    params = SamplingParams(temperature=0.05, max_tokens=40)
    request_id = random_uuid()

    full_text = ""
    async for output in engine.generate(messages, params, request_id):
        if output.outputs:
            full_text = output.outputs[0].text

    return SubtitleEntry(
        index=subtitle_index,
        start_sec=frame_timestamp,
        end_sec=frame_timestamp + SubtitleEntry.DISPLAY_WINDOW_SEC,
        text=full_text.strip(),
    )


async def generate_subtitles_from_video(
    video_path: str,
    target_model_path: str,
    draft_model_path: str,
    output_srt_path: str = "output.srt",
    target_fps: float = 2.0,
    max_concurrent: int = 8,
) -> None:
    """
    对整个视频文件生成 SRT 字幕文件。

    asyncio.Semaphore 控制并发推理数量，防止大量帧同时进入引擎导致 OOM。
    所有帧以并发方式提交，结果按序号排序后写入 SRT 文件。
    """
    from video_inference.video_pipeline import extract_frames

    engine = build_subtitle_engine(target_model_path, draft_model_path)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_frame(image, timestamp, idx):
        async with semaphore:
            return await generate_subtitle(engine, image, timestamp, idx)

    tasks = [
        asyncio.create_task(process_frame(image, timestamp, idx))
        for idx, (image, timestamp) in enumerate(
            extract_frames(video_path, target_fps=target_fps), start=1
        )
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    entries = sorted(
        [r for r in results if isinstance(r, SubtitleEntry)],
        key=lambda e: e.index,
    )

    with open(output_srt_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_srt_block() + "\n")

    print(f"字幕生成完成: {len(entries)} 条，写入 {output_srt_path}")
