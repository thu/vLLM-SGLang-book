# src/video_inference/video_pipeline.py
# 功能: 视频帧提取与 vLLM 多模态输入构建
#   支持本地文件和 RTSP 流两种输入源
#   关键帧检测：基于帧间灰度均值差，超过阈值时强制采样
#   输出图像格式：base64 JPEG data URI（兼容 vLLM v0.8.x Python API）
#
# vLLM 图像传入说明：
#   vLLM v0.8.x 的 llm.chat() 在 image_url 字段中接受：
#     (a) 公网 URL 字符串（生产环境不推荐，有网络依赖）
#     (b) base64 data URI，格式："data:image/jpeg;base64,<b64>"
#   本模块统一使用 (b)，适合离线部署，避免网络依赖。
#
# 版本基线: vLLM v0.8.x, OpenCV 4.9+, Pillow 10+

import cv2
import base64
import io
import numpy as np
from PIL import Image
from typing import Iterator, Optional

# 固定系统前缀：所有帧请求共享，触发 vLLM Prefix Caching。
# ⚠ 修改此字符串会导致所有已缓存的 prefix KV 失效，
#   产生一次性延迟毛刺，之后重新建立缓存。
SYSTEM_PREFIX = (
    "你是一个实时视频内容分析助手，负责对视频帧进行简洁准确的语义描述。"
    "描述应包含场景类型、主要对象和正在发生的动作，不超过50字。"
)


def pil_to_data_uri(image: Image.Image, quality: int = 85) -> str:
    """
    将 PIL Image 转换为 base64 JPEG data URI。

    quality=85 在文件大小与画质之间取得平衡
    （实测 1080p 帧约 80~150KB，base64 后约 107~200KB）。

    返回格式: "data:image/jpeg;base64,<base64_string>"
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_frames(
    source: str,
    target_fps: float = 5.0,
    scene_change_threshold: float = 15.0,
    max_frames: Optional[int] = None,
) -> Iterator[tuple[Image.Image, float]]:
    """
    从视频文件或 RTSP 流提取帧，结合固定采样率与场景变化检测。

    参数：
      source                 : 本地文件路径或 RTSP URL（如 "rtsp://..."）
      target_fps             : 基础采样帧率（如 5.0 = 每秒最多提取 5 帧）
      scene_change_threshold : 帧间灰度均值差阈值（0~255 范围）
                               监控摄像头（静态背景）建议 25~30
                               体育赛事（剧烈运动）建议 8~12
                               典型值 15.0 对应约 6% 的亮度变化
      max_frames             : 最多提取的帧数（None = 不限）

    Yields: (PIL Image RGB 格式, 视频时间戳（秒）)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(src_fps / target_fps)))

    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0
    extracted = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_keyframe = False

            if frame_idx % frame_interval == 0:
                is_keyframe = True
            elif prev_gray is not None:
                diff = float(np.mean(np.abs(
                    gray.astype(np.float32) - prev_gray.astype(np.float32)
                )))
                if diff > scene_change_threshold:
                    is_keyframe = True

            if is_keyframe:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(rgb), frame_idx / src_fps
                extracted += 1
                if max_frames is not None and extracted >= max_frames:
                    break

            prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()


def build_vllm_messages(
    image: Image.Image,
    user_prompt: str = "描述这个视频帧中的内容。",
    include_system_prefix: bool = True,
    jpeg_quality: int = 85,
) -> list:
    """
    构建 vLLM 多模态消息格式（OpenAI Chat Completion 兼容）。

    参数：
      image                : PIL Image 对象（任意分辨率）
      user_prompt          : 用户提示文本
      include_system_prefix: 是否添加固定系统前缀（启用时触发 Prefix Caching）
      jpeg_quality         : JPEG 编码质量

    图像以 base64 data URI 形式传入，兼容 vLLM v0.8.x 的 Python API。
    不要修改 SYSTEM_PREFIX 内容，否则会破坏 Prefix Caching 命中率。
    """
    data_uri = pil_to_data_uri(image, quality=jpeg_quality)
    messages = []
    if include_system_prefix:
        messages.append({"role": "system", "content": SYSTEM_PREFIX})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_prompt},
        ],
    })
    return messages
