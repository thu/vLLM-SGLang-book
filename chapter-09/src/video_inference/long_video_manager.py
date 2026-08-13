# src/video_inference/long_video_manager.py
# 功能: 长视频滑动窗口上下文管理
#
# 实现说明：
#   本模块管理"应用层上下文"（文本历史），即将哪些历史帧描述
#   拼接进当前帧的 prompt。KV Cache 的物理驱逐由 vLLM 内部 LRU 机制负责，
#   应用层无法直接控制 vLLM 的 KV Cache 生命周期。
#
# KV Cache 显存估算（30 秒窗口，256 token/帧，5 帧/秒）：
#   注意：应用层历史主要影响 prompt 文本长度，而非 visual token KV Cache。
#   每帧 visual token 是独立的（不共享 KV），因此 KV Cache 主要来自
#   历史文本上下文部分，实际显存开销远低于直接按帧数估算的结果。
#
# 版本基线: Python 3.10+（无 vLLM 依赖，纯应用层逻辑）

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameRecord:
    """单帧分析记录。"""
    timestamp_sec: float
    description: str


class SlidingWindowContextManager:
    """
    滑动窗口上下文管理器。

    维护最近 window_sec 秒内的帧描述历史，为当前帧推理请求提供
    历史上下文 prompt，支持场景级事件追踪（人物行动轨迹、多轮对话等）。

    参数：
      window_sec           : 保留的历史帧时间窗口（秒）
      max_history_frames   : 历史帧数上限（防止 prompt 超出 max_model_len）
      scene_change_gap_sec : 帧间时间戳差超过此值时视为场景切换，自动重置窗口
                             实时流建议设为较大值（如 10.0），
                             有剪辑跳转的离线视频建议 5.0
    """

    def __init__(
        self,
        window_sec: float = 30.0,
        max_history_frames: int = 10,
        scene_change_gap_sec: float = 5.0,
    ):
        self.window_sec = window_sec
        self.max_history_frames = max_history_frames
        self.scene_change_gap_sec = scene_change_gap_sec
        self._history: deque = deque()

    def add_frame(self, timestamp_sec: float, description: str) -> bool:
        """
        添加新帧记录，自动处理时间窗口驱逐和场景切换检测。

        返回 True 表示检测到场景切换（窗口已重置），False 为正常添加。
        """
        scene_changed = False
        if (self._history
                and timestamp_sec - self._history[-1].timestamp_sec
                > self.scene_change_gap_sec):
            self._history.clear()
            scene_changed = True

        self._history.append(FrameRecord(timestamp_sec, description.strip()))

        # 驱逐超出时间窗口的历史帧
        cutoff = timestamp_sec - self.window_sec
        while self._history and self._history[0].timestamp_sec < cutoff:
            self._history.popleft()

        # 驱逐超出帧数上限的历史帧
        while len(self._history) > self.max_history_frames:
            self._history.popleft()

        return scene_changed

    def reset(self) -> None:
        """手动重置滑动窗口（外部触发场景切换信号时调用）。"""
        self._history.clear()

    def get_context_prompt(self) -> str:
        """
        返回当前窗口内帧描述的历史上下文，格式化为 prompt 文本。

        若窗口为空（首帧或场景切换后），返回空字符串。
        每条记录包含视频时间戳，帮助模型理解时序关系。

        示例输出：
          历史上下文（最近帧描述）:
          [0.0s] 一名男子从停车场走向大楼入口
          [1.0s] 男子刷卡进入大楼，保安在旁登记
        """
        if not self._history:
            return ""
        lines = [f"[{r.timestamp_sec:.1f}s] {r.description}" for r in self._history]
        return "历史上下文（最近帧描述）:\n" + "\n".join(lines) + "\n\n"

    @property
    def history_count(self) -> int:
        return len(self._history)

    @property
    def oldest_timestamp(self) -> Optional[float]:
        return self._history[0].timestamp_sec if self._history else None

    @property
    def newest_timestamp(self) -> Optional[float]:
        return self._history[-1].timestamp_sec if self._history else None


def build_contextual_messages(
    image,
    context_manager: SlidingWindowContextManager,
    system_prefix: str,
    base_prompt: str = "结合历史上下文，描述当前帧的内容和事件进展。",
) -> list:
    """
    构建携带历史上下文的多模态消息。

    将滑动窗口的历史帧描述拼接到当前帧的 user prompt 中，
    使模型能够理解时序事件的演进。
    """
    from video_inference.video_pipeline import pil_to_data_uri

    context = context_manager.get_context_prompt()
    full_prompt = context + base_prompt if context else base_prompt
    data_uri = pil_to_data_uri(image)
    return [
        {"role": "system", "content": system_prefix},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": full_prompt},
            ],
        },
    ]
