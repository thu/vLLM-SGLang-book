# pipeline/cache_scheduler.py
# 自适应缓存调度器（ch11_code_08）
#
# 设计说明：
#   SGLang 的 RadixAttention 在内核层使用 LRU 策略管理
#   缓存淘汰，应用层无法直接"锁定"基数树节点以阻止淘汰。
#   引用计数（refcount）是 SGLang 内部机制，仅由框架
#   在请求处理的生命周期内自动维护：请求开始时 refcount+1，
#   请求完成后 refcount-1；refcount=0 且无子节点时，
#   节点才进入 LRU 淘汰候选集。
#
#   应用层能够做的是：通过调整请求的处理顺序和并发策略，
#   使高频文书的 KV Cache 尽量晚进入淘汰候选集。
#   具体手段是：将高频文书的处理任务密集安排在连续时间
#   窗口内（使其长时间保持 refcount>0），而不是在
#   批次中分散穿插（导致长时间 refcount=0 而被淘汰）。
#
# 淘汰行为的关键差异：
#   SGLang（基数树）：叶节点优先淘汰，父节点（共享前缀）
#                      只有在所有子节点被淘汰后才可能被淘汰，
#                      保证缓存结构的完整性。
#   vLLM（哈希表）  ：按 LRU 时间戳独立淘汰各物理块，
#                      可能出现文书中间段块已淘汰而首尾块
#                      仍在缓存的碎片化状态，此时命中中间块
#                      也无法被利用（前缀匹配必须从头开始）。

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from data.dataset_builder import ContractDocument


@dataclass
class DocAccessRecord:
    """单份文书的访问记录。"""
    doc_id: str
    access_count: int = 0
    last_access_ts: float = field(
        default_factory=time.time
    )
    first_access_ts: float = field(
        default_factory=time.time
    )


class AdaptiveCacheScheduler:
    """
    自适应缓存调度器。

    核心功能：
      1. 记录文书访问频率，识别高频文书。
      2. 对文书批次重排序：使高频文书的处理任务集中安排
         在连续时间窗口内，延长其 KV Cache 在缓存中的
         有效保留时间（通过保持 refcount>0 实现，而非
         直接操作框架内部状态）。
      3. 根据访问模式选择最优的并发度配置。

    访问模式识别：
      纯批次处理（最常见）：LRU 天然适配，无需干预。
      批次+临时查询：识别临时查询目标文书，在重排序时
                     提高其处理优先级，确保其 KV Cache
                     在临时查询到来时仍在缓存中。
      多客户端并发：通过频率计数识别跨客户端的热点文书，
                    优先安排以降低其重复 prefill 的概率。
    """

    # 频率计数的统计窗口（秒）
    FREQ_WINDOW_SECONDS: float = 3600.0
    # 高频阈值：窗口内访问次数超过此值视为高频文书
    HIGH_FREQ_THRESHOLD: int = 10

    def __init__(self) -> None:
        # doc_id → 访问记录
        self._records: dict[str, DocAccessRecord] = {}
        # 时间窗口内的访问事件列表：[(timestamp, doc_id)]
        self._window_events: list[tuple[float, str]] = []

    def record_access(self, doc_id: str) -> None:
        """记录一次文书访问。"""
        now = time.time()
        if doc_id not in self._records:
            self._records[doc_id] = DocAccessRecord(
                doc_id=doc_id,
                first_access_ts=now,
                last_access_ts=now,
            )
        rec = self._records[doc_id]
        rec.access_count += 1
        rec.last_access_ts = now
        self._window_events.append((now, doc_id))
        # 清理过期事件，保持窗口大小
        cutoff = now - self.FREQ_WINDOW_SECONDS
        self._window_events = [
            (ts, did) for ts, did in self._window_events
            if ts >= cutoff
        ]

    def get_access_frequency(self, doc_id: str) -> int:
        """
        获取文书在当前统计窗口内的访问频率。

        返回值用于判断是否为高频文书，而非用于直接控制
        框架的缓存淘汰行为。
        """
        now = time.time()
        cutoff = now - self.FREQ_WINDOW_SECONDS
        return sum(
            1 for ts, did in self._window_events
            if ts >= cutoff and did == doc_id
        )

    def is_high_frequency(self, doc_id: str) -> bool:
        """判断文书是否为高频访问文书。"""
        return (
            self.get_access_frequency(doc_id)
            >= self.HIGH_FREQ_THRESHOLD
        )

    def reorder_batch(
        self,
        docs: list[ContractDocument],
    ) -> list[ContractDocument]:
        """
        对文书批次重排序，使高频文书优先处理。

        排序逻辑（优先级由高到低）：
          1. 高频文书（访问次数超过阈值）
          2. 中频文书（有历史访问记录）
          3. 新文书（首次出现）
          同优先级内按访问频率降序排列。

        重排序使高频文书的 KV Cache 在批次开始时就写入
        缓存，并在后续大量其他文书处理期间持续被引用
        （若批次中存在针对高频文书的临时查询），
        从而降低其被 LRU 淘汰的概率。
        """
        def sort_key(
            doc: ContractDocument,
        ) -> tuple[int, int]:
            freq = self.get_access_frequency(doc.doc_id)
            if freq >= self.HIGH_FREQ_THRESHOLD:
                # 高频：优先级 0，频率降序
                return (0, -freq)
            elif doc.doc_id in self._records:
                # 中频：优先级 1
                return (1, -freq)
            else:
                # 新文书：优先级 2
                return (2, 0)

        return sorted(docs, key=sort_key)

    def recommend_concurrency(self) -> int:
        """
        根据当前访问模式推荐并发文书数。

        判断逻辑：
          若高频文书比例较高（多客户端并发模式），
          建议降低并发度以减少缓存竞争；
          否则使用标准并发度。
        """
        if not self._window_events:
            return 8  # 默认值
        total = len(self._window_events)
        high_freq_count = sum(
            1 for _, did in self._window_events
            if self.is_high_frequency(did)
        )
        high_freq_ratio = high_freq_count / max(total, 1)
        if high_freq_ratio > 0.5:
            # 多客户端并发模式：降低并发度减少缓存竞争
            return 4
        return 8
