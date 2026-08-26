from typing import List
from enum import Enum


class SequenceState(Enum):
    """序列状态枚举"""
    WAITING = 1
    RUNNING = 2
    SWAPPED = 3
    FINISHED = 4


class SequenceScheduler:
    """
    序列调度器
    功能：管理序列的调度和抢占
    """

    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.running_sequences = []
        self.swapped_sequences = []

    def preempt_by_lru(self, required_blocks: int) -> List:
        """
        按最近最少使用原则抢占序列
        功能：选择最久未执行的序列进行抢占
        """
        # 按最后执行时间排序
        candidates = sorted(
            self.running_sequences,
            key=lambda seq: seq.last_exec_time
        )

        preempted = []
        freed_blocks = 0

        for seq in candidates:
            if freed_blocks >= required_blocks:
                break

            # 抢占序列
            seq_blocks = seq.get_num_blocks()
            seq.set_state(SequenceState.SWAPPED)
            self.running_sequences.remove(seq)
            self.swapped_sequences.append(seq)

            freed_blocks += seq_blocks
            preempted.append(seq)

        return preempted


# 使用示例
if __name__ == "__main__":
    from dataclasses import dataclass
    import time


    @dataclass
    class MockSequence:
        seq_id: int
        last_exec_time: float = 0.0
        state: SequenceState = SequenceState.RUNNING
        num_blocks: int = 10

        def get_num_blocks(self):
            return self.num_blocks

        def set_state(self, state):
            self.state = state


    @dataclass
    class MockBlockManager:
        pass


    # 创建调度器
    scheduler = SequenceScheduler(MockBlockManager())

    # 添加运行中的序列
    for i in range(5):
        seq = MockSequence(seq_id=i, last_exec_time=time.time() - i)
        scheduler.running_sequences.append(seq)

    # 抢占以释放30个块
    preempted = scheduler.preempt_by_lru(required_blocks=30)

    print(f"抢占的序列数：{len(preempted)}")
    print(f"抢占的序列ID:{[s.seq_id for s in preempted]}")
    print(f"剩余运行序列：{len(scheduler.running_sequences)}")