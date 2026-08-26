class PartialPreemptionScheduler(SequenceScheduler):
    """
    部分抢占调度器
    功能：保留前缀，只释放后缀KV Cache
    """

    def partial_preempt(self, seq, keep_prefix_len: int) -> int:
        """
        部分抢占：保留前缀，释放后缀
        功能：减少恢复时的重计算开销
        """
        total_blocks = seq.get_num_blocks()
        prefix_blocks = (keep_prefix_len + seq.block_size - 1) // seq.block_size

        # 释放后缀块
        suffix_blocks = seq.block_table.blocks[prefix_blocks:]
        for block in suffix_blocks:
            self.block_manager.free(block)

        # 截断块表
        seq.block_table.blocks = seq.block_table.blocks[:prefix_blocks]
        seq.num_computed_tokens = keep_prefix_len

        return total_blocks - prefix_blocks  # 返回释放的块数


# 使用示例
if __name__ == "__main__":
    @dataclass
    class MockSequenceWithBlockTable:
        seq_id: int
        block_size: int = 16
        num_computed_tokens: int = 100

        def __post_init__(self):
            # 模拟块表
            num_blocks = (self.num_computed_tokens + self.block_size - 1) // self.block_size
            self.block_table = type('BlockTable', (), {
                'blocks': list(range(num_blocks))
            })()

        def get_num_blocks(self):
            return len(self.block_table.blocks)


    @dataclass
    class MockBlockManagerWithFree:
        def free(self, block_id):
            print(f"释放块：{block_id}")


    # 创建调度器
    scheduler = PartialPreemptionScheduler(MockBlockManagerWithFree())

    # 创建序列
    seq = MockSequenceWithBlockTable(seq_id=1, num_computed_tokens=100)
    print(f"原始块数：{seq.get_num_blocks()}")

    # 部分抢占，保留前32个token
    freed = scheduler.partial_preempt(seq, keep_prefix_len=32)

    print(f"释放的块数：{freed}")
    print(f"剩余块数：{len(seq.block_table.blocks)}")