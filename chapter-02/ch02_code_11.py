class CheckpointedSequence:
    """
    支持检查点的序列
    功能：通过检查点加速抢占后的恢复
    """

    def __init__(self):
        self.checkpoints = []  # (position, block_table_snapshot)
        self.input_ids = []
        self.num_computed_tokens = 0

    def create_checkpoint(self):
        """
        创建检查点
        功能：保存当前状态快照
        """
        checkpoint = (
            self.num_computed_tokens,
            self.block_table.copy() if hasattr(self, 'block_table') else None
        )
        self.checkpoints.append(checkpoint)

    def restore_from_checkpoint(self, checkpoint_idx: int):
        """
        从检查点恢复
        功能：恢复到指定检查点状态
        """
        if checkpoint_idx >= len(self.checkpoints):
            raise ValueError("检查点索引越界")

        position, block_table = self.checkpoints[checkpoint_idx]
        self.block_table = block_table
        self.num_computed_tokens = position

        # 只需重计算从检查点到当前位置的部分
        return self.input_ids[position:]


# 使用示例
if __name__ == "__main__":
    # 创建序列
    seq = CheckpointedSequence()
    seq.input_ids = list(range(100))

    # 模拟计算
    seq.num_computed_tokens = 50
    seq.create_checkpoint()

    # 继续计算
    seq.num_computed_tokens = 100

    # 抢占后恢复
    tokens_to_recompute = seq.restore_from_checkpoint(0)
    print(f"需要重计算的tokens:{len(tokens_to_recompute)}")
    print(f"恢复到位置：{seq.num_computed_tokens}")