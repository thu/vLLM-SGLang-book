class BlockManagerMetrics:
    """
    块管理器监控指标
    功能：收集和报告内存使用统计
    """

    def __init__(self):
        self.total_blocks = 0
        self.free_blocks = 0
        self.used_blocks = 0
        self.shared_blocks = 0

    def get_utilization(self) -> float:
        """
        内存利用率
        功能：计算已使用块占总块数的比例
        """
        if self.total_blocks == 0:
            return 0.0
        return self.used_blocks / self.total_blocks

    def get_sharing_ratio(self) -> float:
        """
        共享率：共享块占总使用块的比例
        功能：评估前缀共享的效果
        """
        if self.used_blocks == 0:
            return 0.0
        return self.shared_blocks / self.used_blocks

    def get_fragmentation(self) -> float:
        """
        碎片率：未充分利用的块比例
        功能：评估内存碎片化程度
        """
        if hasattr(self, 'blocks'):
            partially_filled = sum(
                1 for block in self.blocks
                if 0 < block.utilization < 1.0
            )
            return partially_filled / self.used_blocks if self.used_blocks > 0 else 0.0
        return 0.0


# 使用示例
if __name__ == "__main__":
    # 创建指标收集器
    metrics = BlockManagerMetrics()
    metrics.total_blocks = 1000
    metrics.free_blocks = 300
    metrics.used_blocks = 700
    metrics.shared_blocks = 150

    # 输出指标
    print(f"内存利用率：{metrics.get_utilization():.1%}")
    print(f"共享率：{metrics.get_sharing_ratio():.1%}")
    print(f"碎片率：{metrics.get_fragmentation():.1%}")