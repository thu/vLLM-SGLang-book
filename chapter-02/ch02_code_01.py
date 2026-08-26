class BlockManager:
    """
    物理块管理器
    功能：管理物理块的分配、释放和引用计数
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks = set(range(num_blocks))
        self.ref_counts = [0] * num_blocks  # 引用计数

    def allocate(self) -> int:
        """分配一个物理块"""
        if not self.free_blocks:
            raise OutOfMemoryError("No free blocks available")
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def add_reference(self, block_id: int):
        """增加引用计数，用于共享"""
        self.ref_counts[block_id] += 1

    def free(self, block_id: int):
        """释放一个引用"""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            # 引用计数归零，回收物理块
            self.free_blocks.add(block_id)


# 使用示例
if __name__ == "__main__":
    manager = BlockManager(num_blocks=1000)
    # 分配块
    block1 = manager.allocate()
    print(f"分配块： {block1}")
    # 增加引用，用于共享
    manager.add_reference(block1)
    # 释放引用
    manager.free(block1)
    manager.free(block1)  # 第二次释放，块被回收