class FirstFitAllocator:
    """
    首次适应分配器
    功能：从链表头部开始分配第一个可用块
    """

    def __init__(self, num_blocks: int):
        self.free_blocks = list(range(num_blocks))

    def allocate(self, num_blocks: int) -> list:
        """
        分配指定数量的块
        功能：从空闲列表中取出所需数量的块
        """
        if len(self.free_blocks) < num_blocks:
            return None  # 内存不足

        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        return allocated

    def free(self, blocks: list):
        """
        释放块
        功能：将块归还到空闲列表
        """
        self.free_blocks.extend(blocks)


# 使用示例
if __name__ == "__main__":
    allocator = FirstFitAllocator(num_blocks=100)

    # 分配10个块
    allocated = allocator.allocate(10)
    print(f"分配的块：{allocated}")

    # 释放5个块
    allocator.free(allocated[:5])
    print(f"剩余空闲块数：{len(allocator.free_blocks)}")