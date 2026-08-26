class BestFitAllocator:
    """
    最佳适应分配器
    功能：选择最小的满足需求的区域
    """

    def __init__(self, num_blocks: int):
        self.free_regions = [(0, num_blocks)]  # (起始索引，大小)

    def allocate(self, num_blocks: int) -> list:
        """
        找到最小的满足需求的区域
        功能：最小化碎片产生
        """
        best_fit = None
        best_idx = -1

        for idx, (start, size) in enumerate(self.free_regions):
            if size >= num_blocks and (best_fit is None or size < best_fit[1]):
                best_fit = (start, size)
                best_idx = idx

        if best_fit is None:
            return None

        # 分配并更新空闲区域
        start, size = best_fit
        allocated = list(range(start, start + num_blocks))

        if size == num_blocks:
            del self.free_regions[best_idx]
        else:
            self.free_regions[best_idx] = (start + num_blocks, size - num_blocks)

        return allocated


# 使用示例
if __name__ == "__main__":
    allocator = BestFitAllocator(num_blocks=100)

    # 分配不同大小的块
    alloc1 = allocator.allocate(10)
    alloc2 = allocator.allocate(20)
    alloc3 = allocator.allocate(5)

    print(f"分配1:{alloc1}")
    print(f"分配2:{alloc2}")
    print(f"分配3:{alloc3}")
    print(f"剩余空闲区域：{allocator.free_regions}")