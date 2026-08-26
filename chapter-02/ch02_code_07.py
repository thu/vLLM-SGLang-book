class BlockTable:
    """
    块表
    功能：管理逻辑块到物理块的映射及引用计数
    """

    def __init__(self):
        self.blocks = []  # 逻辑块列表
        self.ref_counts = []  # 每个块的引用计数

    def append_block(self, block_id: int, is_shared: bool = False):
        """
        添加新块到块表
        功能：初始化引用计数
        """
        self.blocks.append(block_id)
        initial_ref = 2 if is_shared else 1
        self.ref_counts.append(initial_ref)

    def fork(self) -> 'BlockTable':
        """
        创建共享前缀的新块表
        功能：用于beam search或multi-turn对话
        """
        new_table = BlockTable()
        new_table.blocks = self.blocks.copy()
        new_table.ref_counts = self.ref_counts.copy()

        # 增加所有共享块的引用计数
        for i in range(len(self.blocks)):
            self.ref_counts[i] += 1
            new_table.ref_counts[i] = self.ref_counts[i]

        return new_table

    def copy_on_write(self, block_index: int) -> int:
        """
        写时复制：如果块被共享，复制一份
        功能：确保写操作不影响其他共享者
        """
        if self.ref_counts[block_index] > 1:
            # 块被共享，需要复制
            old_block = self.blocks[block_index]
            new_block = self.block_manager.copy_block(old_block)

            # 更新块表
            self.ref_counts[block_index] -= 1
            self.blocks[block_index] = new_block
            self.ref_counts[block_index] = 1

            return new_block
        else:
            # 块未被共享，直接返回
            return self.blocks[block_index]


# 使用示例
if __name__ == "__main__":
    # 创建块表
    table1 = BlockTable()
    table1.append_block(block_id=0)
    table1.append_block(block_id=1)

    # Fork创建共享块表
    table2 = table1.fork()
    print(f"Table1块：{table1.blocks}, 引用计数：{table1.ref_counts}")
    print(f"Table2块：{table2.blocks}, 引用计数：{table2.ref_counts}")

    # 写时复制
    new_block = table1.copy_on_write(block_index=0)
    print(f"复制后Table1块：{table1.blocks}, 引用计数：{table1.ref_counts}")