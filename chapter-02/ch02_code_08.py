class Sequence:
    """
    序列
    功能：管理单个推理序列的完整状态
    """

    def __init__(self, block_manager, block_size=16):
        self.block_manager = block_manager
        self.block_size = block_size
        self.block_table = BlockTable()
        self.tokens = []

    def append_token(self, token_id: int):
        """
        向序列追加新token
        功能：处理块分配和写时复制
        """
        current_block_idx = len(self.block_table.blocks) - 1

        if current_block_idx < 0:
            # 第一个token，分配第一个块
            new_block = self.block_manager.allocate()
            self.block_table.append_block(new_block)
            current_block_idx = 0

        current_block = self.block_table.blocks[current_block_idx]

        # 检查当前块是否已满
        if self.get_block_utilization(current_block) == self.block_size:
            # 当前块已满，需要分配新块
            new_block = self.block_manager.allocate()
            self.block_table.append_block(new_block)
            current_block_idx += 1
            current_block = new_block
        else:
            # 当前块未满，检查是否需要写时复制
            current_block = self.block_table.copy_on_write(current_block_idx)

        # 写入新token的KV Cache
        self.write_kv_cache(current_block, token_id)
        self.tokens.append(token_id)

    def get_block_utilization(self, block_id: int) -> int:
        """
        获取块的利用率
        功能：计算块中已存储的token数量
        """
        # 实现细节：查询块中已使用的槽位数
        return len([t for t in self.tokens if self.get_token_block(t) == block_id])

    def get_token_block(self, token_idx: int) -> int:
        """获取token所在的块ID"""
        block_idx = token_idx // self.block_size
        return self.block_table.blocks[block_idx]

    def write_kv_cache(self, block_id: int, token_id: int):
        """写入KV Cache"""
        # 实际实现会调用CUDA kernel写入GPU显存
        pass


# 使用示例
if __name__ == "__main__":
    from dataclasses import dataclass


    @dataclass
    class BlockManager:
        def allocate(self):
            """分配物理块"""
            return 1  # 简化实现

        def copy_block(self, block_id):
            """复制块"""
            return block_id + 100  # 简化实现


    manager = BlockManager()
    seq = Sequence(manager, block_size=16)

    # 追加tokens
    for i in range(20):
        seq.append_token(i)

    print(f"序列长度：{len(seq.tokens)}")
    print(f"使用的块：{seq.block_table.blocks}")