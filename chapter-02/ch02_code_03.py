class SeparatedBatchScheduler(ContinuousBatchingScheduler):
    """
    分离Prefill和Decode的调度器
    功能；针对不同阶段使用不同的批次大小和优化策略
    """

    def __init__(self):
        super().__init__()
        self.prefill_batch = []  # Prefill批次
        self.decode_batch = []  # Decode批次

    def schedule_iteration(self):
        """执行一次调度迭代"""
        # 步骤1，调度Prefill批次
        while (len(self.prefill_batch) < PREFILL_BATCH_SIZE and
               not self.waiting_queue.empty()):
            new_request = self.waiting_queue.get()
            self.prefill_batch.append(new_request)

        if self.prefill_batch:
            execute_prefill(self.prefill_batch)
            # Prefill完成后，移入Decode批次
            self.decode_batch.extend(self.prefill_batch)
            self.prefill_batch.clear()

        # 步骤2，调度Decode批次
        if self.decode_batch:
            execute_decode(self.decode_batch)
            # 移除完成的请求
            self.decode_batch = [
                req for req in self.decode_batch
                if not req.is_finished()
            ]


# 使用示例
if __name__ == "__main__":
    scheduler = SeparatedBatchScheduler()
    # 模拟请求到达
    for i in range(20):
        scheduler.waiting_queue.put(Request(id=i, prompt_len=100))
    # 执行调度
    for _ in range(50):
        scheduler.schedule_iteration()