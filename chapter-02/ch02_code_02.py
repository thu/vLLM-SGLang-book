from queue import Queue
from typing import List


class ContinuousBatchingScheduler:
    """
    连续批处理调度器
    功能；实现迭代级别的动态批次调整
    """

    def __init__(self):
        self.waiting_queue = Queue()  # 等待队列
        self.running_batch = []  # 当前运行批次
        self.finished_requests = []  # 已完成请求

    def schedule_iteration(self):
        """每个迭代步骤的调度逻辑"""
        # 步骤1，移除已完成的请求
        self.running_batch = [
            req for req in self.running_batch
            if not req.is_finished()
        ]

        # 步骤2，尝试添加新请求，直到达到内存或批大小限制
        while (len(self.running_batch) < MAX_BATCH_SIZE and
               not self.waiting_queue.empty() and
               self.has_available_memory()):
            new_request = self.waiting_queue.get()
            self.running_batch.append(new_request)

        # 步骤3，执行一步解码
        for request in self.running_batch:
            request.decode_one_step()

        return self.running_batch


# 使用示例
if __name__ == "__main__":
    scheduler = ContinuousBatchingScheduler()
    # 添加请求到等待队列
    for i in range(10):
        scheduler.waiting_queue.put(Request(id=i))
    # 执行调度
    while not scheduler.waiting_queue.empty() or scheduler.running_batch:
        scheduler.schedule_iteration()