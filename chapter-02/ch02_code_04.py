def handle_out_of_memory(self):
    """
    内存不足时的抢占处理
    功能；选择合适的请求进行抢占，释放内存
    """
    # 按优先级排序，找到可抢占的请求
    preemptable = sorted(
        self.decode_batch,
        key=lambda r: (r.priority, r.last_exec_time)
    )

    for request in preemptable:
        # 释放KV Cache
        freed_blocks = request.release_kv_cache()
        self.block_manager.free(freed_blocks)

        # 移回等待队列
        request.set_state(RequestState.WAITING)
        self.waiting_queue.put(request)

        # 检查内存是否充足
        if self.has_available_memory():
            break


# 使用示例
if __name__ == "__main__":
    # 模拟内存不足场景
    scheduler = SeparatedBatchScheduler()
    scheduler.decode_batch = [Request(id=i) for i in range(100)]
    scheduler.handle_out_of_memory()