class ContinuousBatchingSimulator:
    """
    通过模拟器，展示连续批处理相对于静态批处理的优势。
    """

    def __init__(self):
        self.time = 0
        self.active_requests = []
        self.completed_requests = []

    def static_batching(self, requests, batch_size):
        """静态批处理：等待批次填满或超时。"""
        total_time = 0
        batches = []

        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            # 批次中最长的请求决定整个批次的执行时间
            batch_time = max(req["length"] for req in batch)
            total_time += batch_time

            batches.append({
                "requests": len(batch),
                "time": batch_time,
                "waste": sum(
                    batch_time - req["length"] for req in batch
                )
            })

        return total_time, batches

    def continuous_batching(self, requests):
        """连续批处理：每次迭代后动态调整批次。"""
        total_time = 0
        active = []
        iterations = []
        request_idx = 0

        while request_idx < len(requests) or active:
            # 添加新到达的请求
            while (
                request_idx < len(requests)
                and (
                    not active
                    or requests[request_idx]["arrival"] <= total_time
                )
            ):
                active.append({
                    "id": request_idx,
                    "remaining": requests[request_idx]["length"]
                })
                request_idx += 1

            if not active:
                break

            # 执行一次迭代
            total_time += 1
            iteration_batch_size = len(active)

            # 更新剩余长度，移除已完成的请求
            active = [
                {**req, "remaining": req["remaining"] - 1}
                for req in active
                if req["remaining"] > 1
            ]

            iterations.append({
                "time": total_time,
                "batch_size": iteration_batch_size
            })

        return total_time, iterations


def compare_batching_methods():
    """比较静态批处理和连续批处理。"""

    # 模拟请求：每个请求都有到达时间和生成长度
    requests = [
        {"arrival": 0, "length": 10},
        {"arrival": 0, "length": 100},
        {"arrival": 5, "length": 50},
        {"arrival": 15, "length": 20},
        {"arrival": 20, "length": 80},
    ]

    simulator = ContinuousBatchingSimulator()

    # 静态批处理
    static_time, static_batches = simulator.static_batching(
        requests,
        batch_size=2
    )

    # 连续批处理
    continuous_time, continuous_iterations = (
        simulator.continuous_batching(requests)
    )

    print("=== 批处理方法对比 ===\n")
    print(f"请求数量：{len(requests)}")
    print(f"请求长度：{[req['length'] for req in requests]}\n")

    print("静态批处理：")
    print(f"  总时间：{static_time} 迭代")
    print(f"  批次数：{len(static_batches)}")

    total_waste = sum(batch["waste"] for batch in static_batches)
    print(f"  浪费的计算：{total_waste} 迭代\n")

    print("连续批处理：")
    print(f"  总时间：{continuous_time} 迭代")

    average_batch_size = (
        sum(item["batch_size"] for item in continuous_iterations)
        / len(continuous_iterations)
    )
    print(f"  平均批大小：{average_batch_size:.1f}")

    improvement = (
        (static_time - continuous_time) / static_time * 100
    )
    print(f"\n性能提升：{improvement:.1f}%")


if __name__ == "__main__":
    compare_batching_methods()
