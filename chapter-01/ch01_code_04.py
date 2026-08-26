import numpy as np
import matplotlib.pyplot as plt


def analyze_batch_efficiency():
    """
    分析批处理大小与GPU利用率、延迟的关系
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # 模拟GPU利用率 (随批处理大小增加而提升,但最终饱和)
    gpu_utilization = [
        20, 35, 50, 65, 75, 85, 92, 95, 96
    ]

    # 模拟单个请求延迟 (批处理大小增加导致排队时间增加)
    latency_ms = [
        10, 12, 15, 20, 28, 40, 60, 95, 150
    ]

    # 计算吞吐量 (GPU利用率 × 批处理大小的函数)
    throughput = [
        bs * util / 100 * 10  # 归一化的吞吐量
        for bs, util in zip(batch_sizes, gpu_utilization)
    ]

    print("=== 批处理性能分析 ===\n")
    print(f"{'批大小':<8} {'GPU利用率(%)':<15} {'延迟(ms)':<12} {'相对吞吐量':<12}")
    print("-" * 50)
    for bs, util, lat, thr in zip(batch_sizes, gpu_utilization, latency_ms, throughput):
        print(f"{bs:<8} {util:<15} {lat:<12} {thr:<12.1f}")

    print("\n关键观察:")
    print("  1. 批处理大小从1增至32时,GPU利用率显著提升")
    print("  2. 批处理大小超过64后,利用率提升边际递减")
    print("  3. 延迟随批处理大小近似线性增长")
    print("  4. 最优批处理大小需要在延迟和吞吐量间权衡\n")

    return batch_sizes, gpu_utilization, latency_ms, throughput


# 运行分析
analyze_batch_efficiency()