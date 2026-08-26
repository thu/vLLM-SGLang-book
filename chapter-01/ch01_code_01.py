# 计算GPU算力与带宽比例
def calculate_compute_memory_ratio():
    """
    计算NVIDIA A100的算力与内存带宽比例
    """
    # A100 FP16算力 (TFLOPS)
    compute_power = 312  # TFLOPS

    # A100 HBM2内存带宽 (TB/s)
    memory_bandwidth = 1.5  # TB/s

    # 计算比例 (FLOPS per Byte)
    # 1 TFLOPS = 10^12 FLOPS, 1 TB/s = 10^12 Bytes/s
    ratio = compute_power / memory_bandwidth

    print(f"A100 算力: {compute_power} TFLOPS")
    print(f"A100 带宽: {memory_bandwidth} TB/s")
    print(f"算力/带宽比: {ratio:.1f} FLOPS/Byte")
    print(f"\n这意味着每传输1字节数据,理论上可以执行{ratio:.0f}次浮点运算")
    print(f"但实际推理中,大量时间消耗在数据传输上,算力无法充分利用")

    return ratio


# 运行示例
calculate_compute_memory_ratio()