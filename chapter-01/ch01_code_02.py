import torch


def demonstrate_kv_cache():
    """
    演示KV Cache的工作原理和内存占用
    """
    # 模型配置 (以LLaMA-2 7B为例)
    num_layers = 32
    hidden_dim = 4096
    num_heads = 32
    head_dim = hidden_dim // num_heads  # 128

    # 序列配置
    batch_size = 1
    seq_length = 2048

    # 数据类型
    dtype = torch.float16
    bytes_per_element = 2  # FP16占2字节

    # 计算单个KV Cache的形状
    # [batch_size, num_heads, seq_length, head_dim]
    kv_shape = (batch_size, num_heads, seq_length, head_dim)

    print("=== KV Cache内存占用分析 ===\n")
    print(f"模型配置:")
    print(f"  层数: {num_layers}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  每个头的维度: {head_dim}\n")

    print(f"序列配置:")
    print(f"  批处理大小: {batch_size}")
    print(f"  序列长度: {seq_length}\n")

    # 计算单个KV张量的元素数量
    elements_per_kv = batch_size * num_heads * seq_length * head_dim

    # 每层有K和V两个缓存
    elements_per_layer = 2 * elements_per_kv

    # 所有层的总元素数
    total_elements = num_layers * elements_per_layer

    # 计算内存占用 (字节)
    memory_bytes = total_elements * bytes_per_element
    memory_mb = memory_bytes / (1024 ** 2)
    memory_gb = memory_bytes / (1024 ** 3)

    print(f"内存占用:")
    print(f"  单个KV张量: {elements_per_kv:,} 个元素")
    print(f"  单层KV Cache: {elements_per_layer:,} 个元素")
    print(f"  所有层总计: {total_elements:,} 个元素")
    print(f"  总内存占用: {memory_mb:.2f} MB ({memory_gb:.3f} GB)\n")

    # 计算不同批处理大小的内存占用
    print("不同批处理大小的内存占用:")
    print(f"{'批处理':<10} {'内存占用(GB)':<15}")
    print("-" * 25)
    for bs in [1, 8, 16, 32, 64]:
        mem = (bs * elements_per_layer * num_layers * bytes_per_element) / (1024 ** 3)
        print(f"{bs:<10} {mem:.2f}")

    return memory_gb


# 运行演示
demonstrate_kv_cache()