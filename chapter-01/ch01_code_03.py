def compare_kv_cache_optimizations():
    """
    比较不同KV Cache优化技术的效果
    """
    # 基准配置
    base_memory_gb = 5.24  # 单个请求的KV Cache
    batch_size = 32
    base_total_gb = base_memory_gb * batch_size

    optimizations = {
        "无优化": {"节省比例": 0.0, "复杂度": "低"},
        "PagedAttention": {"节省比例": 0.65, "复杂度": "中"},
        "Prefix Caching": {"节省比例": 0.35, "复杂度": "低"},
        "INT8量化": {"节省比例": 0.50, "复杂度": "高"},
        "INT4量化": {"节省比例": 0.75, "复杂度": "高"},
        "组合优化": {"节省比例": 0.85, "复杂度": "很高"},
    }

    print("=== KV Cache优化技术对比 ===\n")
    print(f"基准场景: {batch_size}个请求,每请求{base_memory_gb:.2f}GB")
    print(f"总内存需求: {base_total_gb:.2f}GB\n")
    print(f"{'优化技术':<20} {'节省比例':<10} {'实际占用(GB)':<15} {'实现复杂度':<10}")
    print("-" * 65)

    for name, info in optimizations.items():
        actual_memory = base_total_gb * (1 - info["节省比例"])
        print(f"{name:<20} {info['节省比例'] * 100:>6.0f}%    {actual_memory:>10.2f}      {info['复杂度']:<10}")

    print("\n推荐策略:")
    print("  - 通用场景: PagedAttention (vLLM)")
    print("  - 多轮对话: Prefix Caching + PagedAttention (SGLang)")
    print("  - 极限优化: 量化 + PagedAttention")


compare_kv_cache_optimizations()