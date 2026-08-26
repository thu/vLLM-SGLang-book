def choose_framework_guide():
    """
    框架选择指南
    """
    scenarios = {
        "API服务 (OpenAI兼容)": {
            "推荐": "vLLM",
            "理由": "开箱即用,性能优异,兼容性好"
        },
        "批量文档处理": {
            "推荐": "vLLM",
            "理由": "高吞吐量,简单部署"
        },
        "多轮对话系统": {
            "推荐": "SGLang",
            "理由": "RadixAttention节省50%计算,支持复杂对话逻辑"
        },
        "多智能体协作": {
            "推荐": "SGLang",
            "理由": "灵活调度,DSL简化编程"
        },
        "RAG应用": {
            "推荐": "vLLM或SGLang",
            "理由": "vLLM简单,SGLang可优化检索逻辑"
        },
        "代码生成": {
            "推荐": "vLLM",
            "理由": "通常单轮生成,vLLM高效"
        },
        "复杂工作流": {
            "推荐": "SGLang",
            "理由": "DSL编排多步骤,调度灵活"
        }
    }

    print("=== 推理框架选择指南 ===\n")
    print(f"{'应用场景':<20} {'推荐框架':<12} {'选择理由':<40}")
    print("-" * 75)

    for scenario, info in scenarios.items():
        print(f"{scenario:<20} {info['推荐']:<12} {info['理由']:<40}")

    print("\n综合建议:")
    print("  1. 首次使用:从vLLM开始,学习曲线平缓")
    print("  2. 性能优先:vLLM的PagedAttention是行业标杆")
    print("  3. 复杂逻辑:SGLang的DSL大幅简化开发")
    print("  4. 两者结合:可在同一系统中使用不同框架处理不同任务")


choose_framework_guide()