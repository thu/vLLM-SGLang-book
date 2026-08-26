def set_learning_goals():
    """
    设定学习目标检查清单
    """
    goals = {
        "基础理解": [
            "理解KV Cache的工作原理和内存开销",
            "掌握延迟vs吞吐量的权衡",
            "了解主流推理框架的演进历史"
        ],
        "vLLM掌握": [
            "理解PagedAttention的设计思想",
            "能够部署和配置vLLM服务",
            "会分析和优化vLLM性能"
        ],
        "SGLang应用": [
            "掌握SGLang DSL编程",
            "理解RadixAttention的优势",
            "能够实现复杂生成逻辑"
        ],
        "进阶能力": [
            "设计分布式推理架构",
            "进行系统监控和故障排查",
            "跟踪和评估新技术"
        ]
    }

    print("=== 学习目标检查清单 ===\n")

    for category, items in goals.items():
        print(f"{category}:")
        for i, item in enumerate(items, 1):
            print(f"  □ {item}")
        print()

    print("建议:")
    print("  - 每完成一章,回顾相关目标")
    print("  - 通过实践项目巩固学习成果")
    print("  - 参与社区讨论,分享经验")


set_learning_goals()