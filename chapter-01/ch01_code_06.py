def demonstrate_radix_attention():
    """
    演示RadixAttention在多轮对话中的优势
    """
    print("=== RadixAttention vs PagedAttention ===\n")
    print("场景:多轮对话,系统提示词固定\n")

    system_prompt = "You are a helpful assistant."
    conversations = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What about Germany?", "The capital of Germany is Berlin."),
        ("And Spain?", "The capital of Spain is Madrid.")
    ]

    # 模拟计算开销(tokens)
    system_tokens = 10
    avg_question_tokens = 8
    avg_answer_tokens = 12

    # PagedAttention: 每轮需要重新处理部分历史
    paged_total = 0
    print("PagedAttention计算:")
    for i, (q, a) in enumerate(conversations):
        # 需要处理系统提示词 + 所有历史对话 + 当前问题
        context_tokens = system_tokens + i * (avg_question_tokens + avg_answer_tokens) + avg_question_tokens
        paged_total += context_tokens
        print(f"  轮次{i + 1}: {context_tokens} tokens")

    print(f"\nPagedAttention总计: {paged_total} tokens\n")

    # RadixAttention: 共享前缀,只处理增量
    radix_total = system_tokens  # 系统提示词只处理一次
    print("RadixAttention计算:")
    print(f"  系统提示词(共享): {system_tokens} tokens")
    for i, (q, a) in enumerate(conversations):
        # 只需处理新的问题和答案
        new_tokens = avg_question_tokens + avg_answer_tokens
        radix_total += new_tokens
        print(f"  轮次{i + 1}(增量): {new_tokens} tokens")

    print(f"\nRadixAttention总计: {radix_total} tokens\n")

    # 计算节省
    savings = (paged_total - radix_total) / paged_total * 100
    print(f"计算节省: {savings:.1f}%")
    print(f"加速比: {paged_total / radix_total:.2f}x")

    return savings


demonstrate_radix_attention()