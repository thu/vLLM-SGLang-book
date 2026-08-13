"""
第3章 代码示例 25
主题：原子函数组合与情感驱动的自适应控制流
场景：将情感分析作为原子函数，其结果驱动后续
      回应策略的选择，演示LLM输出与Python控制流
      的无缝结合以及函数组合的工程价值

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time
from typing import Dict, List

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def analyze_sentiment(s, text: str):
    """原子函数：情感分析，职责单一"""
    s += sgl.user(f"分析以下文本的情感倾向：\n{text}")
    s += sgl.assistant(
        sgl.gen("sentiment",
                choices=["积极", "消极", "中性", "混合"])
    )


@sgl.function
def generate_response_by_sentiment(s, text: str):
    """
    组合函数：先分析情感，再根据结果选择回应策略。
    展示SGLang函数间的数据流转与控制流驱动。
    """
    # 调用原子函数，同步等待结果
    sent_state = analyze_sentiment.run(text=text)
    sentiment = sent_state["sentiment"]

    # 以LLM的分类结果驱动后续生成策略
    s += sgl.system("你是一位专业的客服助手。")
    if sentiment == "消极":
        s += sgl.user(
            f"用户反馈情绪消极：{text}\n请先表达理解，再提供帮助："
        )
    else:
        s += sgl.user(f"用户反馈：{text}\n请给出专业回应：")
    s += sgl.assistant(sgl.gen("response", max_tokens=200))


# ── 扩展：更细粒度的情感驱动策略 ─────────────────────────────

@sgl.function
def classify_issue_type(s, text: str):
    """原子函数：问题类型分类。"""
    s += sgl.user(
        f"将以下用户反馈归类：\n{text}\n从给定类别中选择一项："
    )
    s += sgl.assistant(
        sgl.gen(
            "issue_type",
            choices=["产品质量", "物流配送", "售后服务", "账户问题", "其他"]
        )
    )


@sgl.function
def full_customer_service(s, text: str):
    """
    完整客服工作流：情感分析 + 问题分类 + 个性化回应。
    三个原子函数串联，展示多函数组合的数据流编排。
    """
    # 并发分析情感和问题类型（两者无数据依赖）
    sent_state = analyze_sentiment.run(text=text)
    issue_state = classify_issue_type.run(text=text)

    sentiment = sent_state["sentiment"]
    issue_type = issue_state["issue_type"]

    # 根据情感和问题类型共同决定回应策略
    s += sgl.system(
        "你是一位专业的电商客服，善于处理各类用户反馈。"
    )

    if sentiment == "消极" and issue_type == "产品质量":
        s += sgl.user(
            f"用户对产品质量不满：{text}\n"
            f"请先道歉，再说明退换货流程："
        )
    elif sentiment == "消极":
        s += sgl.user(
            f"用户情绪消极（问题类型：{issue_type}）：{text}\n"
            f"请表达理解并提供解决方案："
        )
    elif sentiment == "混合":
        s += sgl.user(
            f"用户反馈情绪复杂（{issue_type}）：{text}\n"
            f"请先肯定正面反馈，再针对不满之处提供帮助："
        )
    else:
        s += sgl.user(
            f"用户反馈（{issue_type}，{sentiment}情绪）：{text}\n"
            f"请给出专业友好的回应："
        )

    s += sgl.assistant(sgl.gen("response", max_tokens=200))

    return {
        "sentiment": sentiment,
        "issue_type": issue_type,
    }


def main():
    # ── 演示1：原子函数独立使用 ───────────────────────────────
    print("=" * 60)
    print("【演示1：原子函数独立使用——情感分析】")
    print("=" * 60)
    print()

    test_texts = [
        "这个产品真的很好用，超出了我的预期！",
        "等了两周还没收到货，非常失望！",
        "产品还行，但快递太慢了，整体感受一般。",
        "请问这款产品支持哪些操作系统？",
    ]

    print("批量情感分析：\n")
    states = analyze_sentiment.run_batch(
        [{"text": t} for t in test_texts],
        num_threads=4,
    )
    for text, state in zip(test_texts, states):
        print(
            f"  [{state['sentiment']:2s}] {text[:32]}..."
        )

    # ── 演示2：组合函数——情感驱动回应 ────────────────────────
    print()
    print("=" * 60)
    print("【演示2：组合函数——情感驱动的自适应回应】")
    print("=" * 60)
    print()

    for text in test_texts[:3]:
        start = time.time()
        state = generate_response_by_sentiment.run(text=text)
        elapsed = (time.time() - start) * 1000

        # 通过子函数调用的状态获取情感结果
        sent_state = analyze_sentiment.run(text=text)
        sentiment = sent_state["sentiment"]

        print(f"用户反馈：{text[:35]}...")
        print(f"  情感判断：{sentiment}")
        print(f"  回应策略：{'先共情后解决' if sentiment=='消极' else '标准专业回应'}")
        print(f"  回应内容：{state['response'][:70]}...")
        print(f"  总耗时：{elapsed:.0f}ms")
        print()

    # ── 演示3：多原子函数组合的完整工作流 ────────────────────
    print("=" * 60)
    print("【演示3：情感+类型双分析的完整客服工作流】")
    print("=" * 60)
    print()

    complex_cases = [
        "快递丢失了，我要求赔偿！产品质量也不行！",
        "客服响应很快，但是退款流程太繁琐了。",
        "请问退货地址是哪里？",
    ]

    for text in complex_cases:
        start = time.time()
        state = full_customer_service.run(text=text)
        elapsed = (time.time() - start) * 1000

        meta = state.ret_value
        print(f"用户反馈：{text[:40]}...")
        print(f"  情感：{meta['sentiment']}  "
              f"问题类型：{meta['issue_type']}")
        print(f"  回应：{state['response'][:80]}...")
        print(f"  耗时：{elapsed:.0f}ms")
        print()

    # ── 函数组合的工程价值说明 ────────────────────────────────
    print("=" * 60)
    print("【函数组合的工程价值】")
    print("=" * 60)
    print()
    print("可测试性：analyze_sentiment可独立单元测试，")
    print("  无需运行完整的客服工作流。")
    print()
    print("可复用性：analyze_sentiment在多个工作流中共享，")
    print("  修改一处，所有调用方自动受益。")
    print()
    print("可维护性：替换某个步骤的实现只需修改对应函数，")
    print("  组合逻辑无需改动。")
    print()
    print("数据流清晰：函数间通过Python变量传递，")
    print("  每个步骤的输入输出均可在调试时直接检查。")


if __name__ == "__main__":
    main()
