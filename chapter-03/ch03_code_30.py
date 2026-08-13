"""
第3章 代码示例 30
主题：混合流式策略——根据内容类型选择最优展示方式
场景：代码生成任务中，函数签名批量输出（结构性），
      文档字符串和函数实现流式输出（叙述性），
      在正确性与用户体验之间取得最优平衡

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

策略原则：
    stream=True ：叙述性内容，边生成边显示，降低感知延迟
    stream=False：结构性内容，完整后一次性显示，避免残缺误导
"""

import time

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def generate_code_with_mixed_streaming(s, task: str):
    """
    混合流式策略：对不同内容部分采用不同的流式设置。
    叙述性内容流式输出，结构性内容批量输出。
    """
    s += sgl.system("你是一位Python编程专家。")
    s += sgl.user(f"请为以下任务生成Python代码：\n{task}")

    # 文档字符串：叙述性，适合流式（用户边读边理解）
    s += sgl.assistant("```python\ndef solution(")
    s += sgl.assistant(
        sgl.gen("signature", max_tokens=60,
                stop="):", stream=False)  # 签名需要完整
    )
    s += sgl.assistant("):\n    \"\"\"")
    s += sgl.assistant(
        sgl.gen("docstring", max_tokens=100,
                stop='"""', stream=True)  # 文档适合流式
    )
    s += sgl.assistant('"""\n')

    # 函数实现：逐行流式输出
    s += sgl.assistant(
        sgl.gen("implementation", max_tokens=300,
                stop="```", stream=True)
    )
    s += sgl.assistant("```")


@sgl.function
def generate_analysis_mixed(s, topic: str):
    """
    分析报告的混合流式策略：
      结论句（stream=False）：需完整才有意义
      详细分析（stream=True）：长文适合流式
      建议摘要（stream=False）：结构性内容一次性展示
    """
    s += sgl.system("你是一位专业的技术分析师。")
    s += sgl.user(f"请对【{topic}】进行简要技术分析。")

    # 核心结论：非流式，确保完整呈现后再显示
    s += sgl.assistant(
        sgl.gen(
            "conclusion_sentence",
            max_tokens=40,
            stop="。",
            stream=False,
        )
    )
    s += sgl.assistant("。\n\n")

    # 详细分析：流式，长文逐步呈现
    s += sgl.assistant(
        sgl.gen(
            "analysis_body",
            max_tokens=200,
            stop="\n\n",
            stream=True,
        )
    )

    # 行动建议：非流式，完整列表一次性展示
    s += sgl.assistant("\n\n建议：")
    s += sgl.assistant(
        sgl.gen(
            "recommendations",
            max_tokens=80,
            stream=False,
        )
    )


def main():
    # ── 演示1：代码生成混合流式——逐部分消费 ─────────────────
    print("=" * 60)
    print("【演示1：代码生成——混合流式策略】")
    print("=" * 60)
    print()

    task = "实现一个二分查找函数，接受有序列表和目标值，返回索引或-1"
    print(f"任务：{task}\n")
    print("-" * 40)

    start = time.time()
    state = generate_code_with_mixed_streaming.run(
        task=task, stream=True
    )

    # 函数签名：stream=False，直接通过键名访问完整值
    # 用户在签名完整生成后才看到，不会看到不完整的语法
    signature = state["signature"]
    print(f"[批量输出] 函数签名（一次性完整显示）：")
    print(f"  def solution({signature}):")
    print()

    # 文档字符串：stream=True，通过text_iter()流式消费
    print('[流式输出] 文档字符串（逐步显示）：')
    print('  """', end="", flush=True)
    for chunk in state.text_iter("docstring"):
        print(chunk, end="", flush=True)
    print('"""')
    print()

    # 函数实现：stream=True，通过text_iter()流式消费
    print("[流式输出] 函数实现（逐行显示）：")
    for chunk in state.text_iter("implementation"):
        print(chunk, end="", flush=True)

    elapsed = (time.time() - start) * 1000
    print(f"\n\n总耗时：{elapsed:.0f}ms")

    # ── 演示2：分析报告混合流式 ───────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：分析报告——结论批量，正文流式】")
    print("=" * 60)
    print()

    topic = "大语言模型推理加速技术"
    print(f"分析主题：{topic}\n")
    print("-" * 40)

    start2 = time.time()
    state2 = generate_analysis_mixed.run(
        topic=topic, stream=True
    )

    # 结论句：批量，完整后一次性展示
    conclusion = state2["conclusion_sentence"]
    print(f"[批量] 核心结论：{conclusion}。")
    print()

    # 详细分析：流式消费
    print("[流式] 详细分析：", end="", flush=True)
    for chunk in state2.text_iter("analysis_body"):
        print(chunk, end="", flush=True)
    print()

    # 建议摘要：批量，结构性内容完整展示
    recommendations = state2["recommendations"]
    print(f"\n[批量] 行动建议：{recommendations}")

    elapsed2 = (time.time() - start2) * 1000
    print(f"\n总耗时：{elapsed2:.0f}ms")

    # ── 演示3：多任务批量并发代码生成 ────────────────────────
    print()
    print("=" * 60)
    print("【演示3：run_batch批量并发代码生成】")
    print("=" * 60)
    print()

    tasks = [
        "实现冒泡排序算法",
        "实现斐波那契数列（递归版）",
        "实现字符串反转函数",
    ]

    print(f"并发生成{len(tasks)}个代码任务...\n")
    start3 = time.time()
    states = generate_code_with_mixed_streaming.run_batch(
        [{"task": t} for t in tasks],
        num_threads=3,
    )
    elapsed3 = (time.time() - start3) * 1000

    for task, st in zip(tasks, states):
        sig = st["signature"]
        impl_preview = "".join(
            chunk for chunk in st.text_iter("implementation")
        )[:40]
        print(f"  任务：{task}")
        print(f"  签名：def solution({sig}):")
        print(f"  实现：{impl_preview}...")
        print()

    print(f"批量总耗时：{elapsed3:.0f}ms")
    print(f"平均每个：{elapsed3/len(tasks):.0f}ms")

    # ── 策略选择指南 ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("【混合流式策略选择指南】")
    print("=" * 60)
    print()
    print("stream=True（流式）：")
    print("  代码实现主体、文档字符串、叙述性分析、对话回复")
    print()
    print("stream=False（批量）：")
    print("  函数签名、JSON结构、数值结论、标题行、摘要句")
    print()
    print("核心原则：残缺会误导用户的内容 → 批量输出")
    print("         逐步显示有助于理解的内容 → 流式输出")


if __name__ == "__main__":
    main()
