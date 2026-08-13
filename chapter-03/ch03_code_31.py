"""
第3章 代码示例 31
主题：多路并发流式生成
场景：使用fork创建多个并发流式分支，
      三个视角同时生成，各分支独立通过text_iter()消费，
      总耗时约等于单个最慢分支的耗时

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

说明：
    多路并发流与顺序流的区别：
      顺序流：视角A全部完成 → 视角B开始 → 视角C开始
              总耗时 = A + B + C
      并发流：三个视角同时开始，同时推进
              总耗时 ≈ max(A, B, C)
"""

import time
from typing import Dict, List

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def multi_stream_generation(s, text: str):
    """
    同时从三个角度流式生成分析，展示并发流式的实现模式。
    三个fork分支并发执行，总耗时≈单个分支耗时。
    """
    s += sgl.system("你是一位多维度分析专家。")
    s += sgl.user(f"分析文本：{text}")

    # fork：三个分支并发生成，共享公共前缀KV Cache
    forks = s.fork(3)
    perspectives = ["优势分析", "劣势分析", "改进建议"]

    for i, f in enumerate(forks):
        f += sgl.user(f"请进行{perspectives[i]}（流式输出）：")
        # 每个分支独立流式生成
        f += sgl.assistant(
            sgl.gen("analysis", max_tokens=150,
                    stream=True, temperature=0.3)
        )

    # 各分支结果通过forks[i].text_iter()独立消费
    # 可以根据完成顺序动态展示，不必等待所有分支完成
    results = {}
    for i in range(3):
        # 在实际UI中，可以为每个视角开辟独立的显示区域
        full_text = ""
        for chunk in forks[i].text_iter("analysis"):
            full_text += chunk
        results[perspectives[i]] = full_text

    return results


@sgl.function
def sequential_stream_generation(s, text: str):
    """
    顺序流式生成（对照组）：三个视角依次生成。
    总耗时 = 三个视角耗时之和。
    """
    s += sgl.system("你是一位多维度分析专家。")
    s += sgl.user(f"分析文本：{text}")

    perspectives = ["优势分析", "劣势分析", "改进建议"]
    results = {}

    for perspective in perspectives:
        s += sgl.user(f"请进行{perspective}：")
        s += sgl.assistant(
            sgl.gen(
                f"analysis_{perspective}",
                max_tokens=150,
                stream=True,
                temperature=0.3,
            )
        )
        # 顺序消费：当前视角完成后才开始下一个
        full_text = ""
        for chunk in s.text_iter(f"analysis_{perspective}"):
            full_text += chunk
        results[perspective] = full_text

    return results


@sgl.function
def dynamic_multi_stream(
    s, text: str, perspectives: List[str]
):
    """
    动态数量的并发流式生成：
    分支数由perspectives参数决定，运行时灵活配置。
    """
    s += sgl.system("你是一位专业分析师。")
    s += sgl.user(f"分析对象：{text}")

    n = len(perspectives)
    forks = s.fork(n)

    for i, f in enumerate(forks):
        f += sgl.user(
            f"从【{perspectives[i]}】角度分析（80字以内）："
        )
        f += sgl.assistant(
            sgl.gen(
                "analysis",
                max_tokens=120,
                stream=True,
                temperature=0.3,
            )
        )

    # 收集所有分支结果
    results = {}
    for i, perspective in enumerate(perspectives):
        chunks = []
        for chunk in forks[i].text_iter("analysis"):
            chunks.append(chunk)
        results[perspective] = "".join(chunks)

    # 主分支聚合所有视角，生成综合建议
    s += sgl.user("综合以上各视角，给出核心建议：")
    combined = "\n".join([
        f"{p}：{results[p][:60]}..."
        for p in perspectives
    ])
    s += sgl.assistant(
        combined + "\n\n核心建议："
        + sgl.gen("synthesis", max_tokens=120, stream=True)
    )

    return results


def main():
    sample_text = (
        "我们的AI推理服务在过去一个月内QPS从50增长到了200，"
        "但P99延迟也从800ms上升到了2.1秒，"
        "用户投诉增加了30%。"
    )

    # ── 演示1：并发 vs 顺序性能对比 ──────────────────────────
    print("=" * 60)
    print("【演示1：并发流式 vs 顺序流式性能对比】")
    print("=" * 60)
    print(f"\n分析文本：{sample_text[:45]}...\n")

    print("顺序流式（三视角串行）...")
    start = time.time()
    seq_state = sequential_stream_generation.run(
        text=sample_text, stream=True
    )
    seq_elapsed = (time.time() - start) * 1000
    print(f"  顺序耗时：{seq_elapsed:.0f}ms")

    print("\n并发流式（三视角并发）...")
    start = time.time()
    par_state = multi_stream_generation.run(
        text=sample_text, stream=True
    )
    par_elapsed = (time.time() - start) * 1000
    print(f"  并发耗时：{par_elapsed:.0f}ms")

    if par_elapsed > 0:
        speedup = seq_elapsed / par_elapsed
        print(f"\n  加速比：{speedup:.2f}x")
        print(f"  （理论最大：3x，实际受调度开销影响）")

    # ── 演示2：展示各分支流式结果 ─────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：并发流式各分支结果展示】")
    print("=" * 60)
    print()

    results = par_state.ret_value
    perspectives = ["优势分析", "劣势分析", "改进建议"]

    for perspective in perspectives:
        text = results.get(perspective, "")
        print(f"[{perspective}]")
        print(f"  {text[:80]}...")
        print()

    # ── 演示3：动态数量并发流式 ───────────────────────────────
    print("=" * 60)
    print("【演示3：动态分支数量并发流式（5个视角）】")
    print("=" * 60)
    print()

    custom_perspectives = [
        "性能瓶颈", "成本分析", "用户影响", "技术风险", "优化路径"
    ]
    print(f"并发分析视角：{custom_perspectives}\n")

    start = time.time()
    state3 = dynamic_multi_stream.run(
        text=sample_text,
        perspectives=custom_perspectives,
        stream=True,
    )
    elapsed3 = (time.time() - start) * 1000

    results3 = state3.ret_value
    print(f"并发{len(custom_perspectives)}个视角，"
          f"耗时：{elapsed3:.0f}ms\n")

    for p, analysis in results3.items():
        print(f"  [{p}] {analysis[:55]}...")

    # 流式消费综合建议
    print()
    print("综合建议（流式）：", end="", flush=True)
    for chunk in state3.text_iter("synthesis"):
        print(chunk, end="", flush=True)
    print()

    # ── 多路流式原理说明 ──────────────────────────────────────
    print()
    print("=" * 60)
    print("【多路并发流式原理说明】")
    print("=" * 60)
    print()
    print("fork并发 + stream=True 的组合效果：")
    print("  1. fork点之前的公共前缀KV Cache自动共享")
    print("  2. 各分支在独立线程中并发执行流式生成")
    print("  3. 通过forks[i].text_iter()独立消费各分支")
    print("  4. 消费顺序可自由控制（如按完成先后顺序）")
    print()
    print("实际UI应用场景：")
    print("  为每个视角开辟独立的显示区域")
    print("  哪个分支先完成就先开始显示")
    print("  用户同时看到多个分析视角逐步填充")


if __name__ == "__main__":
    main()
