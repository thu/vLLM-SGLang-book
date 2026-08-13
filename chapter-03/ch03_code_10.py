"""
第3章 代码示例 10
主题：fork/join并行执行原语——多视角分析
场景：使用s.fork(3)创建并行分支，从三个视角
      同时分析同一份文本，演示扇出-扇入工作流模式

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

说明：
    s.fork(n)在fork点创建n个独立的ProgramState副本，
    每个副本在独立线程中并发执行各自的gen()调用，
    fork点之前的公共上下文KV Cache由所有分支自动共享。
    总耗时约等于单个最慢分支的耗时，而非n倍。
"""

import time
from typing import List

import sglang as sgl

# 连接本地SGLang推理服务
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def multi_perspective_analysis(s, text: str):
    s += sgl.user(f"待分析文本：{text}")

    # fork创建3个独立状态副本，在独立线程中并行执行
    # 总耗时约等于单个视角的分析时间，而非三者之和
    forks = s.fork(3)
    perspectives = ["技术可行性", "商业价值", "潜在风险"]

    for i, f in enumerate(forks):
        f += sgl.user(
            f"请从{perspectives[i]}角度分析上述文本："
        )
        f += sgl.assistant(
            sgl.gen(f"analysis", max_tokens=150)
        )

    # 扇入：所有并行分支完成后聚合结果
    s += sgl.user("综合以上三个视角，给出整体评估：")
    analyses = "\n".join([
        f"{perspectives[i]}：{forks[i]['analysis']}"
        for i in range(3)
    ])
    s += sgl.assistant(
        analyses + "\n整体评估：" +
        sgl.gen("overall", max_tokens=200)
    )


@sgl.function
def sequential_analysis(s, text: str):
    """
    顺序执行三视角分析，作为并行版本的对照组。
    总耗时 = 三个视角耗时之和。
    """
    s += sgl.user(f"待分析文本：{text}")
    perspectives = ["技术可行性", "商业价值", "潜在风险"]

    for perspective in perspectives:
        s += sgl.user(
            f"请从{perspective}角度分析上述文本："
        )
        s += sgl.assistant(
            sgl.gen(
                f"analysis_{perspective}",
                max_tokens=150,
            )
        )

    s += sgl.user("综合以上三个视角，给出整体评估：")
    s += sgl.assistant(
        sgl.gen("overall", max_tokens=200)
    )


@sgl.function
def dynamic_fork_analysis(
    s, text: str, perspectives: List[str]
):
    """
    动态数量的fork：fork数量由perspectives列表长度决定，
    不需要在编写代码时固定分支数。
    """
    s += sgl.user(f"待分析文本：{text}")

    n = len(perspectives)
    forks = s.fork(n)

    for i, f in enumerate(forks):
        f += sgl.user(
            f"请从【{perspectives[i]}】角度分析（80字以内）："
        )
        f += sgl.assistant(
            sgl.gen("analysis", max_tokens=120)
        )

    # 聚合所有分支结果
    s += sgl.user("综合以上分析，给出综合建议：")
    summary_input = "\n".join([
        f"{perspectives[i]}：{forks[i]['analysis']}"
        for i in range(n)
    ])
    s += sgl.assistant(
        summary_input + "\n\n综合建议："
        + sgl.gen("recommendation", max_tokens=150)
    )

    return {
        p: forks[i]["analysis"]
        for i, p in enumerate(perspectives)
    }


def main():
    proposal = (
        "开发一套基于大语言模型的企业代码审查工具，"
        "能够自动检测安全漏洞、性能问题和代码风格违规，"
        "并提供修复建议，目标用户为500人规模的研发团队。"
    )

    # ── 演示1：并行 vs 顺序性能对比 ──────────────────────────
    print("=" * 60)
    print("【演示1：并行 vs 顺序执行性能对比】")
    print("=" * 60)
    print(f"\n分析方案：{proposal[:40]}...\n")

    print("顺序执行（三视角串行）...")
    start = time.time()
    sequential_analysis.run(text=proposal)
    seq_elapsed = (time.time() - start) * 1000
    print(f"  顺序耗时：{seq_elapsed:.0f}ms")

    print("\n并行执行（三视角并发）...")
    start = time.time()
    par_state = multi_perspective_analysis.run(text=proposal)
    par_elapsed = (time.time() - start) * 1000
    print(f"  并行耗时：{par_elapsed:.0f}ms")

    if par_elapsed > 0:
        speedup = seq_elapsed / par_elapsed
        print(f"\n加速比：{speedup:.2f}x")
        print(f"（理论最大：3x，实际受调度和网络开销影响）")

    # ── 演示2：展示各分支分析结果 ─────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：各视角分析结果展示】")
    print("=" * 60)

    # 重新执行以获取结果（或复用上面的par_state）
    state = multi_perspective_analysis.run(text=proposal)
    perspectives = ["技术可行性", "商业价值", "潜在风险"]

    # fork生成的变量名均为"analysis"，通过forks[i]索引访问
    # 此处通过重新fork方式验证结果（演示访问模式）
    print()
    print(f"整体评估：")
    print(f"  {state['overall'][:120]}...")

    # ── 演示3：动态fork数量 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：动态fork数量（4个视角）】")
    print("=" * 60)

    custom_perspectives = [
        "用户体验", "安全合规", "运维成本", "竞争优势"
    ]
    print(f"\n分析视角：{custom_perspectives}")
    print(f"方案：{proposal[:40]}...\n")

    start = time.time()
    state3 = dynamic_fork_analysis.run(
        text=proposal,
        perspectives=custom_perspectives,
    )
    elapsed3 = (time.time() - start) * 1000

    result = state3.ret_value
    print(f"并发执行{len(custom_perspectives)}个视角，"
          f"耗时：{elapsed3:.0f}ms\n")

    for perspective, analysis in result.items():
        print(f"  {perspective}：{analysis[:60]}...")

    print(f"\n综合建议：{state3['recommendation'][:100]}...")

    # ── fork机制原理说明 ──────────────────────────────────────
    print()
    print("=" * 60)
    print("【fork机制原理说明】")
    print("=" * 60)
    print()
    print("s.fork(n) 的工作机制：")
    print("  1. 在fork点创建n个独立的ProgramState副本")
    print("  2. 每个副本共享fork点之前的完整上下文")
    print("     （公共前缀KV Cache由RadixAttention自动共享）")
    print("  3. 各副本在独立线程中并发执行gen()调用")
    print("  4. 通过forks[i]['变量名']访问各分支结果")
    print()
    print("并发安全保证：")
    print("  每个分支持有独立ProgramState，互不干扰，")
    print("  某分支失败不影响其他分支的结果。")
    print()
    print("注意：fork数量受GPU显存限制，")
    print("  建议单次不超过8-16个分支。")


if __name__ == "__main__":
    main()
