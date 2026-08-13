"""
第3章 代码示例 28
主题：fork并行评估——扇出-扇入工作流模式
场景：对同一方案从四个维度并发评估，
      展示fork的公共前缀KV Cache共享机制，
      以及并行执行带来的显著延迟优势

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
def parallel_evaluation(s, proposal: str):
    """
    对同一提案从多个维度并行评估。
    fork点之前的公共前缀KV Cache被所有分支自动共享，
    并行执行而不增加内存开销。
    """
    # 公共前缀：所有分支共享，KV Cache自动复用
    s += sgl.system("你是一位全面的方案评估专家。")
    s += sgl.user(f"待评估方案：\n{proposal}")

    # fork创建4个独立状态副本，在独立线程中并发执行
    forks = s.fork(4)
    dimensions = ["技术可行性", "商业价值", "实施风险", "成本效益"]

    for i, f in enumerate(forks):
        f += sgl.user(
            f"请从{dimensions[i]}角度评估（100字以内）："
        )
        f += sgl.assistant(
            sgl.gen("eval", max_tokens=150, temperature=0.3)
        )

    # 扇入：聚合所有分支的评估结果
    evals = "\n".join([
        f"{dimensions[i]}：{forks[i]['eval']}"
        for i in range(4)
    ])
    s += sgl.user("综合以上四个维度，给出总体评估与建议：")
    s += sgl.assistant(
        evals + "\n\n综合建议："
        + sgl.gen("conclusion", max_tokens=200)
    )


@sgl.function
def sequential_evaluation(s, proposal: str):
    """
    顺序执行四维度评估（与parallel_evaluation对照）。
    总耗时 = 四个维度耗时之和。
    """
    s += sgl.system("你是一位全面的方案评估专家。")
    s += sgl.user(f"待评估方案：\n{proposal}")

    dimensions = ["技术可行性", "商业价值", "实施风险", "成本效益"]
    evals_text = ""

    for dim in dimensions:
        s += sgl.user(f"请从{dim}角度评估（100字以内）：")
        s += sgl.assistant(
            sgl.gen(f"eval_{dim}", max_tokens=150, temperature=0.3)
        )
        evals_text += f"{dim}：{s[f'eval_{dim}']}\n"

    s += sgl.user("综合以上四个维度，给出总体评估与建议：")
    s += sgl.assistant(
        evals_text + "\n综合建议："
        + sgl.gen("conclusion", max_tokens=200)
    )


@sgl.function
def dynamic_parallel_review(
    s, content: str, perspectives: List[str]
):
    """
    动态数量的并行评审：分支数量由perspectives列表决定，
    支持运行时灵活配置，不需要编写时固定分支数。
    """
    s += sgl.system("你是一位多维度分析专家。")
    s += sgl.user(f"待分析内容：{content}")

    n = len(perspectives)
    forks = s.fork(n)

    for i, f in enumerate(forks):
        f += sgl.user(
            f"请从【{perspectives[i]}】角度分析（80字以内）："
        )
        f += sgl.assistant(
            sgl.gen("analysis", max_tokens=120, temperature=0.3)
        )

    # 聚合所有分支结果
    s += sgl.user("综合各视角，给出整体建议：")
    combined = "\n".join([
        f"{perspectives[i]}：{forks[i]['analysis']}"
        for i in range(n)
    ])
    s += sgl.assistant(
        combined + "\n\n整体建议："
        + sgl.gen("recommendation", max_tokens=150)
    )

    return {p: forks[i]["analysis"] for i, p in enumerate(perspectives)}


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
    print(f"\n方案：{proposal[:45]}...\n")

    print("顺序执行（四维度串行）...")
    start = time.time()
    sequential_evaluation.run(proposal=proposal)
    seq_elapsed = (time.time() - start) * 1000
    print(f"  耗时：{seq_elapsed:.0f}ms")

    print("\n并行执行（四维度并发）...")
    start = time.time()
    par_state = parallel_evaluation.run(proposal=proposal)
    par_elapsed = (time.time() - start) * 1000
    print(f"  耗时：{par_elapsed:.0f}ms")

    if par_elapsed > 0:
        speedup = seq_elapsed / par_elapsed
        print(f"\n  加速比：{speedup:.2f}x")
        print(f"  （理论最大：4x，实际受调度开销影响）")

    # ── 演示2：各分支评估结果 ─────────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：各维度评估结果展示】")
    print("=" * 60)
    print()

    # 重新执行获取详细结果
    state = parallel_evaluation.run(proposal=proposal)
    dimensions = ["技术可行性", "商业价值", "实施风险", "成本效益"]

    # 注意：fork分支的eval通过forks[i]访问，
    # 在函数外部通过conclusion来验证整体输出
    print(f"综合建议：")
    print(f"  {state['conclusion'][:120]}...")

    # ── 演示3：动态分支数量 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：动态fork数量（5个视角）】")
    print("=" * 60)
    print()

    custom_perspectives = [
        "用户体验", "安全合规", "运维成本", "竞争优势", "技术债务"
    ]
    print(f"分析视角：{custom_perspectives}\n")

    start = time.time()
    state3 = dynamic_parallel_review.run(
        content=proposal,
        perspectives=custom_perspectives,
    )
    elapsed3 = (time.time() - start) * 1000

    result = state3.ret_value
    print(f"并发{len(custom_perspectives)}个视角，耗时：{elapsed3:.0f}ms\n")

    for perspective, analysis in result.items():
        print(f"  [{perspective}]")
        print(f"  {analysis[:60]}...")
        print()

    print(f"整体建议：{state3['recommendation'][:100]}...")

    # ── fork机制原理说明 ──────────────────────────────────────
    print()
    print("=" * 60)
    print("【fork并行机制原理说明】")
    print("=" * 60)
    print()
    print("公共前缀KV Cache共享：")
    print("  fork点之前的内容（系统提示词+方案描述）")
    print("  在所有分支间由RadixAttention自动共享，")
    print("  并行不等于内存翻倍。")
    print()
    print("并发安全保证：")
    print("  每个分支持有独立ProgramState，互不干扰，")
    print("  某分支失败不影响其他分支的结果。")
    print()
    print("动态分支数量：")
    print("  s.fork(n)支持运行时确定n，不需要编译期固定。")
    print()
    print("适用上限：")
    print("  分支数受GPU显存限制，建议单次不超过8-16个。")


if __name__ == "__main__":
    main()
