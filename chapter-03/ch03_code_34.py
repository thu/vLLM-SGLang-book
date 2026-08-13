"""
第3章 代码示例 34
主题：并行扇出-扇入工作流——多标准内容审核
场景：三个审核维度并发执行，各分支独立持有状态，
      所有分支完成后汇聚到综合评定步骤，
      演示并行执行的正确性保证与性能优势

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
def multi_criteria_review(s, content: str):
    """
    多标准并行内容审核：同时从准确性、完整性、
    可读性三个维度独立审核，最后综合评定。

    并行执行的正确性保证：
      - 三个fork分支持有独立的ProgramState，互不干扰
      - 每个分支的gen()调用在独立线程中执行
      - 聚合步骤（综合评定）在所有分支完成后才执行
    """
    s += sgl.system("你是一位专业的内容质量审核员。")
    s += sgl.user(f"待审核内容：\n{content}")

    criteria = ["准确性", "完整性", "可读性"]
    forks = s.fork(3)

    for i, f in enumerate(forks):
        f += sgl.user(
            f"请从{criteria[i]}角度给出评分（1-10）和简短说明："
        )
        f += sgl.assistant(
            sgl.gen(f"review", max_tokens=100,
                    temperature=0.2)
        )

    # 收集各维度评审结果后进行综合
    reviews_text = "\n".join([
        f"{criteria[i]}审核：{forks[i]['review']}"
        for i in range(3)
    ])
    s += sgl.user(
        f"基于以下三个维度的审核结果，"
        f"给出综合评定（通过/需修改/拒绝）：\n{reviews_text}"
    )
    s += sgl.assistant(
        sgl.gen("final_verdict",
                choices=["通过", "需修改", "拒绝"])
    )
    s += sgl.assistant(
        "\n综合说明："
        + sgl.gen("summary", max_tokens=150)
    )

    return {
        "reviews": {
            criteria[i]: forks[i]["review"] for i in range(3)
        },
        "verdict": s["final_verdict"],
        "summary": s["summary"],
    }


@sgl.function
def sequential_criteria_review(s, content: str):
    """
    顺序执行三维度审核（与multi_criteria_review对照）。
    总耗时 = 三个维度耗时之和。
    """
    s += sgl.system("你是一位专业的内容质量审核员。")
    s += sgl.user(f"待审核内容：\n{content}")

    criteria = ["准确性", "完整性", "可读性"]
    reviews = {}

    for criterion in criteria:
        s += sgl.user(
            f"请从{criterion}角度给出评分（1-10）和简短说明："
        )
        s += sgl.assistant(
            sgl.gen(f"review_{criterion}",
                    max_tokens=100, temperature=0.2)
        )
        reviews[criterion] = s[f"review_{criterion}"]

    reviews_text = "\n".join([
        f"{c}审核：{reviews[c]}" for c in criteria
    ])
    s += sgl.user(
        f"给出综合评定（通过/需修改/拒绝）：\n{reviews_text}"
    )
    s += sgl.assistant(
        sgl.gen("final_verdict",
                choices=["通过", "需修改", "拒绝"])
    )


@sgl.function
def extended_review(
    s, content: str, criteria: List[str]
):
    """
    可扩展的动态维度审核：
    维度数量由criteria参数决定，不固定为3个。
    """
    s += sgl.system("你是一位专业的内容审核专家。")
    s += sgl.user(f"待审核内容：\n{content}")

    n = len(criteria)
    forks = s.fork(n)

    for i, f in enumerate(forks):
        f += sgl.user(
            f"请从【{criteria[i]}】角度审核（评分1-10，50字说明）："
        )
        f += sgl.assistant(
            sgl.gen("review", max_tokens=80, temperature=0.2)
        )

    # 聚合所有维度
    reviews_text = "\n".join([
        f"{criteria[i]}：{forks[i]['review']}"
        for i in range(n)
    ])
    s += sgl.user(
        f"基于{n}个维度的审核结果，给出综合评定："
    )
    s += sgl.assistant(
        reviews_text + "\n\n综合评定："
        + sgl.gen(
            "verdict",
            choices=["优秀", "良好", "需改进", "不合格"]
        )
    )

    return {
        criteria[i]: forks[i]["review"]
        for i in range(n)
    }


def main():
    test_content = (
        "本报告分析了2024年AI推理框架市场的发展趋势。"
        "SGLang、vLLM等开源框架在性能和易用性方面取得了显著进展，"
        "推动了大语言模型的广泛部署。"
        "报告认为，未来两年内推理成本将降低60-80%，"
        "这将使百亿参数模型的部署成为主流。"
    )

    # ── 演示1：并行 vs 顺序性能对比 ──────────────────────────
    print("=" * 60)
    print("【演示1：并行 vs 顺序审核性能对比】")
    print("=" * 60)
    print(f"\n待审核内容：{test_content[:50]}...\n")

    print("顺序执行（三维度串行）...")
    start = time.time()
    sequential_criteria_review.run(content=test_content)
    seq_elapsed = (time.time() - start) * 1000
    print(f"  顺序耗时：{seq_elapsed:.0f}ms")

    print("\n并行执行（三维度并发）...")
    start = time.time()
    par_state = multi_criteria_review.run(content=test_content)
    par_elapsed = (time.time() - start) * 1000
    print(f"  并行耗时：{par_elapsed:.0f}ms")

    if par_elapsed > 0:
        speedup = seq_elapsed / par_elapsed
        print(f"\n  加速比：{speedup:.2f}x")

    # ── 演示2：审核结果详情 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：各维度审核结果与综合判定】")
    print("=" * 60)
    print()

    result = par_state.ret_value
    print("各维度审核结果：")
    for criterion, review in result["reviews"].items():
        print(f"\n  [{criterion}]")
        print(f"  {review[:80]}...")

    print(f"\n综合判定：{result['verdict']}")
    print(f"综合说明：{result['summary'][:100]}...")

    # ── 演示3：批量内容审核 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：run_batch批量内容审核】")
    print("=" * 60)
    print()

    batch_contents = [
        "机器学习是人工智能的子领域，专注于让计算机从数据中学习。"
        "监督学习、无监督学习和强化学习是三种主要范式。",

        "Python是最流行的编程语言。它非常好用。每个人都应该学Python。",

        "本研究采用双盲随机对照试验设计，纳入500名受试者，"
        "通过12周干预评估新疗法的安全性和有效性，"
        "主要终点为治愈率，次要终点包括副作用发生率和生活质量评分。",
    ]

    print(f"并发审核{len(batch_contents)}篇内容...\n")
    start = time.time()
    states = multi_criteria_review.run_batch(
        [{"content": c} for c in batch_contents],
        num_threads=3,
    )
    elapsed = (time.time() - start) * 1000

    print(f"批量耗时：{elapsed:.0f}ms\n")
    for i, (content, st) in enumerate(
        zip(batch_contents, states)
    ):
        r = st.ret_value
        print(
            f"  内容{i+1}：{content[:30]}..."
            f" → 判定：{r['verdict']}"
        )

    # ── 演示4：动态维度扩展审核 ───────────────────────────────
    print()
    print("=" * 60)
    print("【演示4：动态维度扩展审核（5个维度）】")
    print("=" * 60)
    print()

    extended_criteria = [
        "事实准确性", "逻辑严密性", "表达清晰度",
        "数据支撑", "结论合理性"
    ]
    print(f"审核维度：{extended_criteria}\n")

    start = time.time()
    state4 = extended_review.run(
        content=test_content,
        criteria=extended_criteria,
    )
    elapsed4 = (time.time() - start) * 1000

    result4 = state4.ret_value
    print(f"并发{len(extended_criteria)}个维度，耗时：{elapsed4:.0f}ms")
    print(f"综合评定：{state4['verdict']}\n")
    for criterion, review in result4.items():
        print(f"  [{criterion}] {review[:50]}...")

    print()
    print("=" * 60)
    print("【并行审核的工程价值】")
    print("=" * 60)
    print()
    print("正确性：各分支独立ProgramState，无共享可变状态")
    print("性能  ：总耗时≈最慢维度耗时，而非所有维度之和")
    print("可扩展：新增维度只需在criteria列表中添加")
    print("隔离性：某维度生成失败不影响其他维度结果")


if __name__ == "__main__":
    main()
