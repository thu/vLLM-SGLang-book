"""
第3章 代码示例 35
主题：迭代优化工作流
场景：循环改进草稿直到质量评分达标或达到最大轮次，
      演示双重终止条件、regex评分约束和历史积累机制

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def iterative_refinement(
    s, initial_draft: str, quality_threshold: int = 8,
    max_iterations: int = 4,
):
    """
    迭代优化工作流：反复改进草稿直到质量达标或达到最大轮次。

    设计要点：
      1. 评分使用regex约束，确保输出为合法整数，无需解析验证
      2. 每轮草稿以迭代编号命名，所有版本可追溯回溯
      3. quality_threshold和max_iterations双重终止条件，
         防止无限循环，同时保证最低质量标准
      4. s对象自动积累改进历史，下一轮可参考前序版本
    """
    current = initial_draft
    final_score = 0

    for iteration in range(max_iterations):
        s += sgl.user(
            f"第{iteration+1}轮草稿：\n{current}\n\n"
            f"请评估质量（1-10分），只输出数字："
        )
        # regex约束：确保评分为1-10的合法整数
        s += sgl.assistant(
            sgl.gen(f"score_{iteration}", max_tokens=3,
                    regex=r"([1-9]|10)", temperature=0.0)
        )
        score = int(s[f"score_{iteration}"])
        final_score = score

        if score >= quality_threshold:
            # 质量达标，提前终止，避免过度修改
            break

        # 质量不足：生成具体改进建议，指导下一轮修改
        s += sgl.user(
            f"当前得分{score}分，低于目标{quality_threshold}分。"
            f"请指出最重要的改进方向（50字以内）："
        )
        s += sgl.assistant(
            sgl.gen(f"feedback_{iteration}",
                    max_tokens=80, stop="\n\n")
        )

        # 基于反馈生成改进版本
        s += sgl.user(
            f"请根据以上反馈改进草稿，"
            f"保留原稿优点同时修正指出的问题："
        )
        s += sgl.assistant(
            sgl.gen(f"draft_{iteration}",
                    max_tokens=400, temperature=0.4)
        )
        current = s[f"draft_{iteration}"]

    return {
        "final_draft": current,
        "final_score": final_score,
        "iterations_used": iteration + 1,
        "converged": final_score >= quality_threshold,
    }


@sgl.function
def iterative_refinement_with_history(
    s,
    initial_draft: str,
    task_description: str = "",
    quality_threshold: int = 8,
    max_iterations: int = 4,
):
    """
    带任务描述和完整历史记录的迭代优化版本。
    在每轮评分时提供任务背景，使评估更有针对性。
    返回包含每轮评分和反馈的完整历史。
    """
    s += sgl.system(
        "你是一位严格的内容质量专家，能够准确评估内容质量"
        "并给出有建设性的改进建议。"
    )

    if task_description:
        s += sgl.user(
            f"任务要求：{task_description}\n\n"
            f"初始草稿：{initial_draft}"
        )
    else:
        s += sgl.user(f"待优化内容：\n{initial_draft}")

    current = initial_draft
    history = []
    final_score = 0

    for iteration in range(max_iterations):
        # 评分
        s += sgl.user(
            f"【第{iteration+1}轮】当前内容：\n{current}\n"
            f"请给出质量评分（1-10），只输出数字："
        )
        s += sgl.assistant(
            sgl.gen(
                f"score_{iteration}",
                max_tokens=3,
                regex=r"([1-9]|10)",
                temperature=0.0,
            )
        )
        score = int(s[f"score_{iteration}"])
        final_score = score
        round_record = {"iteration": iteration + 1, "score": score}

        if score >= quality_threshold:
            round_record["action"] = "converged"
            history.append(round_record)
            break

        # 获取改进建议
        s += sgl.user(
            f"得分{score}，目标{quality_threshold}，"
            f"请指出最重要的改进方向（50字以内）："
        )
        s += sgl.assistant(
            sgl.gen(f"feedback_{iteration}",
                    max_tokens=80, stop="\n\n")
        )
        feedback = s[f"feedback_{iteration}"]

        # 执行改进
        s += sgl.user(
            f"请根据反馈改进内容，保留优点修正问题："
        )
        s += sgl.assistant(
            sgl.gen(f"draft_{iteration}",
                    max_tokens=400, temperature=0.4)
        )
        current = s[f"draft_{iteration}"]

        round_record["feedback"] = feedback
        round_record["action"] = "refined"
        history.append(round_record)

    return {
        "final_draft": current,
        "final_score": final_score,
        "iterations_used": len(history),
        "converged": final_score >= quality_threshold,
        "history": history,
    }


def main():
    initial_draft = (
        "AI很有用。它可以帮助我们做很多事情。"
        "比如翻译和写作。我们应该好好利用AI。"
    )
    task_desc = (
        "撰写一段专业的AI技术介绍，面向企业决策者，"
        "需要提及具体应用场景和商业价值，约150字"
    )

    # ── 演示1：基础迭代优化 ───────────────────────────────────
    print("=" * 60)
    print("【演示1：基础迭代优化工作流】")
    print("=" * 60)
    print(f"\n初始草稿：{initial_draft}")
    print(f"质量阈值：8分，最大轮次：4\n")

    start = time.time()
    state = iterative_refinement.run(
        initial_draft=initial_draft,
        quality_threshold=8,
        max_iterations=4,
    )
    elapsed = (time.time() - start) * 1000

    result = state.ret_value
    print(f"使用轮次：{result['iterations_used']}/4")
    print(f"最终评分：{result['final_score']}/10")
    print(f"是否达标：{'✓' if result['converged'] else '✗'}")
    print(f"总耗时：{elapsed:.0f}ms")
    print(f"\n最终草稿：\n{result['final_draft'][:200]}...")

    # ── 演示2：带任务描述的迭代优化（含完整历史）────────────
    print()
    print("=" * 60)
    print("【演示2：带任务描述的迭代优化（含过程记录）】")
    print("=" * 60)
    print(f"\n任务要求：{task_desc[:40]}...")
    print(f"初始草稿：{initial_draft}\n")

    start = time.time()
    state2 = iterative_refinement_with_history.run(
        initial_draft=initial_draft,
        task_description=task_desc,
        quality_threshold=8,
        max_iterations=4,
    )
    elapsed2 = (time.time() - start) * 1000

    result2 = state2.ret_value
    print("迭代过程记录：")
    for record in result2["history"]:
        action_str = (
            "✓达标，停止" if record["action"] == "converged"
            else "↑继续改进"
        )
        print(
            f"  第{record['iteration']}轮  "
            f"[{action_str}]  评分={record['score']}/10"
        )
        if "feedback" in record:
            print(
                f"    反馈：{record['feedback'][:50]}..."
            )

    print(f"\n最终评分：{result2['final_score']}/10")
    print(f"使用轮次：{result2['iterations_used']}")
    print(f"是否收敛：{'✓' if result2['converged'] else '✗'}")
    print(f"总耗时：{elapsed2:.0f}ms")
    print(f"\n最终草稿（前150字）：")
    print(f"{result2['final_draft'][:150]}...")

    # ── 演示3：不同阈值对比 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：不同质量阈值的迭代行为对比】")
    print("=" * 60)
    print()

    for threshold in [6, 8, 10]:
        start = time.time()
        st = iterative_refinement.run(
            initial_draft=initial_draft,
            quality_threshold=threshold,
            max_iterations=4,
        )
        elapsed_t = (time.time() - start) * 1000
        r = st.ret_value
        print(
            f"  阈值={threshold}：使用{r['iterations_used']}轮，"
            f"最终{r['final_score']}分，"
            f"{'达标✓' if r['converged'] else '未达标✗'}，"
            f"耗时{elapsed_t:.0f}ms"
        )

    print()
    print("=" * 60)
    print("【迭代优化工作流设计要点】")
    print("=" * 60)
    print()
    print("双重终止：质量阈值（主动收敛）+ 最大轮次（安全兜底）")
    print("regex约束：评分为合法整数，直接int()转换无需验证")
    print("历史积累：s对象保存所有轮次，LLM可参考改进轨迹")
    print("反馈具体：不只评分，同时给出改进方向，效率更高")


if __name__ == "__main__":
    main()
