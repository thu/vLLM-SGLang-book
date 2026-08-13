"""
第3章 代码示例 02
主题：SGLang声明式多步推理实现
场景：与 ch03_code_01.py 形成对比，用SGLang重写
      相同的问题分解-求解-综合三步推理流程，
      展示regex约束、状态自动管理、上下文复用等优势

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import json
import time

import sglang as sgl

# 连接本地SGLang推理服务
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def solve_question_sglang(s, question: str):
    s += f"将以下问题分解为3-5个子问题：\n问题：{question}\n"
    s += "以JSON格式输出子问题列表：\n"
    # regex约束确保输出为合法JSON数组，无需事后验证
    s += sgl.gen(
        "sub_questions",
        max_tokens=200,
        regex=r'\{"sub_questions": \["[^"]*"(, "[^"]*")*\]\}'
    )

    sub_qs = json.loads(s["sub_questions"])["sub_questions"]
    sub_answers = []
    for i, sub_q in enumerate(sub_qs):
        s += f"\n请回答：{sub_q}\n简洁回答："
        # stop参数控制生成边界，结果自动以键名存储
        s += sgl.gen(f"answer_{i}", max_tokens=100, stop="\n\n")
        sub_answers.append(s[f"answer_{i}"])

    # s自动携带完整上下文，综合阶段无需重新传入历史内容
    s += f"\n综合以上分析，回答原问题：{question}\n最终答案："
    s += sgl.gen("final_answer", max_tokens=300)


def _print_comparison():
    """打印与传统方式的对比说明。"""
    print()
    print("=" * 60)
    print("【SGLang vs 传统方式对比】")
    print("=" * 60)
    print()
    print("✓ regex约束：输出格式在生成阶段强制保证，")
    print("  json.loads()直接调用，无需try/except包裹。")
    print()
    print("✓ 状态自动管理：s对象自动积累对话历史，")
    print("  综合阶段无需重新拼接前序内容。")
    print()
    print("✓ 上下文复用：RadixAttention自动识别并复用")
    print("  公共前缀的KV Cache，减少重复Prefill计算。")
    print()
    print("✓ 可读性：代码结构直接映射业务逻辑，")
    print("  每行描述「要什么」而非「怎么做」。")


def main():
    question = (
        "为什么现代大语言模型在长链推理任务上仍然面临挑战，"
        "有哪些主要的技术路径可以改善这一问题？"
    )

    print("=" * 60)
    print("【SGLang声明式多步推理演示】")
    print("=" * 60)
    print(f"\n问题：{question}\n")
    print("正在执行多步推理...\n")

    start = time.time()
    state = solve_question_sglang.run(question=question)
    elapsed = (time.time() - start) * 1000

    # 解析子问题列表
    sub_qs = json.loads(
        state["sub_questions"]
    )["sub_questions"]

    print(f"子问题分解（共{len(sub_qs)}个）：")
    for i, q in enumerate(sub_qs):
        answer = state.get(f"answer_{i}", "")
        print(f"  {i+1}. {q}")
        print(f"     → {answer[:60]}...")

    print(f"\n最终答案：")
    print(state["final_answer"])
    print(f"\n总耗时：{elapsed:.0f}ms")

    _print_comparison()

    # 演示run_batch批量并发执行
    print()
    print("=" * 60)
    print("【run_batch批量并发演示】")
    print("=" * 60)

    questions = [
        "什么是Transformer架构的核心创新？",
        "KV Cache的作用是什么？",
    ]

    print(f"\n并发执行{len(questions)}个问题...\n")
    start = time.time()
    states = solve_question_sglang.run_batch(
        [{"question": q} for q in questions],
        num_threads=2,
    )
    elapsed = (time.time() - start) * 1000

    for q, st in zip(questions, states):
        print(f"问题：{q}")
        print(f"答案：{st['final_answer'][:80]}...")
        print()

    print(f"批量总耗时：{elapsed:.0f}ms")
    print(f"（顺序执行预计约 {elapsed * len(questions):.0f}ms）")


if __name__ == "__main__":
    main()
