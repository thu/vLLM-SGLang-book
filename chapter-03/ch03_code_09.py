"""
第3章 代码示例 09
主题：程序化提示工程范式——单元测试实践
场景：对SGLang函数编写结构化单元测试，
      演示函数化设计带来的可测试性优势：
      每个步骤可独立测试，无需运行完整调用链路

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import sys
import time
import unittest

import sglang as sgl

# 设置测试用后端
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def classify_intent(s, question: str):
    s += sgl.user(
        f"将以下问题分类：{question}\n"
        f"从以下类别中选择：事实查询/操作指导/概念解释/问题诊断"
    )
    s += sgl.assistant(
        sgl.gen(
            "intent",
            choices=["事实查询", "操作指导", "概念解释", "问题诊断"]
        )
    )


@sgl.function
def generate_answer(s, question: str, context: str = ""):
    """生成回答，供 TestGenerateAnswer 测试。"""
    if context:
        s += sgl.user(
            f"参考信息：{context}\n\n问题：{question}\n请给出准确回答："
        )
    else:
        s += sgl.user(f"问题：{question}\n请给出准确回答：")
    s += sgl.assistant(
        sgl.gen("answer", max_tokens=200, temperature=0.1)
    )


@sgl.function
def score_answer(s, question: str, answer: str):
    """对回答质量评分（1-10），供 TestScoreAnswer 测试。"""
    s += sgl.user(
        f"评估以下回答的质量（1-10分，只输出数字）：\n"
        f"问题：{question}\n回答：{answer}"
    )
    s += sgl.assistant(
        sgl.gen(
            "score",
            max_tokens=3,
            regex=r"([1-9]|10)",
            temperature=0.0,
        )
    )


# ── 测试套件 ──────────────────────────────────────────────────

class TestClassifyIntent(unittest.TestCase):
    def test_output_within_valid_choices(self):
        # select机制保证输出必然在预定义选项内
        # 此测试验证该约束在实际执行中生效
        state = classify_intent.run(question="Python是什么？")
        self.assertIn(
            state["intent"],
            ["事实查询", "操作指导", "概念解释", "问题诊断"]
        )

    def test_multiple_inputs_consistent(self):
        # run_batch并发执行，每个结果独立验证
        states = classify_intent.run_batch([
            {"question": "如何安装numpy？"},
            {"question": "numpy的版本号是多少？"},
        ])
        for state in states:
            self.assertIn(
                state["intent"],
                ["事实查询", "操作指导", "概念解释", "问题诊断"]
            )

    def test_procedural_question_classified_correctly(self):
        # 操作指导类问题的期望类别验证
        state = classify_intent.run(
            question="如何在Python中安装第三方库？"
        )
        self.assertIn(
            state["intent"],
            ["事实查询", "操作指导", "概念解释", "问题诊断"],
        )
        # 软断言：操作类问题大概率属于操作指导
        self.assertEqual(state["intent"], "操作指导")

    def test_response_time_acceptable(self):
        # 验证单次调用响应时间在可接受范围内
        start = time.time()
        classify_intent.run(question="测试响应时间的问题")
        elapsed = time.time() - start
        # 宽松阈值：15秒内完成（取决于硬件和模型大小）
        self.assertLess(
            elapsed, 15.0,
            f"响应时间{elapsed:.2f}s超过15秒阈值"
        )

    def test_batch_all_outputs_valid(self):
        # 批量测试：验证所有并发结果均合法
        questions = [
            {"question": "如何安装numpy？"},
            {"question": "什么是梯度下降？"},
            {"question": "程序报错AttributeError怎么解决？"},
            {"question": "Python的版本号是多少？"},
        ]
        states = classify_intent.run_batch(
            questions, num_threads=4
        )
        for state in states:
            self.assertIn(
                state["intent"],
                ["事实查询", "操作指导", "概念解释", "问题诊断"],
                f"批量结果中发现非法输出：{state['intent']}"
            )


class TestGenerateAnswer(unittest.TestCase):
    def test_answer_not_empty(self):
        # 验证生成的回答不为空
        state = generate_answer.run(
            question="什么是Python的生成器？"
        )
        self.assertGreater(
            len(state["answer"].strip()), 0,
            "回答不应为空字符串"
        )

    def test_answer_length_reasonable(self):
        # 验证回答长度在合理范围内
        state = generate_answer.run(
            question="什么是机器学习？"
        )
        length = len(state["answer"])
        self.assertGreater(length, 10, "回答过短，可能无效")
        self.assertLess(length, 3000, "回答过长，超出预期")


class TestScoreAnswer(unittest.TestCase):
    def test_score_is_valid_integer(self):
        # 验证regex约束生效：输出为1-10的合法整数
        import re
        state = score_answer.run(
            question="什么是Python？",
            answer="Python是一种高级编程语言。",
        )
        self.assertRegex(
            state["score"],
            r'^([1-9]|10)$',
            f"评分格式不合法：'{state['score']}'"
        )

    def test_score_within_range(self):
        # 验证评分值在1-10范围内
        state = score_answer.run(
            question="什么是Python？",
            answer="Python是由Guido van Rossum创建的高级编程语言。",
        )
        score = int(state["score"])
        self.assertGreaterEqual(score, 1)
        self.assertLessEqual(score, 10)


def main():
    print("=" * 60)
    print("【SGLang函数单元测试演示】")
    print("=" * 60)
    print()
    print("测试说明：")
    print("  SGLang函数化设计使每个步骤可独立测试，")
    print("  无需运行完整调用链路即可验证各环节正确性。")
    print()
    print("运行测试套件...\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in [
        TestClassifyIntent,
        TestGenerateAnswer,
        TestScoreAnswer,
    ]:
        suite.addTests(
            loader.loadTestsFromTestCase(test_class)
        )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 60)
    print("【测试结果摘要】")
    print("=" * 60)
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"运行测试数：{total}")
    print(f"通过：{passed}  失败：{failed}")
    print()
    print("程序化范式的可测试性优势：")
    print("  传统方式：只能端到端测试完整调用链路")
    print("  SGLang：每个函数独立测试，精确定位问题")
    print("  choices约束：输出合法性可通过assertIn自动验证")
    print("  regex约束：格式正确性可通过assertRegex自动验证")

    # 按unittest惯例：有失败则返回非零退出码
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
