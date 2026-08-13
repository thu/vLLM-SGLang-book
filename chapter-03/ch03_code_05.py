"""
第3章 代码示例 05
主题：sgl.select原语与条件分支控制流
场景：使用choices参数让模型从预定义选项中选择，
      并将选择结果直接驱动Python控制流，
      实现情绪感知的自适应回应策略

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

说明：
    sgl.gen(choices=[...]) 通过对数概率评分从候选项
    中选取最可能的一项，而非开放式生成。
    输出必然是 choices 中的某一项，可安全用于控制流。
"""

import time

import sglang as sgl

# 连接本地SGLang推理服务
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def adaptive_response(s, user_input: str):
    s += sgl.user(f"用户反馈：{user_input}\n请评估用户情绪：")
    # select通过对数概率评分，从候选项中选取最可能的一项
    s += sgl.assistant(
        sgl.gen(
            "emotion",
            choices=["满意", "中性", "不满", "困惑"]
        )
    )

    # select结果直接驱动Python控制流，逻辑清晰无歧义
    if s["emotion"] == "不满":
        s += sgl.user("请提供解决方案。")
        s += sgl.assistant(sgl.gen("solution", max_tokens=200))
    elif s["emotion"] == "困惑":
        s += sgl.user("请用更简单的方式重新解释。")
        s += sgl.assistant(
            sgl.gen("clarification", max_tokens=150)
        )
    else:
        s += sgl.assistant(sgl.gen("reply", max_tokens=100))


@sgl.function
def tool_dispatch(s, question: str):
    """
    工具路由示例：根据问题类型选择合适的工具，
    演示select在分类决策场景中的典型应用。
    """
    s += sgl.user(
        f"问题：{question}\n"
        f"回答这个问题需要使用哪种工具？"
    )
    s += sgl.assistant(
        sgl.gen(
            "tool",
            choices=["计算器", "搜索引擎", "代码解释器", "直接回答"]
        )
    )

    if s["tool"] == "计算器":
        s += sgl.user("请给出数学表达式和计算步骤：")
        s += sgl.assistant(
            sgl.gen("result", max_tokens=100)
        )
    elif s["tool"] == "代码解释器":
        s += sgl.user("请给出解决这个问题的Python代码：")
        s += sgl.assistant(
            sgl.gen("result", max_tokens=200, stop="```")
        )
    else:
        s += sgl.user("请直接回答这个问题：")
        s += sgl.assistant(
            sgl.gen("result", max_tokens=150)
        )


def _get_response_key(state) -> str:
    """根据情绪选择对应的回应变量名。"""
    emotion = state["emotion"]
    if emotion == "不满":
        return "solution"
    elif emotion == "困惑":
        return "clarification"
    else:
        return "reply"


def _print_select_explanation():
    """打印select机制与open-ended gen的区别说明。"""
    print()
    print("=" * 60)
    print("【select机制说明】")
    print("=" * 60)
    print()
    print("sgl.gen(choices=[...]) 的工作方式：")
    print("  对每个候选项分别计算对数概率，选取最高项")
    print("  不执行自由生成，输出必然是choices中的某项")
    print("  速度快于open-ended gen（无需多步采样）")
    print()
    print("适用场景：")
    print("  分类决策、工具路由、意图识别、情绪判断")
    print()
    print("与regex约束的区别：")
    print("  choices：从有限离散集合中选择最优项")
    print("  regex  ：约束生成内容符合指定格式模式")


def main():
    print("=" * 60)
    print("【演示1：情绪感知的自适应回应】")
    print("=" * 60)
    print()

    test_inputs = [
        "这个产品真的很好用，超出了我的预期！",
        "等了两周还没收到货，非常失望！",
        "我不太明白这个功能怎么操作，能解释一下吗？",
        "收到了，包装完好，谢谢。",
    ]

    for user_input in test_inputs:
        start = time.time()
        state = adaptive_response.run(user_input=user_input)
        elapsed = (time.time() - start) * 1000

        emotion = state["emotion"]
        key = _get_response_key(state)
        response = state.get(key, "")

        print(f"用户输入：{user_input[:35]}...")
        print(f"  情绪判断：{emotion}")
        print(f"  回应内容：{response[:70]}...")
        print(f"  耗时：{elapsed:.0f}ms")
        print()

    print("=" * 60)
    print("【演示2：工具路由决策】")
    print("=" * 60)
    print()

    questions = [
        "23乘以47等于多少？",
        "Python中如何对列表进行去重？",
        "今天天气怎么样？",
    ]

    for q in questions:
        state = tool_dispatch.run(question=q)
        print(f"问题：{q}")
        print(f"  选择工具：{state['tool']}")
        result = state.get("result", "")
        print(f"  处理结果：{result[:60]}...")
        print()

    print("=" * 60)
    print("【演示3：run_batch并发情绪分析】")
    print("=" * 60)
    print()

    batch_inputs = [
        {"user_input": "物流太慢了，已经等了一个月！"},
        {"user_input": "产品质量很好，会继续购买。"},
        {"user_input": "这个功能我不太会用，怎么操作？"},
    ]

    print(f"并发分析{len(batch_inputs)}条用户反馈...\n")
    start = time.time()
    states = adaptive_response.run_batch(
        batch_inputs,
        num_threads=3,
    )
    elapsed = (time.time() - start) * 1000

    for item, st in zip(batch_inputs, states):
        print(
            f"  反馈：{item['user_input'][:25]}...  "
            f"→ 情绪：{st['emotion']}"
        )

    print(f"\n批量总耗时：{elapsed:.0f}ms")

    _print_select_explanation()


if __name__ == "__main__":
    main()
