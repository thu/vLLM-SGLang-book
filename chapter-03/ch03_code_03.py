"""
第3章 代码示例 03
主题：SGLang多轮对话与状态自动管理
场景：使用角色原语（system/user/assistant）构建
      标准对话格式，演示s对象自动积累历史、
      通过变量名直接访问已生成内容的核心机制

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time

import sglang as sgl

# 连接本地SGLang推理服务
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def multi_turn_demo(s):
    # 使用角色原语构建标准对话格式
    s += sgl.system("你是一位专业的Python编程助手。")
    s += sgl.user("什么是Python的生成器？")
    s += sgl.assistant(sgl.gen("explanation", max_tokens=150))

    # s自动积累完整对话历史，第二轮无需重新传入前序内容
    s += sgl.user("能给我一个实际的使用例子吗？")
    s += sgl.assistant(sgl.gen("example", max_tokens=200))

    # 通过变量名直接访问任意已生成内容
    print(f"概念解释：{s['explanation']}")
    print(f"代码示例：{s['example']}")


@sgl.function
def multi_turn_extended(s, topic: str):
    """
    扩展版多轮对话：三轮递进问答，
    演示每轮都能无缝引用前序所有内容。
    """
    s += sgl.system(
        f"你是一位专业的编程助手，擅长讲解{topic}相关知识。"
    )
    # 第一轮：基础概念
    s += sgl.user(f"请简要介绍{topic}的核心概念。")
    s += sgl.assistant(sgl.gen("intro", max_tokens=150))

    # 第二轮：深入细节，s已包含第一轮完整内容
    s += sgl.user("能解释一下其中最重要的技术细节吗？")
    s += sgl.assistant(sgl.gen("detail", max_tokens=200))

    # 第三轮：实践建议，s已包含前两轮全部内容
    s += sgl.user("在实际项目中有哪些使用建议？")
    s += sgl.assistant(sgl.gen("advice", max_tokens=150))


def _print_state_explanation():
    """打印s对象状态管理的原理说明。"""
    print()
    print("=" * 60)
    print("【s对象状态管理原理】")
    print("=" * 60)
    print()
    print("传统方式需要手动维护messages列表：")
    print("  messages = []")
    print("  messages.append({'role':'user', 'content':...})")
    print("  messages.append({'role':'assistant', ...})")
    print("  # 每轮都要重新传入完整列表")
    print()
    print("SGLang的s对象自动处理以上所有操作：")
    print("  s += sgl.user(...)      # 自动追加user消息")
    print("  s += sgl.assistant(...) # 自动追加assistant消息")
    print("  s['var_name']           # 直接按变量名读取结果")
    print()
    print("RadixAttention额外收益：")
    print("  每轮新增内容追加到已有KV Cache之后，")
    print("  前序轮次的KV Cache无需重新计算。")


def main():
    print("=" * 60)
    print("【演示1：基础多轮对话（multi_turn_demo）】")
    print("=" * 60)
    print()

    start = time.time()
    # multi_turn_demo内部已有print语句，直接run即可
    state = multi_turn_demo.run()
    elapsed = (time.time() - start) * 1000

    print(f"\n完整对话文本长度：{len(state.text())} 字符")
    print(f"执行耗时：{elapsed:.0f}ms")

    print()
    print("=" * 60)
    print("【演示2：三轮递进对话（multi_turn_extended）】")
    print("=" * 60)
    print()

    topic = "Python装饰器"
    print(f"话题：{topic}\n")

    start = time.time()
    state2 = multi_turn_extended.run(topic=topic)
    elapsed2 = (time.time() - start) * 1000

    print(f"第1轮 - 核心概念：")
    print(f"  {state2['intro'][:100]}...")
    print(f"\n第2轮 - 技术细节：")
    print(f"  {state2['detail'][:100]}...")
    print(f"\n第3轮 - 实践建议：")
    print(f"  {state2['advice'][:100]}...")
    print(f"\n执行耗时：{elapsed2:.0f}ms")

    print()
    print("=" * 60)
    print("【演示3：run_batch并发执行多个独立对话】")
    print("=" * 60)
    print()

    topics = ["Python生成器", "Python上下文管理器"]
    print(f"并发执行{len(topics)}个独立话题的三轮对话...\n")

    start = time.time()
    states = multi_turn_extended.run_batch(
        [{"topic": t} for t in topics],
        num_threads=2,
    )
    elapsed3 = (time.time() - start) * 1000

    for topic, st in zip(topics, states):
        print(f"话题：{topic}")
        print(f"  简介：{st['intro'][:60]}...")
        print()

    print(f"批量总耗时：{elapsed3:.0f}ms")

    _print_state_explanation()


if __name__ == "__main__":
    main()
