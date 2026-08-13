"""
第3章 代码示例 20
主题：多轮对话上下文管理——滑动窗口策略
场景：只保留最近N轮对话，系统提示词和首轮受保护，
      演示固定内存开销与内容连贯性的平衡

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time
from collections import deque
from typing import Dict, List

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


class SlidingWindowConversation:
    def __init__(
        self,
        window_size: int = 5,
        system_prompt: str = "",
        protect_first_turn: bool = True,
    ):
        self.system_prompt = system_prompt
        self.protect_first_turn = protect_first_turn
        # deque(maxlen=N)：超出容量时自动从左端淘汰最旧轮次
        self.recent_turns = deque(maxlen=window_size)
        # 首轮独立保存，不受滑动窗口淘汰影响
        self.first_turn = None
        self._total_turns = 0
        self._evicted_count = 0

    def add_turn(
        self, user_msg: str, assistant_msg: str
    ):
        turn = {
            "user": user_msg,
            "assistant": assistant_msg
        }
        self._total_turns += 1
        if self.protect_first_turn and self.first_turn is None:
            self.first_turn = turn
            return
        # 超出maxlen时自动淘汰最旧的轮次
        if len(self.recent_turns) == self.recent_turns.maxlen:
            self._evicted_count += 1
        self.recent_turns.append(turn)

    def build_messages(self) -> List[Dict]:
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        if self.first_turn:
            messages.extend([
                {"role": "user",
                 "content": self.first_turn["user"]},
                {"role": "assistant",
                 "content": self.first_turn["assistant"]},
            ])
        for turn in self.recent_turns:
            messages.extend([
                {"role": "user", "content": turn["user"]},
                {"role": "assistant",
                 "content": turn["assistant"]},
            ])
        return messages

    def estimate_tokens(self) -> int:
        """估算当前上下文的token数。"""
        all_text = self.system_prompt
        if self.first_turn:
            all_text += (
                self.first_turn["user"]
                + self.first_turn["assistant"]
            )
        for turn in self.recent_turns:
            all_text += turn["user"] + turn["assistant"]
        cn = sum(1 for c in all_text if '\u4e00' <= c <= '\u9fff')
        other = len(all_text) - cn
        return int(cn * 1.5 + other * 0.25)

    def get_stats(self) -> Dict:
        """返回窗口管理统计信息。"""
        protected = 1 if self.first_turn else 0
        return {
            "total_turns_added": self._total_turns,
            "turns_in_window": len(self.recent_turns),
            "protected_turns": protected,
            "evicted_turns": self._evicted_count,
            "effective_history": (
                len(self.recent_turns) + protected
            ),
            "estimated_tokens": self.estimate_tokens(),
            "memory_savings_pct": (
                self._evicted_count
                / max(1, self._total_turns) * 100
            ),
        }


@sgl.function
def windowed_chat(
    s, messages: List[Dict], current_query: str
):
    """
    使用滑动窗口管理的多轮对话函数。
    messages为SlidingWindowConversation.build_messages()的输出，
    已包含系统提示词、受保护首轮和最近N轮历史。
    RadixAttention对系统提示词部分自动复用KV Cache。
    """
    for msg in messages:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
        elif msg["role"] == "assistant":
            s += sgl.assistant(msg["content"])

    s += sgl.user(current_query)
    s += sgl.assistant(
        sgl.gen("response", max_tokens=200, temperature=0.3)
    )


def main():
    # ── 演示1：滑动窗口可见性追踪 ─────────────────────────────
    print("=" * 60)
    print("【演示1：滑动窗口可见性追踪（window_size=3）】")
    print("=" * 60)
    print()

    conv = SlidingWindowConversation(
        window_size=3,
        system_prompt="你是一位专业的编程助手。",
        protect_first_turn=True,
    )

    turns_data = [
        ("我在学Python，从哪里开始？",
         "建议从官方教程开始，学习基础语法和数据类型。"),
        ("变量和数据类型有哪些？",
         "Python有int、float、str、list、dict等基本类型。"),
        ("如何定义函数？",
         "使用def关键字，例如：def greet(name): ..."),
        ("什么是列表推导式？",
         "列表推导式是[expr for item in iterable]的语法糖。"),
        ("如何处理文件读写？",
         "使用open()函数，推荐with语句确保文件正确关闭。"),
        ("什么是装饰器？",
         "装饰器是修改函数行为的语法糖，使用@符号应用。"),
    ]

    for user_msg, asst_msg in turns_data:
        conv.add_turn(user_msg, asst_msg)

    stats = conv.get_stats()
    msgs = conv.build_messages()
    visible_users = [
        m["content"] for m in msgs if m["role"] == "user"
    ]

    print(f"共添加 {stats['total_turns_added']} 轮对话")
    print(f"窗口大小=3，首轮保护=True\n")
    print(f"当前可见的用户消息：")
    for i, u in enumerate(visible_users, 1):
        print(f"  {i}. {u[:35]}...")
    print()
    print(f"统计信息：")
    print(f"  可见轮次（窗口+保护）：{stats['effective_history']}")
    print(f"  已淘汰轮次：{stats['evicted_turns']}")
    print(f"  估算token数：{stats['estimated_tokens']}")
    print(f"  内存节省：{stats['memory_savings_pct']:.0f}%")

    # ── 演示2：与SGLang集成的实际对话 ─────────────────────────
    print()
    print("=" * 60)
    print("【演示2：与SGLang集成的实际多轮对话】")
    print("=" * 60)
    print()

    conv2 = SlidingWindowConversation(
        window_size=4,
        system_prompt=(
            "你是一位专业的Python编程教师，"
            "回答简洁准确，适合初学者理解。"
        ),
        protect_first_turn=True,
    )

    # 建立前序上下文
    context_turns = [
        ("我是Python初学者，正在学习面向对象编程。",
         "很好！面向对象编程（OOP）是Python的核心特性，"
         "主要包括类、对象、继承和多态四个概念。"),
        ("能解释一下什么是类和对象吗？",
         "类是对象的模板，对象是类的具体实例。"
         "例如'汽车'是类，你的那辆红色本田就是对象。"),
    ]
    for u, a in context_turns:
        conv2.add_turn(u, a)

    current_query = "那么继承是什么意思？"
    messages = conv2.build_messages()

    print(f"发送给LLM的消息数：{len(messages)}")
    print(f"估算token数：{conv2.estimate_tokens()}")
    print(f"当前问题：{current_query}\n")

    start = time.time()
    state = windowed_chat.run(
        messages=messages,
        current_query=current_query,
    )
    elapsed = (time.time() - start) * 1000

    print(f"回答：{state['response']}")
    print(f"耗时：{elapsed:.0f}ms")

    # ── 演示3：不同窗口大小资源对比 ──────────────────────────
    print()
    print("=" * 60)
    print("【演示3：不同窗口大小的资源消耗对比（20轮）】")
    print("=" * 60)
    print()

    sample_turns = [
        (f"第{i+1}轮问题：请详细解释一个技术概念",
         "这是一个详细的回答，包含了技术细节和示例代码。" * 3)
        for i in range(20)
    ]

    print(
        f"{'窗口大小':>8}  {'可见轮次':>8}  "
        f"{'估算tokens':>10}  {'内存节省':>8}"
    )
    print("-" * 44)

    for ws in [3, 5, 8, 10, 20]:
        c = SlidingWindowConversation(
            window_size=ws,
            system_prompt="你是助手。",
            protect_first_turn=True,
        )
        for u, a in sample_turns:
            c.add_turn(u, a)
        s = c.get_stats()
        print(
            f"  {ws:>6}轮  {s['effective_history']:>8}轮  "
            f"{s['estimated_tokens']:>10,}  "
            f"{s['memory_savings_pct']:>7.0f}%"
        )

    print()
    print("选择建议：")
    print("  window=3-5 ：独立问答，轮次间关联性弱")
    print("  window=8-10：连续对话，需要近期上下文")
    print("  window=15+ ：长任务协作，历史持续参考")


if __name__ == "__main__":
    main()
