"""
第3章 代码示例 21
主题：多轮对话上下文管理——重要性选择策略
场景：根据内容长度、关键词命中率和位置权重
      对每轮对话计算重要性评分，在token预算约束下
      贪心选择高价值历史轮次，保留语义最重要的内容

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


class ImportanceBasedConversation:
    def __init__(
        self,
        max_tokens: int = 2000,
        system_prompt: str = "",
        importance_keywords: List[str] = None,
    ):
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.keywords = importance_keywords or [
            "重要", "关键", "必须", "核心", "记住", "总结"
        ]
        self.turns = []

    def _score_turn(
        self, user_msg: str, assistant_msg: str, idx: int
    ) -> float:
        combined = user_msg + assistant_msg
        score = 0.0
        # 长度得分：较长的轮次信息量通常更大
        score += min(len(combined), 500) / 500 * 0.3
        # 关键词得分：包含重要词汇的轮次优先保留
        kw_hits = sum(
            1 for kw in self.keywords if kw in combined
        )
        score += min(kw_hits, 3) / 3 * 0.3
        # 位置得分：首轮定义任务背景，权重最高
        score += (1.0 if idx == 0 else 0.3) * 0.4
        return min(score, 1.0)

    def add_turn(
        self, user_msg: str, assistant_msg: str
    ):
        importance = self._score_turn(
            user_msg, assistant_msg, len(self.turns)
        )
        self.turns.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "importance": importance,
            # 粗略token数估算
            "token_count": int(
                len(user_msg + assistant_msg) * 1.5
            ),
        })

    def build_messages(self) -> List[Dict]:
        # 按重要性降序贪心选择，直到填满token预算
        sorted_turns = sorted(
            enumerate(self.turns),
            key=lambda x: x[1]["importance"],
            reverse=True,
        )
        selected_indices, budget = [], self.max_tokens
        if self.system_prompt:
            budget -= int(len(self.system_prompt) * 1.5)

        for idx, turn in sorted_turns:
            if turn["token_count"] <= budget:
                selected_indices.append(idx)
                budget -= turn["token_count"]

        # 恢复原始时间顺序以保证对话连贯性
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        for idx in sorted(selected_indices):
            turn = self.turns[idx]
            messages.extend([
                {"role": "user", "content": turn["user"]},
                {"role": "assistant",
                 "content": turn["assistant"]},
            ])
        return messages

    def get_selection_report(self) -> Dict:
        """返回当前选择情况的详细报告。"""
        msgs = self.build_messages()
        # 从消息列表中识别被选中的轮次索引
        selected_users = {
            m["content"] for m in msgs if m["role"] == "user"
        }
        total_tokens = sum(t["token_count"] for t in self.turns)
        selected_tokens = sum(
            t["token_count"]
            for t in self.turns
            if t["user"] in selected_users
        )
        return {
            "total_turns": len(self.turns),
            "selected_turns": len(selected_users),
            "total_tokens": total_tokens,
            "selected_tokens": selected_tokens,
            "token_budget": self.max_tokens,
            "budget_usage": selected_tokens / self.max_tokens,
            "turn_details": [
                {
                    "index": i,
                    "score": t["importance"],
                    "selected": t["user"] in selected_users,
                    "tokens": t["token_count"],
                    "preview": t["user"][:35] + "...",
                }
                for i, t in enumerate(self.turns)
            ],
        }


@sgl.function
def importance_chat(
    s, messages: List[Dict], current_query: str
):
    """
    使用重要性选择上下文的对话函数。
    messages已经过重要性评分筛选，
    在token预算内保留语义价值最高的历史轮次。
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
        sgl.gen("response", max_tokens=250, temperature=0.3)
    )


def main():
    # ── 演示1：评分详情展示 ───────────────────────────────────
    print("=" * 60)
    print("【演示1：重要性评分计算详情】")
    print("=" * 60)
    print()

    conv = ImportanceBasedConversation(
        max_tokens=800,
        system_prompt="你是一位AI架构顾问。",
    )

    turns_data = [
        (
            "我们在构建一个核心目标是高吞吐量的推理服务，"
            "这是关键需求。",
            "高吞吐量服务需要关注批处理、缓存和并发调度。"
        ),
        ("今天天气怎么样？", "这不在我的知识范围内。"),
        (
            "重要提醒：系统必须在3秒内响应，这是强制要求。",
            "了解，3秒响应时间是重要的性能约束。"
        ),
        (
            "连续批处理（Continuous Batching）的原理是什么？",
            "连续批处理允许新请求在当前批次处理中途加入，"
            "大幅提高GPU利用率和整体吞吐量。"
        ),
        (
            "记住：我们的延迟SLA要求是P99<500ms。",
            "P99<500ms是重要约束，会影响批大小的选择。"
        ),
    ]

    for u, a in turns_data:
        conv.add_turn(u, a)

    print(
        f"{'轮次':>4}  {'评分':>6}  {'选中':>6}  "
        f"{'tokens':>7}  {'内容预览'}"
    )
    print("-" * 62)

    report = conv.get_selection_report()
    for d in report["turn_details"]:
        status = "✓" if d["selected"] else "✗"
        print(
            f"  {d['index']:>4}  {d['score']:.3f}  "
            f"{status:>6}  {d['tokens']:>7}  "
            f"{d['preview']}"
        )

    print()
    print(f"共{report['total_turns']}轮，"
          f"选中{report['selected_turns']}轮，"
          f"使用{report['selected_tokens']}/"
          f"{report['token_budget']}tokens "
          f"（{report['budget_usage']:.0%}）")

    # ── 演示2：与滑动窗口的对比 ──────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：重要性选择 vs 滑动窗口信息保留对比】")
    print("=" * 60)
    print()

    # 用相同的轮次数据对比两种策略选中的内容
    from ch03_code_20 import SlidingWindowConversation
    sw_conv = SlidingWindowConversation(
        window_size=3,
        system_prompt="你是一位AI架构顾问。",
        protect_first_turn=True,
    )
    for u, a in turns_data:
        sw_conv.add_turn(u, a)

    sw_msgs = sw_conv.build_messages()
    imp_msgs = conv.build_messages()

    sw_users = [
        m["content"][:25] for m in sw_msgs
        if m["role"] == "user"
    ]
    imp_users = [
        m["content"][:25] for m in imp_msgs
        if m["role"] == "user"
    ]

    print(f"滑动窗口（window=3+保护首轮，"
          f"共{len(sw_users)}条用户消息）：")
    for u in sw_users:
        print(f"  - {u}...")

    print(f"\n重要性选择（budget=800tokens，"
          f"共{len(imp_users)}条用户消息）：")
    for u in imp_users:
        print(f"  - {u}...")

    # ── 演示3：与SGLang集成 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：与SGLang集成的实际对话】")
    print("=" * 60)
    print()

    current_query = (
        "综合我们讨论的需求和约束，推荐一个具体的技术方案。"
    )
    messages = conv.build_messages()
    print(f"发送给LLM的历史轮次：{len(messages)//2}轮")
    print(f"当前问题：{current_query}\n")

    start = time.time()
    state = importance_chat.run(
        messages=messages,
        current_query=current_query,
    )
    elapsed = (time.time() - start) * 1000

    print(f"回答：{state['response'][:200]}...")
    print(f"耗时：{elapsed:.0f}ms")

    print()
    print("=" * 60)
    print("【重要性策略适用场景】")
    print("=" * 60)
    print()
    print("优势：相同token预算下保留更多高价值信息")
    print("适用：项目协作对话、需要回溯关键决策的场景")
    print("代价：O(n log n)排序，n为历史轮次数")
    print("建议：根据业务场景定制关键词列表和权重参数")


if __name__ == "__main__":
    main()
