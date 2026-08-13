"""
第3章 代码示例 22
主题：多轮对话上下文管理——分层摘要压缩策略
场景：将超出活跃窗口的历史对话通过LLM压缩为摘要，
      在极长对话中维持高质量的上下文感知能力，
      适用于50轮以上的持续性对话场景

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time
from collections import deque
from typing import Dict, List, Optional

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def _summarize_turns(s, turns_text: str):
    """将一组对话轮次压缩为摘要。"""
    s += sgl.user(
        f"请将以下对话内容压缩为不超过100字的摘要，"
        f"保留关键信息和结论：\n\n{turns_text}"
    )
    s += sgl.assistant(
        sgl.gen("summary", max_tokens=150, temperature=0.1)
    )


class HierarchicalSummaryConversation:
    def __init__(
        self,
        active_window: int = 5,
        compress_threshold: int = 10,
    ):
        self.active_window = active_window
        self.compress_threshold = compress_threshold
        # 活跃窗口：最近N轮，不压缩
        self.active_turns = deque(maxlen=active_window)
        # 摘要层：已压缩的历史摘要列表
        self.summaries: List[str] = []
        self._total_turns = 0
        self._compression_count = 0

    def add_turn(
        self, user_msg: str, assistant_msg: str
    ):
        # 活跃窗口满时，将最旧的轮次纳入压缩候选
        if len(self.active_turns) == self.active_window:
            self._maybe_compress()
        self.active_turns.append({
            "user": user_msg,
            "assistant": assistant_msg,
        })
        self._total_turns += 1

    def _maybe_compress(self):
        """
        当活跃轮次超出阈值时，将最旧的批次压缩为摘要。
        使用SGLang调用LLM完成摘要生成。
        """
        # 提取最旧的一批轮次进行压缩
        turns_to_compress = list(self.active_turns)[
            :self.compress_threshold // 2
        ]
        turns_text = "\n".join([
            f"用户：{t['user']}\n助手：{t['assistant']}"
            for t in turns_to_compress
        ])
        state = _summarize_turns.run(
            turns_text=turns_text
        )
        self.summaries.append(state["summary"])
        self._compression_count += 1

    def build_context(self) -> str:
        """构建包含摘要历史与活跃轮次的完整上下文。"""
        parts = []
        if self.summaries:
            parts.append(
                "【历史摘要】\n" + "\n".join(self.summaries)
            )
        parts.append("【近期对话】")
        for turn in self.active_turns:
            parts.append(
                f"用户：{turn['user']}\n"
                f"助手：{turn['assistant']}"
            )
        return "\n\n".join(parts)

    def estimate_context_tokens(self) -> int:
        """估算当前上下文的总token数。"""
        context = self.build_context()
        cn = sum(
            1 for c in context if '\u4e00' <= c <= '\u9fff'
        )
        other = len(context) - cn
        return int(cn * 1.5 + other * 0.25)

    def get_structure_report(self) -> Dict:
        """返回分层摘要结构的统计报告。"""
        return {
            "total_turns": self._total_turns,
            "active_turns": len(self.active_turns),
            "summary_count": len(self.summaries),
            "compression_count": self._compression_count,
            "estimated_context_tokens": (
                self.estimate_context_tokens()
            ),
        }


@sgl.function
def hierarchical_chat(
    s,
    system_prompt: str,
    context_summary: str,
    current_query: str,
):
    """
    使用分层摘要上下文的对话函数。
    context_summary由build_context()生成，
    包含历史摘要层和近期完整对话轮次。
    """
    s += sgl.system(system_prompt)

    if context_summary:
        s += sgl.user(
            f"以下是此前对话的背景信息：\n\n{context_summary}"
        )
        s += sgl.assistant(
            "我已了解此前对话的背景，请继续。"
        )

    s += sgl.user(current_query)
    s += sgl.assistant(
        sgl.gen("response", max_tokens=300, temperature=0.3)
    )


def main():
    system_prompt = (
        "你是一位AI架构顾问，正在协助规划一个推理服务项目。"
    )

    # ── 演示1：逐轮添加并观察结构变化 ─────────────────────────
    print("=" * 60)
    print("【演示1：逐轮添加——观察分层摘要结构变化】")
    print("=" * 60)
    print()

    conv = HierarchicalSummaryConversation(
        active_window=3,
        compress_threshold=4,
    )

    dialogue = [
        ("我们需要构建支持100QPS的LLM推理服务。",
         "100QPS是中等规模，需要考虑模型大小、硬件和批处理策略。"),
        ("预算每月5万元，主要用于GPU服务器。",
         "5万月预算可租用2-4张A100，建议优先考虑H100。"),
        ("P99延迟要求是500ms以内。",
         "500ms P99是可达的，需配合连续批处理和量化。"),
        ("选择7B还是13B模型？",
         "100QPS场景7B模型完全够用，建议从7B开始。"),
        ("KV Cache应该怎么配置？",
         "KV Cache通常设置为GPU显存的60-70%，其余留给权重。"),
        ("需要用到流式输出吗？",
         "推荐流式输出，能将感知延迟从秒级降到毫秒级。"),
    ]

    for i, (user_msg, asst_msg) in enumerate(dialogue):
        conv.add_turn(user_msg, asst_msg)
        report = conv.get_structure_report()

        if (i + 1) in {2, 4, 6}:
            print(f"第{i+1}轮后的结构：")
            print(f"  活跃窗口：{report['active_turns']}轮")
            print(f"  摘要数量：{report['summary_count']}个")
            print(
                f"  上下文tokens≈"
                f"{report['estimated_context_tokens']}"
            )
            print()

    # 打印当前完整上下文
    context = conv.build_context()
    print(f"当前上下文内容：\n{'-'*40}")
    print(context[:400] + ("..." if len(context) > 400 else ""))
    print(f"{'-'*40}")
    print(f"上下文总长度：{len(context)}字符")
    print(
        f"估算tokens：{conv.estimate_context_tokens()}"
    )

    # ── 演示2：基于压缩上下文的对话 ──────────────────────────
    print()
    print("=" * 60)
    print("【演示2：基于压缩上下文的继续对话】")
    print("=" * 60)
    print()

    current_query = (
        "综合我们讨论的所有需求和约束，"
        "给出一份完整的技术选型建议。"
    )
    print(f"当前问题：{current_query}\n")

    start = time.time()
    state = hierarchical_chat.run(
        system_prompt=system_prompt,
        context_summary=conv.build_context(),
        current_query=current_query,
    )
    elapsed = (time.time() - start) * 1000

    print(f"回答：{state['response'][:200]}...")
    print(f"耗时：{elapsed:.0f}ms")

    # ── 演示3：压缩效果量化 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：压缩效果量化（10轮对话）】")
    print("=" * 60)
    print()

    # 无压缩：累积所有历史
    total_chars_no_compress = sum(
        len(u) + len(a) for u, a in dialogue
    ) * 2  # 粗略估算含格式字符
    tokens_no_compress = int(total_chars_no_compress * 1.5)

    # 有压缩：只保留摘要+活跃窗口
    tokens_with_compress = conv.estimate_context_tokens()

    print(f"总对话轮次：{conv._total_turns}")
    print(f"触发压缩次数：{conv._compression_count}")
    print()
    print(f"无压缩时估算tokens：{tokens_no_compress:,}")
    print(f"有压缩后估算tokens：{tokens_with_compress:,}")
    if tokens_no_compress > 0:
        reduction = (
            1 - tokens_with_compress / tokens_no_compress
        ) * 100
        print(f"上下文压缩率：{reduction:.0f}%")

    print()
    print("=" * 60)
    print("【分层摘要策略适用场景与权衡】")
    print("=" * 60)
    print()
    print("最适合：50轮以上的超长对话，跨会话持续项目")
    print("质量优势：保留全局语义，不依赖时间顺序淘汰")
    print("主要代价：每次压缩需要额外LLM调用（约100-300ms）")
    print("参数建议：active_window=5，compress_threshold=5")


if __name__ == "__main__":
    main()
