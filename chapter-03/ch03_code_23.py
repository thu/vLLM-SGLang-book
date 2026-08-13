"""
第3章 代码示例 23
主题：上下文管理策略与RadixAttention的协同集成
场景：将上下文压缩（滑动窗口/重要性选择/分层摘要）
      与SGLang的RadixAttention结合，演示两者的乘数效应：
      上下文压缩减少总token量，RadixAttention减少实际计算量

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
def managed_conversation(
    s,
    system_prompt: str,
    compressed_history: str,
    active_turns: List[Dict],
    current_query: str,
):
    """
    结合上下文压缩与RadixAttention的多轮对话函数。
    系统提示词和压缩历史作为公共前缀，
    RadixAttention对其KV Cache自动复用。
    """
    # 系统提示词：所有会话共享，KV Cache自动复用
    s += sgl.system(system_prompt)

    if compressed_history:
        # 压缩历史作为背景上下文传入，体积已经过压缩
        s += sgl.user(
            f"以下是此前对话的摘要：\n{compressed_history}"
        )
        s += sgl.assistant("我已了解此前的对话背景。")

    # 活跃窗口：最近几轮完整对话
    for turn in active_turns:
        s += sgl.user(turn["user"])
        s += sgl.assistant(turn["assistant"])

    # 当前轮次的用户输入
    s += sgl.user(current_query)
    s += sgl.assistant(
        sgl.gen("response", max_tokens=300, temperature=0.3)
    )

    return s["response"]


# ── 三种上下文管理策略的简化接口 ─────────────────────────────

class SimpleContextManager:
    """
    统一接口的上下文管理器，支持三种策略模式：
      "sliding"    ：滑动窗口
      "importance" ：重要性选择（简化版）
      "none"       ：无管理（全量保留，用于对比）
    """

    def __init__(
        self,
        strategy: str = "sliding",
        window_size: int = 5,
        max_tokens: int = 2000,
        system_prompt: str = "",
    ):
        self.strategy = strategy
        self.system_prompt = system_prompt
        self._all_turns: List[Dict] = []
        self._window_size = window_size
        self._max_tokens = max_tokens
        self._summaries: List[str] = []

    def add_turn(self, user: str, assistant: str):
        self._all_turns.append({
            "user": user,
            "assistant": assistant,
        })

    def add_summary(self, summary: str):
        """添加外部生成的历史摘要（用于分层摘要策略）。"""
        self._summaries.append(summary)

    def get_active_turns(self) -> List[Dict]:
        """根据策略返回活跃轮次列表。"""
        if self.strategy == "none":
            return list(self._all_turns)
        elif self.strategy == "sliding":
            # 首轮保护 + 最近N轮
            if not self._all_turns:
                return []
            first = [self._all_turns[0]]
            recent = list(self._all_turns[-self._window_size:])
            # 避免首轮重复
            if len(self._all_turns) <= self._window_size:
                return list(self._all_turns)
            if recent[0] == first[0]:
                return recent
            return first + recent[1:] if len(recent) > 0 else first
        elif self.strategy == "importance":
            # 简化版重要性选择：首轮 + 含关键词的轮次 + 最新2轮
            keywords = ["重要", "关键", "必须", "记住", "总结"]
            selected = []
            if self._all_turns:
                selected.append(self._all_turns[0])
            for turn in self._all_turns[1:]:
                combined = turn["user"] + turn["assistant"]
                if any(kw in combined for kw in keywords):
                    selected.append(turn)
            # 确保最近2轮总是包含
            for turn in self._all_turns[-2:]:
                if turn not in selected:
                    selected.append(turn)
            return selected
        return list(self._all_turns)

    def get_compressed_history(self) -> str:
        """返回压缩后的历史摘要字符串。"""
        return "\n".join(self._summaries)

    def estimate_tokens(self) -> int:
        """估算当前上下文token数。"""
        active = self.get_active_turns()
        hist = self.get_compressed_history()
        all_text = self.system_prompt + hist
        for t in active:
            all_text += t["user"] + t["assistant"]
        cn = sum(1 for c in all_text if '\u4e00' <= c <= '\u9fff')
        other = len(all_text) - cn
        return int(cn * 1.5 + other * 0.25)


def run_managed_dialogue(
    system_prompt: str,
    history_turns: List[Dict],
    current_query: str,
    strategy: str = "sliding",
    window_size: int = 4,
    compressed_summary: str = "",
) -> Dict:
    """
    使用指定策略运行一次托管对话，返回回答和性能指标。
    """
    manager = SimpleContextManager(
        strategy=strategy,
        window_size=window_size,
        system_prompt=system_prompt,
    )
    for turn in history_turns:
        manager.add_turn(turn["user"], turn["assistant"])
    if compressed_summary:
        manager.add_summary(compressed_summary)

    active_turns = manager.get_active_turns()
    compressed_hist = manager.get_compressed_history()
    estimated_tokens = manager.estimate_tokens()

    start = time.time()
    state = managed_conversation.run(
        system_prompt=system_prompt,
        compressed_history=compressed_hist,
        active_turns=active_turns,
        current_query=current_query,
    )
    elapsed = (time.time() - start) * 1000

    return {
        "response": state.ret_value,
        "strategy": strategy,
        "active_turns_count": len(active_turns),
        "estimated_tokens": estimated_tokens,
        "latency_ms": elapsed,
    }


def main():
    system_prompt = (
        "你是一位专业的云架构师，"
        "正在协助一家初创公司规划AI推理基础设施。"
    )

    # 模拟10轮历史对话
    history = [
        {"user": "我们的主要业务是提供AI写作辅助服务，月活约10万。",
         "assistant": "了解，10万月活的AI写作服务，延迟和成本是关键指标。"},
        {"user": "用户主要分布在亚洲和北美。",
         "assistant": "跨区域需要考虑就近接入，建议亚太和北美各部署集群。"},
        {"user": "核心功能是长文写作，每次生成约1000个token。",
         "assistant": "1000tokens生成量较大，建议使用流式输出减少等待感。"},
        {"user": "并发峰值约200个请求。",
         "assistant": "200QPS峰值需要至少2-3张H100，建议按3倍峰值设计容量。"},
        {"user": "重要约束：预算每月10万元。",
         "assistant": "10万月预算在国内可租用约4-6张A100。"},
        {"user": "P99延迟必须控制在2秒以内。",
         "assistant": "2秒P99对1000token生成是可达的，需配合流式和量化。"},
        {"user": "是否需要考虑多模态能力？",
         "assistant": "建议先专注文本，多模态作为后期迭代方向。"},
        {"user": "安全合规有什么要求？",
         "assistant": "建议部署内容过滤层，特别是针对GDPR和国内合规。"},
        {"user": "如何做服务高可用？",
         "assistant": "建议多节点部署加健康检查，SLA目标99.9%。"},
        {"user": "记住：我们必须在3个月内上线。",
         "assistant": "3个月上线时间紧，建议优先选用成熟方案而非自研。"},
    ]

    current_query = (
        "综合以上所有需求，给出最终的架构选型建议。"
    )

    # ── 演示1：三种策略的资源消耗对比 ────────────────────────
    print("=" * 60)
    print("【演示1：三种策略的上下文token数对比】")
    print("=" * 60)
    print()

    strategies = [
        ("none",       5,  "全量保留（无管理）"),
        ("sliding",    4,  "滑动窗口（window=4）"),
        ("importance", 4,  "重要性选择"),
    ]

    for strategy, ws, desc in strategies:
        manager = SimpleContextManager(
            strategy=strategy,
            window_size=ws,
            system_prompt=system_prompt,
        )
        for turn in history:
            manager.add_turn(turn["user"], turn["assistant"])
        tokens = manager.estimate_tokens()
        active = manager.get_active_turns()
        print(
            f"  {desc:<20}：活跃{len(active):>2}轮，"
            f"约{tokens:>5}tokens"
        )

    # ── 演示2：滑动窗口策略实际对话 ──────────────────────────
    print()
    print("=" * 60)
    print("【演示2：滑动窗口策略的实际对话】")
    print("=" * 60)
    print()
    print(f"历史轮次：{len(history)}轮")
    print(f"当前问题：{current_query}\n")

    result = run_managed_dialogue(
        system_prompt=system_prompt,
        history_turns=history,
        current_query=current_query,
        strategy="sliding",
        window_size=4,
    )

    print(f"策略：{result['strategy']}")
    print(f"活跃轮次：{result['active_turns_count']}")
    print(f"估算tokens：{result['estimated_tokens']}")
    print(f"耗时：{result['latency_ms']:.0f}ms")
    print(f"\n回答：{result['response'][:200]}...")

    # ── 演示3：带压缩摘要的对话 ──────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：含压缩历史摘要的对话（模拟分层策略）】")
    print("=" * 60)
    print()

    # 模拟一段已压缩的历史摘要
    mock_summary = (
        "前期讨论确定：月活10万的AI写作服务，"
        "亚太+北美双区域部署，1000token/次，200QPS峰值，"
        "P99<2秒，月预算10万，3个月上线，需内容安全合规。"
    )

    result3 = run_managed_dialogue(
        system_prompt=system_prompt,
        history_turns=history[-3:],   # 只保留最近3轮
        current_query=current_query,
        strategy="none",
        compressed_summary=mock_summary,
    )

    print(f"压缩摘要：{mock_summary[:60]}...")
    print(f"活跃轮次：{result3['active_turns_count']}（最近3轮）")
    print(f"估算tokens：{result3['estimated_tokens']}")
    print(f"耗时：{result3['latency_ms']:.0f}ms")
    print(f"\n回答：{result3['response'][:200]}...")

    # ── RadixAttention协同效应说明 ────────────────────────────
    print()
    print("=" * 60)
    print("【上下文管理与RadixAttention的协同效应】")
    print("=" * 60)
    print()
    print("系统提示词（固定不变）：")
    print("  → RadixAttention跨所有请求自动复用KV Cache")
    print()
    print("压缩摘要（同一会话内相对稳定）：")
    print("  → 上下文压缩减少总token量")
    print("  → RadixAttention对摘要部分渐进复用")
    print()
    print("活跃轮次（每轮新增）：")
    print("  → 只有新增部分需要完整Prefill计算")
    print()
    print("综合效果：双重减少计算量，产生乘数效应")


if __name__ == "__main__":
    main()
