"""
第3章 代码示例 19
主题：多轮对话上下文管理——朴素实现的资源消耗问题
场景：演示不加任何上下文管理时，KV Cache占用、
      延迟估算随对话轮次线性增长的定量分析，
      为后续三种上下文管理策略提供对照基准

说明：
    KV Cache内存估算基于LLaMA-3-70B的实际参数：
      2（K和V）× 80层 × 8个KV头 × 128维 × 2字节（FP16）
      ≈ 2.5MB / token
    无需启动SGLang服务，可直接运行。

运行前提：
    pip install sglang
"""

import math
from typing import Dict, List


class NaiveConversation:
    """
    无上下文管理的朴素实现。
    每轮直接追加，历史随轮次无限增长。
    用于演示不加管理时的资源消耗问题。
    """
    def __init__(self):
        # 消息列表：随对话轮次线性无界增长
        self.messages = []

    def add_turn(
        self, user_msg: str, assistant_msg: str
    ):
        self.messages.append(
            {"role": "user", "content": user_msg}
        )
        self.messages.append(
            {"role": "assistant", "content": assistant_msg}
        )

    def estimate_tokens(self) -> int:
        """粗略估算当前上下文的token数。"""
        # 简化估算：平均每个汉字约1.5个token
        total_chars = sum(
            len(m["content"]) for m in self.messages
        )
        return int(total_chars * 1.5)

    def get_context_for_turn(self, turn: int) -> dict:
        """统计第n轮时的上下文规模。"""
        tokens = self.estimate_tokens()
        # LLaMA-70B的KV Cache每token约占2.5MB（FP16）
        kv_cache_mb = tokens * 2.5
        return {
            "turn": turn,
            "total_tokens": tokens,
            "kv_cache_mb": kv_cache_mb,
            "estimated_latency_ms": tokens * 0.05,
        }


def simulate_growth(
    n_turns: int,
    avg_user_tokens: int = 50,
    avg_assistant_tokens: int = 200,
) -> List[Dict]:
    """
    模拟n轮对话后各轮次的资源消耗数据。
    avg_user_tokens/avg_assistant_tokens为每轮的平均token数。
    """
    conv = NaiveConversation()
    results = []

    for i in range(1, n_turns + 1):
        # 用重复字符模拟指定长度的消息
        # 中文字符约1.5token，故chars = tokens / 1.5
        user_chars = int(avg_user_tokens / 1.5)
        asst_chars = int(avg_assistant_tokens / 1.5)
        user_msg = "问" * user_chars
        asst_msg = "答" * asst_chars

        conv.add_turn(user_msg, asst_msg)
        stats = conv.get_context_for_turn(i)
        results.append(stats)

    return results


def format_memory(mb: float) -> str:
    """格式化内存大小显示。"""
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    return f"{mb:.0f}MB"


def print_growth_table(
    results: List[Dict],
    h100_vram_gb: float = 80.0,
):
    """以表格形式打印资源增长趋势。"""
    print(
        f"{'轮次':>4}  {'总tokens':>8}  "
        f"{'KV Cache':>10}  {'延迟估算':>10}  "
        f"{'显存占比':>8}"
    )
    print("-" * 52)

    milestones = {1, 5, 10, 20, 30, 50}
    for stats in results:
        turn = stats["turn"]
        if turn not in milestones and turn != results[-1]["turn"]:
            continue

        tokens = stats["total_tokens"]
        kv_mb = stats["kv_cache_mb"]
        kv_gb = kv_mb / 1024
        latency = stats["estimated_latency_ms"]
        vram_pct = kv_gb / h100_vram_gb * 100

        warning = ""
        if kv_gb > h100_vram_gb * 0.8:
            warning = " ⚠ OOM风险"
        elif kv_gb > h100_vram_gb * 0.5:
            warning = " △ 显存紧张"

        print(
            f"{turn:>4}轮  {tokens:>8,}  "
            f"{format_memory(kv_mb):>10}  "
            f"{latency:>8.0f}ms  "
            f"{vram_pct:>7.1f}%{warning}"
        )


def main():
    # ── 演示1：资源增长趋势（LLaMA-70B）────────────────────
    print("=" * 60)
    print("【多轮对话资源消耗分析（无上下文管理）】")
    print("=" * 60)
    print()
    print("模型：LLaMA-3-70B（FP16精度）")
    print("KV Cache估算：约2.5MB / token")
    print("H100显存容量：80GB")
    print("每轮配置：用户50tokens + 助手200tokens")
    print()

    results = simulate_growth(
        n_turns=50,
        avg_user_tokens=50,
        avg_assistant_tokens=200,
    )
    print_growth_table(results, h100_vram_gb=80.0)

    # 找出OOM阈值
    oom_turn = None
    for stats in results:
        if stats["kv_cache_mb"] / 1024 > 80.0:
            oom_turn = stats["turn"]
            break

    print()
    if oom_turn:
        print(f"⚠ 第{oom_turn}轮时KV Cache超过H100显存（80GB），")
        print(f"  将触发OOM错误或强制截断历史。")

    r1 = results[0]
    r10 = results[9]
    r50 = results[-1]
    print(
        f"\n第1轮延迟估算：{r1['estimated_latency_ms']:.0f}ms"
    )
    print(
        f"第10轮延迟估算：{r10['estimated_latency_ms']:.0f}ms"
        f"（增加{r10['estimated_latency_ms']/r1['estimated_latency_ms']:.0f}倍）"
    )
    print(
        f"第50轮KV Cache：{format_memory(r50['kv_cache_mb'])}"
    )

    # ── 演示2：不同消息长度对比 ───────────────────────────────
    print()
    print("=" * 60)
    print("【不同消息长度配置的资源对比（20轮）】")
    print("=" * 60)
    print()

    configs = [
        ("短消息（50+100 tokens）",  50, 100),
        ("中消息（50+200 tokens）",  50, 200),
        ("长消息（100+500 tokens）", 100, 500),
    ]

    print(
        f"{'配置':<22}  {'第20轮tokens':>12}  "
        f"{'第20轮KV':>10}  {'第20轮延迟':>10}"
    )
    print("-" * 60)
    for name, u_tok, a_tok in configs:
        r = simulate_growth(20, u_tok, a_tok)
        s = r[-1]
        print(
            f"  {name:<20}  {s['total_tokens']:>12,}  "
            f"{format_memory(s['kv_cache_mb']):>10}  "
            f"{s['estimated_latency_ms']:>8.0f}ms"
        )

    # ── 演示3：NaiveConversation直接使用演示 ─────────────────
    print()
    print("=" * 60)
    print("【NaiveConversation直接使用演示】")
    print("=" * 60)
    print()

    conv = NaiveConversation()
    sample_turns = [
        ("什么是大语言模型？",
         "大语言模型是基于Transformer架构的神经网络，"
         "通过在海量文本上预训练获得语言理解和生成能力。"),
        ("它有哪些主要应用？",
         "主要应用包括：文本生成、代码辅助、问答系统、"
         "翻译、摘要、情感分析等多个领域。"),
        ("部署时有哪些挑战？",
         "主要挑战包括显存占用大、推理延迟高、"
         "成本高昂以及长上下文处理困难等。"),
    ]

    for user_msg, asst_msg in sample_turns:
        conv.add_turn(user_msg, asst_msg)
        turn_num = len(conv.messages) // 2
        stats = conv.get_context_for_turn(turn_num)
        print(
            f"第{turn_num}轮后："
            f"  tokens={stats['total_tokens']}  "
            f"KV≈{format_memory(stats['kv_cache_mb'])}  "
            f"延迟≈{stats['estimated_latency_ms']:.0f}ms"
        )

    print()
    print("=" * 60)
    print("【结论：为什么需要上下文管理？】")
    print("=" * 60)
    print()
    print("1. 显存限制：70B模型50轮后KV Cache逼近甚至超过单卡容量")
    print("2. 延迟增长：Attention的O(n²)使延迟随轮次快速上升")
    print("3. 成本累积：API按token计费，历史越长成本越高")
    print("4. 窗口溢出：超出上下文窗口后必须截断，导致信息丢失")
    print()
    print("后续示例将介绍三种管理策略：")
    print("  ch03_code_20：滑动窗口策略")
    print("  ch03_code_21：重要性选择策略")
    print("  ch03_code_22：分层摘要压缩策略")


if __name__ == "__main__":
    main()
