"""
第3章 代码示例 11
主题：运行时前缀追踪与静态片段合并优化
场景：在函数执行过程中记录各文本片段的静态/动态属性，
      并将连续静态片段合并为更大的缓存单元，
      最大化RadixAttention的KV Cache命中粒度

说明：
    本示例为教学性实现，展示运行时追踪的核心逻辑。
    SGLang生产环境中该机制与解释器深度集成。
    无需启动SGLang服务，可直接运行。

运行前提：
    pip install sglang
"""

import hashlib
import time
from typing import Dict, List


class RuntimePrefixTracker:
    def __init__(self):
        # trace_key -> 片段列表的映射
        self.traces = {}

    def start_trace(self, func_name: str, args_hash: str) -> str:
        # 以函数名+参数哈希唯一标识一次函数调用
        key = f"{func_name}:{args_hash}"
        self.traces[key] = []
        return key

    def record_segment(
        self, key: str, content: str, is_static: bool
    ):
        # is_static=True：跨请求不变，可缓存
        # is_static=False：每次不同，不可跨请求缓存
        self.traces[key].append(
            {"content": content, "static": is_static}
        )

    def merge_static_segments(self, key: str):
        """
        将连续的静态片段合并为更大的缓存单元。
        粒度越大，RadixAttention命中率越高。
        """
        trace = self.traces.get(key, [])
        merged, buffer = [], []
        for seg in trace:
            if seg["static"]:
                buffer.append(seg["content"])
            else:
                if buffer:
                    merged.append({
                        "content": "".join(buffer),
                        "static": True
                    })
                    buffer = []
                merged.append(seg)
        if buffer:
            merged.append(
                {"content": "".join(buffer), "static": True}
            )
        self.traces[key] = merged
        return merged

    def get_cache_efficiency(self, key: str) -> float:
        """
        计算该次调用的理论缓存效率：
        静态token数 / 总token数 = 可被缓存的比例。
        使用字符数作为token数的近似估算。
        """
        trace = self.traces.get(key, [])
        total = sum(len(seg["content"]) for seg in trace)
        static = sum(
            len(seg["content"])
            for seg in trace if seg["static"]
        )
        return static / total if total > 0 else 0.0

    def get_trace_summary(self, key: str) -> Dict:
        """返回追踪结果的统计摘要。"""
        trace = self.traces.get(key, [])
        total_chars = sum(len(s["content"]) for s in trace)
        static_chars = sum(
            len(s["content"]) for s in trace if s["static"]
        )
        static_segs = sum(1 for s in trace if s["static"])
        dynamic_segs = sum(1 for s in trace if not s["static"])
        return {
            "total_segments": len(trace),
            "static_segments": static_segs,
            "dynamic_segments": dynamic_segs,
            "total_chars": total_chars,
            "static_chars": static_chars,
            "cache_efficiency": (
                static_chars / total_chars if total_chars else 0.0
            ),
        }


def simulate_translation_trace(
    tracker: RuntimePrefixTracker,
    text: str,
    target_lang: str,
) -> str:
    """
    模拟翻译函数执行时的片段追踪。
    静态部分（系统指令骨架）标记为可缓存，
    动态部分（变量值）标记为不可跨请求缓存。
    """
    args_hash = hashlib.md5(
        f"text={text}&lang={target_lang}".encode()
    ).hexdigest()[:8]
    key = tracker.start_trace("translate", args_hash)

    # 静态：系统角色指令，所有请求共享
    tracker.record_segment(
        key, "你是专业翻译，翻译准确自然，保持原文风格。\n",
        is_static=True,
    )
    # 静态：f-string骨架部分
    tracker.record_segment(
        key, "请将以下文本翻译为", is_static=True
    )
    # 动态：变量target_lang，每次可能不同
    tracker.record_segment(key, target_lang, is_static=False)
    # 静态：骨架继续
    tracker.record_segment(key, "：\n原文：", is_static=True)
    # 动态：变量text，每次不同
    tracker.record_segment(key, text, is_static=False)
    # 静态：结尾提示词
    tracker.record_segment(key, "\n译文：", is_static=True)

    return key


def simulate_few_shot_trace(
    tracker: RuntimePrefixTracker,
    query: str,
) -> str:
    """
    模拟Few-shot函数执行时的片段追踪。
    示例部分完全静态，只有查询是动态的，
    缓存效率极高。
    """
    args_hash = hashlib.md5(
        f"query={query}".encode()
    ).hexdigest()[:8]
    key = tracker.start_trace("few_shot_classify", args_hash)

    # 静态：所有Few-shot示例，跨所有请求共享
    tracker.record_segment(
        key,
        "示例1：输入"产品很好用" → 输出：积极\n",
        is_static=True,
    )
    tracker.record_segment(
        key,
        "示例2：输入"服务太差了" → 输出：消极\n",
        is_static=True,
    )
    tracker.record_segment(
        key,
        "示例3：输入"一般般吧" → 输出：中性\n",
        is_static=True,
    )
    tracker.record_segment(
        key, "请分类：", is_static=True
    )
    # 动态：当前查询，每次不同
    tracker.record_segment(key, query, is_static=False)
    tracker.record_segment(key, "\n输出：", is_static=True)

    return key


def main():
    tracker = RuntimePrefixTracker()

    # ── 演示1：翻译函数追踪与合并 ─────────────────────────────
    print("=" * 60)
    print("【演示1：翻译函数的片段追踪与合并】")
    print("=" * 60)
    print()

    key1 = simulate_translation_trace(
        tracker, "Hello, world!", "中文"
    )

    print("合并前的追踪片段：")
    for i, seg in enumerate(
        tracker.traces[key1].copy(), 1
    ):
        flag = "✓可缓存" if seg["static"] else "✗动态"
        print(f"  {i}. [{flag}] {seg['content']!r}")

    # 执行合并
    merged = tracker.merge_static_segments(key1)

    print(f"\n合并后的片段（{len(merged)}个）：")
    for i, seg in enumerate(merged, 1):
        flag = "✓可缓存" if seg["static"] else "✗动态"
        print(f"  {i}. [{flag}] {seg['content']!r}")

    summary1 = tracker.get_trace_summary(key1)
    print(f"\n缓存效率：{summary1['cache_efficiency']:.1%}")
    print(
        f"静态字符：{summary1['static_chars']}  "
        f"动态字符：{summary1['dynamic_chars'] if 'dynamic_chars' in summary1 else summary1['total_chars'] - summary1['static_chars']}"
    )

    # ── 演示2：Few-shot追踪（高缓存效率）────────────────────
    print()
    print("=" * 60)
    print("【演示2：Few-shot函数的片段追踪（高缓存效率）】")
    print("=" * 60)
    print()

    queries = ["这个产品太棒了！", "快递丢失，很生气。"]
    for query in queries:
        key = simulate_few_shot_trace(tracker, query)
        tracker.merge_static_segments(key)
        summary = tracker.get_trace_summary(key)
        print(
            f"  查询：{query!r:<15} "
            f"缓存效率：{summary['cache_efficiency']:.1%}"
        )

    # ── 演示3：多请求缓存复用量化 ─────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：多请求间静态片段复用量化】")
    print("=" * 60)
    print()

    texts = [
        "The quick brown fox",
        "Machine learning is transformative",
        "Natural language processing advances",
    ]

    print("对3个不同文本执行翻译追踪：\n")
    efficiency_list = []
    for text in texts:
        key = simulate_translation_trace(
            tracker, text, "中文"
        )
        tracker.merge_static_segments(key)
        eff = tracker.get_cache_efficiency(key)
        efficiency_list.append(eff)
        print(f"  文本：{text[:25]!r:<30} 效率：{eff:.1%}")

    avg_eff = sum(efficiency_list) / len(efficiency_list)
    print(f"\n  平均缓存效率：{avg_eff:.1%}")
    print(
        f"  含义：平均每次请求约{avg_eff:.0%}的内容"
        f"无需重新计算KV Cache"
    )

    # ── 演示4：合并前后性能对比 ───────────────────────────────
    print()
    print("=" * 60)
    print("【演示4：合并操作性能测试】")
    print("=" * 60)
    print()

    # 构造含大量细碎静态片段的追踪
    bench_tracker = RuntimePrefixTracker()
    key_b = bench_tracker.start_trace("bench", "000")
    for i in range(100):
        # 交替静态/动态片段
        bench_tracker.record_segment(
            key_b, f"静态片段{i} ", is_static=True
        )
    # 再插入一个动态片段打断
    bench_tracker.record_segment(
        key_b, "动态内容", is_static=False
    )
    for i in range(100):
        bench_tracker.record_segment(
            key_b, f"尾部静态{i} ", is_static=True
        )

    before_count = len(bench_tracker.traces[key_b])
    start = time.time()
    bench_tracker.merge_static_segments(key_b)
    elapsed = (time.time() - start) * 1000
    after_count = len(bench_tracker.traces[key_b])

    print(f"  合并前片段数：{before_count}")
    print(f"  合并后片段数：{after_count}")
    print(f"  合并耗时：{elapsed:.3f}ms")
    print(f"  片段压缩比：{before_count/after_count:.1f}x")


if __name__ == "__main__":
    main()
