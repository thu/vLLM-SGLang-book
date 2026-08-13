"""
第3章 代码示例 14
主题：前缀感知批量调度器
场景：将一批请求按共享的前缀节点分组，
      优先调度缓存命中率高的请求，
      最大化批次内的KV Cache复用效率

说明：
    本示例依赖 ch03_code_08.py 中的 RadixTree 实现。
    无需启动SGLang服务，可直接运行。
    调度策略在真实系统中位于SGLang运行时的调度层。

运行前提：
    pip install sglang
    确保 ch03_code_08.py 在同一目录下，或将
    RadixNode/RadixTree 定义复制到本文件。
"""

import sys
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

# 复用 ch03_code_08 中的 RadixTree
# 若导入失败则使用内联简化版本
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from ch03_code_08 import RadixTree, RadixNode
except ImportError:
    # 内联简化版 RadixTree，仅保留 find 接口供调度器使用
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class RadixNode:
        prefix: List[int]
        children: Dict = field(default_factory=dict)
        kv_cache_ref: Optional[object] = None

    class RadixTree:
        def __init__(self):
            self.root = RadixNode(prefix=[])

        def find(self, tokens):
            return self.root, 0

        def insert(self, tokens, kv_cache=None):
            return self.root, 0


class PrefixAwareBatchScheduler:
    def __init__(self, radix_tree):
        self.tree = radix_tree

    def group_by_prefix(
        self, requests: List[Tuple[int, List[int]]]
    ) -> List[List[Tuple[int, List[int]]]]:
        """
        将请求按共享的前缀节点分组。
        映射到同一树节点的请求共享相同的KV Cache，
        调度到同一批次可最大化缓存命中率。
        """
        groups = defaultdict(list)
        for req_id, tokens in requests:
            node, matched = self.tree.find(tokens)
            # 以节点内存地址为键：相同节点表示共享前缀缓存
            groups[id(node)].append(
                (req_id, tokens, matched)
            )
        batches = []
        for group in groups.values():
            # 组内按匹配长度降序排列
            # 缓存命中更多的请求优先执行，
            # 其KV Cache可供后续请求进一步复用
            group.sort(key=lambda x: x[2], reverse=True)
            # 每批最多8个请求，控制单批显存占用
            for i in range(0, len(group), 8):
                batches.append([
                    (rid, toks)
                    for rid, toks, _ in group[i:i+8]
                ])
        return batches

    def compute_batch_efficiency(
        self, batch: List[Tuple[int, List[int]]]
    ) -> Dict:
        """
        计算一个批次的缓存复用效率。
        返回命中token数、总token数和复用率。
        """
        total_tokens = 0
        cached_tokens = 0
        for req_id, tokens in batch:
            _, matched = self.tree.find(tokens)
            total_tokens += len(tokens)
            cached_tokens += matched
        return {
            "batch_size": len(batch),
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "new_tokens": total_tokens - cached_tokens,
            "cache_ratio": (
                cached_tokens / total_tokens
                if total_tokens > 0 else 0.0
            ),
        }

    def schedule_with_stats(
        self, requests: List[Tuple[int, List[int]]]
    ) -> Tuple[List[List[Tuple[int, List[int]]]], Dict]:
        """
        执行调度并返回统计信息。
        """
        batches = self.group_by_prefix(requests)
        efficiencies = [
            self.compute_batch_efficiency(b) for b in batches
        ]
        avg_ratio = (
            sum(e["cache_ratio"] for e in efficiencies)
            / len(efficiencies)
            if efficiencies else 0.0
        )
        total_cached = sum(e["cached_tokens"] for e in efficiencies)
        total_tokens = sum(e["total_tokens"] for e in efficiencies)
        return batches, {
            "num_batches": len(batches),
            "avg_cache_ratio": avg_ratio,
            "overall_cache_ratio": (
                total_cached / total_tokens
                if total_tokens > 0 else 0.0
            ),
            "batch_efficiencies": efficiencies,
        }


def main():
    # 构建基数树并预填充一些缓存
    tree = RadixTree()

    # 模拟系统提示词（tokens 1-10）作为公共前缀
    system_prompt = list(range(1, 11))
    # 模拟三类请求各自的上下文前缀
    group_a_prefix = system_prompt + [100, 101]  # 翻译类
    group_b_prefix = system_prompt + [200, 201]  # 摘要类
    group_c_prefix = system_prompt + [300, 301]  # 分类类

    # 向树中插入已缓存的前缀
    for prefix, name in [
        (system_prompt,   "system_prompt"),
        (group_a_prefix,  "translate_ctx"),
        (group_b_prefix,  "summarize_ctx"),
    ]:
        tree.insert(prefix, kv_cache=f"kv_{name}")

    scheduler = PrefixAwareBatchScheduler(radix_tree=tree)

    # ── 演示1：基础分组调度 ───────────────────────────────────
    print("=" * 60)
    print("【演示1：前缀感知分组调度】")
    print("=" * 60)
    print()

    requests = [
        (1,  group_a_prefix + [10, 11]),  # 翻译类请求1
        (2,  group_b_prefix + [20, 21]),  # 摘要类请求1
        (3,  group_a_prefix + [10, 12]),  # 翻译类请求2
        (4,  group_c_prefix + [30, 31]),  # 分类类请求1（无缓存）
        (5,  group_b_prefix + [20, 22]),  # 摘要类请求2
        (6,  group_a_prefix + [10, 13]),  # 翻译类请求3
        (7,  list(range(50, 55)),         # 独立请求（无公共前缀）
        (8,  group_c_prefix + [30, 32]),  # 分类类请求2
    ]

    batches, stats = scheduler.schedule_with_stats(requests)

    print(f"请求总数：{len(requests)}")
    print(f"调度后批次数：{stats['num_batches']}")
    print(f"整体缓存复用率：{stats['overall_cache_ratio']:.1%}")
    print()

    for i, (batch, eff) in enumerate(
        zip(batches, stats["batch_efficiencies"])
    ):
        req_ids = [rid for rid, _ in batch]
        print(
            f"  批次{i+1}：请求{req_ids}  "
            f"缓存率={eff['cache_ratio']:.1%}  "
            f"（命中{eff['cached_tokens']}/"
            f"{eff['total_tokens']}tokens）"
        )

    # ── 演示2：对比无感知调度 ─────────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：无感知调度 vs 前缀感知调度对比】")
    print("=" * 60)
    print()

    # 无感知调度：所有请求放入同一批次，不考虑前缀
    naive_batch = requests  # 全部请求一批
    naive_eff = scheduler.compute_batch_efficiency(naive_batch)

    print(f"无感知调度（单一批次）：")
    print(f"  缓存复用率：{naive_eff['cache_ratio']:.1%}")
    print(f"  需计算token数：{naive_eff['new_tokens']}")
    print()
    print(f"前缀感知调度（{stats['num_batches']}个批次）：")
    print(f"  平均缓存复用率：{stats['avg_cache_ratio']:.1%}")
    total_new = sum(
        e["new_tokens"]
        for e in stats["batch_efficiencies"]
    )
    print(f"  需计算token数：{total_new}")

    # ── 演示3：max_batch_size的影响 ───────────────────────────
    print()
    print("=" * 60)
    print("【演示3：批次大小上限对调度结果的影响】")
    print("=" * 60)
    print()

    # 构造16条请求，均来自同一个前缀组
    large_group_requests = [
        (i, group_a_prefix + [500 + i])
        for i in range(16)
    ]

    # 当前实现：每批最多8条
    scheduler2 = PrefixAwareBatchScheduler(radix_tree=tree)
    batches2, stats2 = scheduler2.schedule_with_stats(
        large_group_requests
    )
    print(f"16条同组请求，每批上限8条：")
    print(f"  生成批次数：{stats2['num_batches']}")
    for i, b in enumerate(batches2):
        print(f"  批次{i+1}：{len(b)}条请求")

    print()
    print("=" * 60)
    print("【调度策略说明】")
    print("=" * 60)
    print()
    print("分组原则：共享同一RadixTree节点的请求归为一组，")
    print("  同组请求可复用该节点的KV Cache。")
    print()
    print("排序原则：组内按缓存命中长度降序排列，")
    print("  命中最多的请求先执行，其结果可供后续请求复用。")
    print()
    print("批大小上限：每批最多8条请求，")
    print("  防止单批显存占用过大导致OOM。")


if __name__ == "__main__":
    main()
