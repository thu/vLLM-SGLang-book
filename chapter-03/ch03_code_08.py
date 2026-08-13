"""
第3章 代码示例 08
主题：RadixAttention核心数据结构——基数树（Radix Tree）
场景：实现基数树的节点定义与公共前缀长度计算，
      这是RadixAttention自动KV Cache复用的底层基础

说明：
    本示例为教学性实现，用于理解RadixAttention的内部原理。
    SGLang生产环境中的实现在C++层面，性能更高。
    本示例无需启动SGLang服务，可直接运行。

运行前提：
    pip install sglang
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RadixNode:
    # 当前节点存储的token序列片段
    prefix: List[int]
    # 子节点字典，以首token为键进行索引
    children: Dict[int, "RadixNode"] = field(
        default_factory=dict
    )
    # 指向GPU显存中KV Cache的引用，None表示未缓存
    kv_cache_ref: Optional[object] = None
    # 引用计数，降为0时方可被LRU淘汰机制释放
    ref_count: int = 0
    # 最近访问时间戳，用于LRU排序
    last_access_time: float = 0.0

    def is_leaf(self) -> bool:
        """判断是否为叶子节点（无子节点）。"""
        return len(self.children) == 0

    def touch(self):
        """更新访问时间戳，在LRU排序中提升优先级。"""
        self.last_access_time = time.time()


class RadixTree:
    def __init__(self):
        # 根节点前缀为空序列
        self.root = RadixNode(prefix=[])
        # 追踪树中存储的总token数
        self.total_cached_tokens = 0

    def _common_prefix_length(
        self, p1: List[int], p2: List[int]
    ) -> int:
        # 逐位比较两个token序列，返回公共前缀的长度
        for i in range(min(len(p1), len(p2))):
            if p1[i] != p2[i]:
                return i
        return min(len(p1), len(p2))

    def find(
        self, tokens: List[int]
    ) -> Tuple["RadixNode", int]:
        """
        在基数树中查找与tokens最长匹配的前缀节点。
        返回(匹配节点, 已匹配的token数量)。
        时间复杂度：O(L)，L为tokens的长度。
        """
        node = self.root
        matched_len = 0
        remaining = tokens

        while remaining:
            first_token = remaining[0]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            common_len = self._common_prefix_length(
                child.prefix, remaining
            )
            if common_len == 0:
                break
            if common_len < len(child.prefix):
                # 部分匹配：tokens在child.prefix中间分叉
                break

            # 完全匹配child.prefix：进入子节点继续匹配
            child.touch()
            node = child
            matched_len += common_len
            remaining = remaining[common_len:]

        return node, matched_len

    def insert(
        self, tokens: List[int], kv_cache: object = None
    ) -> Tuple["RadixNode", int]:
        """
        将tokens插入基数树，关联KV Cache引用。
        返回(叶子节点, 插入前已匹配的前缀长度)。
        """
        node = self.root
        remaining = tokens
        matched_len = 0

        while remaining:
            first_token = remaining[0]

            if first_token not in node.children:
                # 无匹配子节点：直接创建新叶子节点
                new_node = RadixNode(
                    prefix=list(remaining),
                    kv_cache_ref=kv_cache,
                )
                new_node.touch()
                node.children[first_token] = new_node
                self.total_cached_tokens += len(remaining)
                return new_node, matched_len

            child = node.children[first_token]
            common_len = self._common_prefix_length(
                child.prefix, remaining
            )

            if common_len == len(child.prefix):
                # 完全匹配：沿child继续向下插入
                matched_len += common_len
                node = child
                remaining = remaining[common_len:]
                child.touch()
            else:
                # 部分匹配：在common_len处分裂节点
                mid_node = RadixNode(
                    prefix=list(child.prefix[:common_len])
                )
                mid_node.touch()
                child.prefix = child.prefix[common_len:]
                mid_node.children[child.prefix[0]] = child
                node.children[first_token] = mid_node

                matched_len += common_len
                remaining = remaining[common_len:]

                if not remaining:
                    mid_node.kv_cache_ref = kv_cache
                    return mid_node, matched_len

                new_node = RadixNode(
                    prefix=list(remaining),
                    kv_cache_ref=kv_cache,
                )
                new_node.touch()
                mid_node.children[remaining[0]] = new_node
                self.total_cached_tokens += len(remaining)
                return new_node, matched_len

        # remaining为空：tokens与node完全匹配
        node.kv_cache_ref = kv_cache
        node.touch()
        return node, matched_len

    def get_stats(self) -> Dict:
        """返回基数树的统计信息。"""
        node_count = [0]
        self._count_nodes(self.root, node_count)
        return {
            "total_nodes": node_count[0],
            "total_cached_tokens": self.total_cached_tokens,
            "root_children": len(self.root.children),
        }

    def _count_nodes(
        self, node: "RadixNode", count: List[int]
    ):
        count[0] += 1
        for child in node.children.values():
            self._count_nodes(child, count)


def main():
    print("=" * 60)
    print("【RadixNode与RadixTree基本功能验证】")
    print("=" * 60)
    print()

    # ── 演示1：节点基本属性 ───────────────────────────────────
    print("演示1：RadixNode基本属性")
    node = RadixNode(prefix=[1, 2, 3])
    print(f"  前缀：{node.prefix}")
    print(f"  是否叶子节点：{node.is_leaf()}")
    print(f"  KV Cache引用：{node.kv_cache_ref}")
    print(f"  引用计数：{node.ref_count}")
    node.touch()
    print(f"  更新时间戳后：{node.last_access_time:.4f}")
    print()

    # ── 演示2：公共前缀长度计算 ───────────────────────────────
    print("演示2：_common_prefix_length计算")
    tree = RadixTree()
    cases = [
        ([1, 2, 3, 4], [1, 2, 5, 6], 2),
        ([1, 2, 3],    [1, 2, 3],    3),
        ([1, 2, 3],    [4, 5, 6],    0),
        ([1, 2],       [1, 2, 3, 4], 2),
    ]
    for p1, p2, expected in cases:
        result = tree._common_prefix_length(p1, p2)
        status = "✓" if result == expected else "✗"
        print(
            f"  {status} {p1} vs {p2} "
            f"→ 公共前缀长度={result}（期望{expected}）"
        )
    print()

    # ── 演示3：插入与查找 ─────────────────────────────────────
    print("演示3：插入与查找操作")
    # 模拟：[1,2,3,4,5]为系统提示词，后续为各用户输入
    system_tokens = [1, 2, 3, 4, 5]
    user1_tokens = system_tokens + [10, 11]
    user2_tokens = system_tokens + [10, 12]
    user3_tokens = system_tokens + [20, 21]
    diff_tokens  = [99, 98, 97]

    tree2 = RadixTree()
    for name, toks in [
        ("user1", user1_tokens),
        ("user2", user2_tokens),
        ("user3", user3_tokens),
        ("diff",  diff_tokens),
    ]:
        node, pre = tree2.insert(toks, kv_cache=f"kv_{name}")
        print(
            f"  插入{name}：插入前匹配={pre}，"
            f"节点prefix={node.prefix}"
        )

    print()
    stats = tree2.get_stats()
    print(f"  树统计：节点数={stats['total_nodes']}，"
          f"缓存token数={stats['total_cached_tokens']}")
    print()

    # ── 演示4：查找测试 ───────────────────────────────────────
    print("演示4：查找测试（模拟新请求到达）")
    test_cases = [
        ("与user1完全相同", user1_tokens),
        ("系统提示词+新输入", system_tokens + [30, 31]),
        ("完全不同", [88, 87, 86]),
    ]
    for desc, toks in test_cases:
        found_node, matched = tree2.find(toks)
        ratio = matched / len(toks) * 100
        print(
            f"  {desc}："
            f"匹配={matched}/{len(toks)}token "
            f"（复用率{ratio:.0f}%）"
        )

    # ── 演示5：性能测试 ───────────────────────────────────────
    print()
    print("演示5：1000次插入+查找性能测试")
    base = list(range(50))  # 50个token的公共前缀
    perf_tree = RadixTree()

    start = time.time()
    for i in range(1000):
        toks = base + [1000 + i, 1001 + i]
        perf_tree.insert(toks, kv_cache=f"cache_{i}")
    insert_elapsed = (time.time() - start) * 1000

    start = time.time()
    for i in range(1000):
        toks = base + [1000 + i, 1001 + i]
        perf_tree.find(toks)
    find_elapsed = (time.time() - start) * 1000

    perf_stats = perf_tree.get_stats()
    print(f"  1000次插入耗时：{insert_elapsed:.2f}ms")
    print(f"  1000次查找耗时：{find_elapsed:.2f}ms")
    print(f"  平均每次查找：{find_elapsed/1000:.3f}ms")
    print(f"  树节点总数：{perf_stats['total_nodes']}")
    print(f"  O(L)复杂度，L≈52，实测验证合理")


if __name__ == "__main__":
    main()
