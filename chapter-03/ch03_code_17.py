"""
第3章 代码示例 17
主题：约束缓存机制——将每步约束检查从O(V)降至O(1)
场景：对FSM各状态的合法token集合进行缓存，
      避免每个生成步骤重复扫描完整词汇表，
      量化有无缓存的性能差异

说明：
    本示例依赖 ch03_code_16.py 中的 JSONConstraintDecoder
    和 MockTokenizer。无需启动SGLang服务，可直接运行。

运行前提：
    pip install sglang
    确保 ch03_code_16.py 在同一目录下。
"""

import sys
import os
import time
from functools import lru_cache
from typing import FrozenSet, Optional, Set, Tuple

# 复用 ch03_code_16 中的 JSONConstraintDecoder 和 MockTokenizer
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from ch03_code_16 import (
        JSONConstraintDecoder,
        JSONState,
        MockTokenizer,
    )
except ImportError:
    raise ImportError(
        "请确保 ch03_code_16.py 在同一目录下，"
        "或将 JSONConstraintDecoder/MockTokenizer 复制到本文件。"
    )


class ConstraintCache:
    def __init__(self):
        # (状态哈希, 约束类型) -> 允许token集合
        # 相同FSM状态下约束结果必然相同，可安全跨请求复用
        self._cache: dict = {}
        self._hits = 0
        self._misses = 0

    @lru_cache(maxsize=10000)
    def get_allowed_tokens_cached(
        self,
        state_hash: int,
        constraint_type: str
    ) -> Tuple[int, ...]:
        """
        缓存版的合法token查询。
        首次调用时计算并缓存，后续直接返回。
        将约束检查开销从O(V)降低到O(1)。
        """
        key = (state_hash, constraint_type)
        if key in self._cache:
            # 直接返回缓存结果，跳过O(V)全词表扫描
            return tuple(self._cache[key])
        # 未命中时返回空元组，调用方执行完整计算后再存入
        return ()

    def store(
        self,
        state_hash: int,
        constraint_type: str,
        allowed: set
    ):
        key = (state_hash, constraint_type)
        self._cache[key] = allowed
        # lru_cache依赖参数，存入后需使缓存失效以便下次命中
        self.get_allowed_tokens_cached.cache_clear()

    def get(
        self, state_hash: int, constraint_type: str
    ) -> Optional[FrozenSet[int]]:
        """
        查询缓存，命中返回frozenset，未命中返回None。
        同时更新命中/未命中统计。
        """
        result = self.get_allowed_tokens_cached(
            state_hash, constraint_type
        )
        if result:
            self._hits += 1
            return frozenset(result)
        self._misses += 1
        return None

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "total_lookups": total,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cached_states": len(self._cache),
        }


def simulate_generation_no_cache(
    decoder: JSONConstraintDecoder,
    states_sequence: list,
    n_steps: int,
) -> float:
    """
    模拟无缓存的约束检查：每步都全量计算。
    返回总耗时（毫秒）。
    """
    start = time.perf_counter()
    for i in range(n_steps):
        state = states_sequence[i % len(states_sequence)]
        decoder.state_stack = [state]
        decoder.get_allowed_tokens()
    return (time.perf_counter() - start) * 1000


def simulate_generation_with_cache(
    decoder: JSONConstraintDecoder,
    cache: ConstraintCache,
    states_sequence: list,
    n_steps: int,
    constraint_type: str = "json",
) -> float:
    """
    模拟有缓存的约束检查：首次计算后直接查表。
    返回总耗时（毫秒）。
    """
    start = time.perf_counter()
    for i in range(n_steps):
        state = states_sequence[i % len(states_sequence)]
        decoder.state_stack = [state]
        state_hash = decoder.get_state_hash()

        cached = cache.get(state_hash, constraint_type)
        if cached is None:
            # 缓存未命中：完整计算并存入
            allowed = decoder.get_allowed_tokens()
            cache.store(state_hash, constraint_type, allowed)
        # else：缓存命中，直接使用 cached，O(1)

    return (time.perf_counter() - start) * 1000


def main():
    tokenizer = MockTokenizer()

    # ── 演示1：ConstraintCache基本用法 ───────────────────────
    print("=" * 60)
    print("【演示1：ConstraintCache基本存取操作】")
    print("=" * 60)
    print()

    cache = ConstraintCache()
    decoder = JSONConstraintDecoder(tokenizer)

    # 首次计算并存入缓存
    test_states = [
        JSONState.OBJECT_VALUE,
        JSONState.OBJECT_COMMA,
        JSONState.OBJECT_START,
        JSONState.OBJECT_COLON,
    ]

    for state in test_states:
        decoder.state_stack = [state]
        state_hash = decoder.get_state_hash()

        # 首次：未命中，完整计算
        result = cache.get(state_hash, "json")
        if result is None:
            allowed = decoder.get_allowed_tokens()
            cache.store(state_hash, "json", allowed)
            print(
                f"  [{state.value:14s}] 首次计算，"
                f"存入{len(allowed)}个合法token"
            )
        else:
            print(
                f"  [{state.value:14s}] 缓存命中，"
                f"{len(result)}个合法token"
            )

        # 再次查询：应命中
        result2 = cache.get(state_hash, "json")
        status = "✓命中" if result2 is not None else "✗未命中"
        print(f"    重复查询：{status}")

    stats = cache.get_stats()
    print(f"\n缓存统计：命中{stats['cache_hits']}次，"
          f"未命中{stats['cache_misses']}次，"
          f"命中率{stats['hit_rate']:.1%}")

    # ── 演示2：O(V) vs O(1) 性能对比 ─────────────────────────
    print()
    print("=" * 60)
    print("【演示2：无缓存O(V) vs 有缓存O(1) 性能对比】")
    print("=" * 60)
    print()

    # 模拟200步生成，4种状态循环出现
    states_seq = [
        JSONState.OBJECT_VALUE,
        JSONState.OBJECT_COMMA,
        JSONState.OBJECT_START,
        JSONState.OBJECT_COLON,
    ] * 50  # 200步

    N = 200
    decoder2 = JSONConstraintDecoder(tokenizer)

    # 无缓存
    time_no_cache = simulate_generation_no_cache(
        decoder2, states_seq, N
    )

    # 有缓存
    cache2 = ConstraintCache()
    decoder3 = JSONConstraintDecoder(tokenizer)
    time_with_cache = simulate_generation_with_cache(
        decoder3, cache2, states_seq, N
    )

    cache2_stats = cache2.get_stats()
    print(f"模拟{N}步生成（{len(set(s.value for s in states_seq))}种不同状态）：")
    print(f"  无缓存（每步O(V)扫描）：{time_no_cache:.3f}ms")
    print(f"  有缓存（O(1)查表）   ：{time_with_cache:.3f}ms")
    if time_with_cache > 0:
        speedup = time_no_cache / time_with_cache
        print(f"  加速比：{speedup:.1f}x")
    print(f"\n缓存统计：")
    print(f"  总查询：{cache2_stats['total_lookups']}")
    print(f"  命中：{cache2_stats['cache_hits']}")
    print(f"  命中率：{cache2_stats['hit_rate']:.1%}")
    print(
        f"  已缓存状态数：{cache2_stats['cached_states']}"
        f"（仅{cache2_stats['cached_states']}个状态覆盖了"
        f"{cache2_stats['hit_rate']:.0%}的查询）"
    )

    # ── 演示3：跨请求复用验证 ─────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：跨请求缓存复用验证】")
    print("=" * 60)
    print()

    # 模拟100个独立请求，每个请求包含相同的状态序列
    shared_cache = ConstraintCache()
    decoder4 = JSONConstraintDecoder(tokenizer)
    short_seq = [
        JSONState.OBJECT_VALUE,
        JSONState.OBJECT_COMMA,
    ]

    print("模拟100个请求，每个请求包含2个生成步骤：\n")
    start = time.perf_counter()
    for req_id in range(100):
        for state in short_seq:
            decoder4.state_stack = [state]
            h = decoder4.get_state_hash()
            cached = shared_cache.get(h, "json")
            if cached is None:
                allowed = decoder4.get_allowed_tokens()
                shared_cache.store(h, "json", allowed)
    total_ms = (time.perf_counter() - start) * 1000

    shared_stats = shared_cache.get_stats()
    print(f"  总耗时：{total_ms:.3f}ms")
    print(f"  总查询次数：{shared_stats['total_lookups']}")
    print(f"  实际计算次数（未命中）："
          f"{shared_stats['cache_misses']}")
    print(f"  缓存复用次数（命中）："
          f"{shared_stats['cache_hits']}")
    print(f"  命中率：{shared_stats['hit_rate']:.1%}")
    print(
        f"\n  含义：200次查询中仅需"
        f"{shared_stats['cache_misses']}次实际计算，"
        f"其余{shared_stats['cache_hits']}次直接从缓存读取。"
    )


if __name__ == "__main__":
    main()
