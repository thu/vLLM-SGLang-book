"""
第3章 代码示例 32
主题：流式生成的应用层缓存回放策略
场景：对高频重复请求缓存token序列，
      命中时以高速回放实现「伪流式」效果，
      显著降低重复请求的首字延迟（TTFT）

说明：
    本示例演示缓存层的核心逻辑，
    无需启动SGLang服务可直接运行（使用模拟模式）。
    与SGLang集成时，将真实流式生成替换模拟生成即可。

运行前提：
    pip install sglang
"""

import hashlib
import time
from typing import Dict, Generator, Iterator, List, Optional, Tuple


class StreamCacheLayer:
    """
    流式生成的应用层缓存。
    首次生成时将token序列缓存，命中时回放缓存，
    实现接近零延迟的伪流式输出。
    """
    def __init__(self, max_entries: int = 200):
        # prompt哈希 -> token序列的映射
        self._cache: dict = {}
        # 访问顺序记录（用于LRU淘汰）
        self._access_order: List[str] = []
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def _cache_key(
        self, prompt: str, params: dict
    ) -> str:
        key_str = prompt + str(sorted(params.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[List[str]]:
        if key in self._cache:
            self._hits += 1
            self._access_order.append(key)
            return self._cache[key]
        self._misses += 1
        return None

    def store(self, key: str, tokens: List[str]):
        if len(self._cache) >= self.max_entries:
            # LRU淘汰：删除最久未访问的条目
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = tokens

    def replay_stream(
        self, tokens: List[str],
        replay_speed_multiplier: float = 10.0
    ) -> Iterator[str]:
        """
        以高于原始生成速度的速率回放缓存token序列。
        replay_speed_multiplier=10意味着回放速度是原始生成的10倍。
        """
        base_delay = 0.03  # 模拟原始生成的30ms/token
        replay_delay = base_delay / replay_speed_multiplier
        for token in tokens:
            time.sleep(replay_delay)
            yield token

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict:
        """返回缓存统计信息。"""
        total = self._hits + self._misses
        return {
            "total_queries": total,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": self.hit_rate,
            "cached_entries": len(self._cache),
            "max_entries": self.max_entries,
        }


# ── 模拟流式生成器（供演示使用）─────────────────────────────

def simulated_llm_stream(
    prompt: str,
    tokens_per_second: float = 33.0,
) -> Tuple[Iterator[str], float]:
    """
    模拟LLM流式生成，无需实际API调用。
    返回 (token迭代器, 首字延迟估算ms)。
    """
    # 根据prompt生成确定性的模拟回答
    answer = (
        f"这是对「{prompt[:20]}」的回答。"
        f"在真实场景中此处为LLM生成的内容，"
        f"通过缓存机制可将重复请求的TTFT降低至原始的1/{tokens_per_second:.0f}。"
    )
    tokens = list(answer)  # 按字符模拟token

    delay_per_token = 1.0 / tokens_per_second
    ttft_ms = delay_per_token * 1000  # 首字延迟

    def _stream():
        for token in tokens:
            time.sleep(delay_per_token)
            yield token

    return _stream(), ttft_ms


class CachedStreamingService:
    """
    整合StreamCacheLayer的流式生成服务。
    对外提供统一的 generate_stream() 接口，
    内部自动判断走真实生成路径还是缓存回放路径。
    """

    def __init__(
        self,
        cache: StreamCacheLayer,
        replay_speed: float = 10.0,
    ):
        self.cache = cache
        self.replay_speed = replay_speed

    def generate_stream(
        self,
        prompt: str,
        params: dict = None,
    ) -> Tuple[Iterator[str], bool]:
        """
        流式生成接口。
        返回 (token迭代器, 是否来自缓存)。
        调用方无需关心来源，只需消费迭代器。
        """
        params = params or {}
        key = self.cache._cache_key(prompt, params)
        cached_tokens = self.cache.get(key)

        if cached_tokens is not None:
            # 缓存命中：以高速回放
            return (
                self.cache.replay_stream(
                    cached_tokens,
                    replay_speed_multiplier=self.replay_speed,
                ),
                True,
            )

        # 缓存未命中：真实生成并缓存结果
        stream_iter, _ = simulated_llm_stream(prompt)
        collected_tokens: List[str] = []

        def _generate_and_cache() -> Iterator[str]:
            for token in stream_iter:
                collected_tokens.append(token)
                yield token
            # 生成完成后存入缓存
            self.cache.store(key, collected_tokens)

        return _generate_and_cache(), False


def main():
    # ── 演示1：StreamCacheLayer基本操作 ──────────────────────
    print("=" * 60)
    print("【演示1：StreamCacheLayer基本存取与回放】")
    print("=" * 60)
    print()

    cache = StreamCacheLayer(max_entries=5)

    # 存入一个token序列
    prompt = "什么是机器学习？"
    params = {"max_tokens": 200}
    key = cache._cache_key(prompt, params)
    mock_tokens = list("机器学习是让计算机从数据中自动学习规律的技术。")
    cache.store(key, mock_tokens)

    # 查询并回放
    retrieved = cache.get(key)
    print(f"存入{len(mock_tokens)}个token后查询：")
    print(f"  命中：{retrieved is not None}")
    print(f"  token数：{len(retrieved)}")
    print()

    # 回放演示
    print("回放（speed=10x）：", end="", flush=True)
    start = time.time()
    for token in cache.replay_stream(
        mock_tokens, replay_speed_multiplier=10.0
    ):
        print(token, end="", flush=True)
    replay_elapsed = (time.time() - start) * 1000
    print(f"\n回放耗时：{replay_elapsed:.0f}ms")

    # ── 演示2：缓存命中率与TTFT对比 ──────────────────────────
    print()
    print("=" * 60)
    print("【演示2：缓存命中率与TTFT延迟对比】")
    print("=" * 60)
    print()

    service = CachedStreamingService(
        cache=StreamCacheLayer(max_entries=20),
        replay_speed=8.0,
    )

    # 模拟FAQ场景：有限的高频问题集
    faq_questions = [
        "什么是机器学习？",
        "Python有哪些主要优点？",
        "什么是API？",
        "什么是机器学习？",   # 重复：应命中缓存
        "Python有哪些主要优点？",  # 重复：应命中缓存
        "什么是云计算？",     # 新问题
        "什么是机器学习？",   # 再次重复
    ]

    print(f"模拟{len(faq_questions)}次FAQ查询...\n")
    ttft_real_list = []
    ttft_cached_list = []

    for i, question in enumerate(faq_questions):
        params = {"max_tokens": 150}
        stream_iter, from_cache = service.generate_stream(
            prompt=question, params=params
        )

        # 测量首字延迟
        start = time.time()
        first_token = None
        all_tokens = []
        for token in stream_iter:
            if first_token is None:
                first_token = token
                ttft = (time.time() - start) * 1000
            all_tokens.append(token)
        total_ms = (time.time() - start) * 1000

        source = "✓缓存回放" if from_cache else "✗真实生成"
        print(
            f"  查询{i+1} [{source}] {question[:18]}"
            f"... TTFT={ttft:.0f}ms 总耗时={total_ms:.0f}ms"
        )

        if from_cache:
            ttft_cached_list.append(ttft)
        else:
            ttft_real_list.append(ttft)

    stats = service.cache.get_stats()
    print()
    print(f"缓存统计：")
    print(f"  总查询：{stats['total_queries']}")
    print(f"  命中：{stats['cache_hits']}  "
          f"命中率：{stats['hit_rate']:.1%}")
    print(f"  已缓存条目：{stats['cached_entries']}")

    if ttft_real_list and ttft_cached_list:
        avg_real = sum(ttft_real_list) / len(ttft_real_list)
        avg_cached = sum(ttft_cached_list) / len(ttft_cached_list)
        print()
        print(f"  真实生成平均TTFT：{avg_real:.0f}ms")
        print(f"  缓存回放平均TTFT：{avg_cached:.0f}ms")
        if avg_cached > 0:
            print(f"  TTFT加速比：{avg_real/avg_cached:.1f}x")

    # ── 演示3：LRU淘汰验证 ────────────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：LRU淘汰机制验证（max_entries=3）】")
    print("=" * 60)
    print()

    small_cache = StreamCacheLayer(max_entries=3)
    entries = ["问题A", "问题B", "问题C"]

    for q in entries:
        k = small_cache._cache_key(q, {})
        small_cache.store(k, list(f"答案：{q}"))
        print(f"  存入：{q}  当前条目数：{len(small_cache._cache)}")

    # 访问A，提升其LRU优先级
    k_a = small_cache._cache_key("问题A", {})
    small_cache.get(k_a)
    print(f"\n访问「问题A」（提升LRU优先级）")

    # 插入D，应淘汰最旧的B
    k_d = small_cache._cache_key("问题D", {})
    small_cache.store(k_d, list("答案：D"))
    print(f"插入「问题D」（应淘汰「问题B」）")

    k_b = small_cache._cache_key("问题B", {})
    b_hit = small_cache.get(k_b) is not None
    a_hit = small_cache.get(k_a) is not None
    print(f"\n验证：问题B命中={b_hit}（期望False）")
    print(f"验证：问题A命中={a_hit}（期望True）")

    print()
    print("=" * 60)
    print("【缓存回放策略适用性】")
    print("=" * 60)
    print()
    print("高适用性（命中率高）：FAQ系统、模板化内容生成")
    print("低适用性（命中率低）：个性化内容、创意生成")
    print()
    print("replay_speed_multiplier参数建议：")
    print("  5-10x ：保持流式视觉效果，用户可感知流式")
    print("  50x+  ：接近瞬时输出，极致低延迟场景")


if __name__ == "__main__":
    main()
