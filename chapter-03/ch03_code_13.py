"""
第3章 代码示例 13
主题：参数化前缀缓存——模板骨架复用优化
场景：对含变量插值的模板前缀提取骨架并缓存，
      骨架相同的不同参数化请求共享同一份KV Cache，
      减少重复的Prefill计算

说明：
    本示例为教学性实现，展示参数化缓存的核心逻辑。
    骨架缓存在真实系统中对应GPU显存中的KV Cache块。
    无需启动SGLang服务，可直接运行。

运行前提：
    pip install sglang
"""

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple


class ParameterizedPrefixCache:
    def __init__(self):
        # 骨架哈希值 -> 骨架KV Cache的映射
        self.skeleton_cache: Dict[str, Any] = {}
        # 访问统计
        self._hits = 0
        self._misses = 0

    def extract_skeleton(
        self, template: str
    ) -> Tuple[str, list]:
        """
        从模板字符串中提取骨架与变量列表。
        例："翻译{text}为{lang}" ->
            骨架："翻译___为___"，变量：["text","lang"]
        """
        variables = re.findall(r'\{(\w+)\}', template)
        # 将所有变量替换为统一占位符，得到可跨请求共享的骨架
        skeleton = re.sub(r'\{\w+\}', '___', template)
        return skeleton, variables

    def get_skeleton_hash(self, skeleton: str) -> str:
        return hashlib.md5(skeleton.encode()).hexdigest()

    def should_use_cache(self, template: str) -> bool:
        """判断该模板骨架是否已有缓存可复用。"""
        skeleton, _ = self.extract_skeleton(template)
        key = self.get_skeleton_hash(skeleton)
        return key in self.skeleton_cache

    def store_skeleton(
        self, template: str, kv_cache: Any
    ):
        """首次计算后将骨架KV Cache存入缓存。"""
        skeleton, _ = self.extract_skeleton(template)
        key = self.get_skeleton_hash(skeleton)
        self.skeleton_cache[key] = kv_cache

    def get_skeleton_cache(
        self, template: str
    ) -> Optional[Any]:
        """
        查询骨架缓存。
        命中时更新统计并返回缓存值，未命中返回None。
        """
        skeleton, _ = self.extract_skeleton(template)
        key = self.get_skeleton_hash(skeleton)
        if key in self.skeleton_cache:
            self._hits += 1
            return self.skeleton_cache[key]
        self._misses += 1
        return None

    def get_stats(self) -> Dict:
        """返回缓存命中率统计。"""
        total = self._hits + self._misses
        return {
            "total_queries": total,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cached_skeletons": len(self.skeleton_cache),
        }


def simulate_requests(
    cache: ParameterizedPrefixCache,
    templates: List[str],
) -> List[Dict]:
    """
    模拟一批请求的参数化缓存处理流程：
      首次请求：完整计算骨架KV Cache并存储
      后续相同骨架请求：直接复用，只需增量计算变量部分
    """
    results = []
    for template in templates:
        skeleton, variables = cache.extract_skeleton(template)
        cached_kv = cache.get_skeleton_cache(template)

        if cached_kv is not None:
            # 缓存命中：只需计算变量部分的增量KV
            # 节省的计算量 ≈ 骨架长度对应的Prefill计算
            saved_chars = len(skeleton.replace('___', ''))
            results.append({
                "template": template[:50],
                "cache_hit": True,
                "skeleton": skeleton[:40],
                "variables": variables,
                "saved_chars": saved_chars,
            })
        else:
            # 缓存未命中：完整计算后存储骨架KV Cache
            mock_kv = f"kv_for_{cache.get_skeleton_hash(skeleton)[:8]}"
            cache.store_skeleton(template, mock_kv)
            results.append({
                "template": template[:50],
                "cache_hit": False,
                "skeleton": skeleton[:40],
                "variables": variables,
                "saved_chars": 0,
            })

    return results


def main():
    cache = ParameterizedPrefixCache()

    # ── 演示1：骨架提取详情 ───────────────────────────────────
    print("=" * 60)
    print("【演示1：模板骨架提取详情】")
    print("=" * 60)
    print()

    test_templates = [
        "将{text}翻译为{lang}，保持原文风格",
        "你是专业{role}，请帮我{task}",
        "评估以下{content_type}的质量（1-10分）：\n{content}",
        "将以下内容总结为不超过{max_words}字：\n{text}",
    ]

    for tmpl in test_templates:
        skeleton, variables = cache.extract_skeleton(tmpl)
        sk_hash = cache.get_skeleton_hash(skeleton)
        print(f"模板：{tmpl}")
        print(f"  骨架：{skeleton}")
        print(f"  变量：{variables}")
        print(f"  哈希：{sk_hash[:16]}...")
        print()

    # ── 演示2：同骨架不同参数的缓存复用 ──────────────────────
    print("=" * 60)
    print("【演示2：同骨架不同参数的缓存命中演示】")
    print("=" * 60)
    print()

    # 这些模板具有相同的骨架："将___翻译为___，保持原文风格"
    translation_requests = [
        "将{text}翻译为{lang}，保持原文风格",   # 首次：未命中
        "将{text}翻译为{lang}，保持原文风格",   # 重复：命中
        "将{article}翻译为{target}，保持原文风格",  # 变量名不同但骨架相同：命中
        "请用{style}风格改写：{content}",        # 不同骨架：未命中
        "请用{tone}风格改写：{text}",            # 骨架同上一条：命中
    ]

    results = simulate_requests(cache, translation_requests)

    for r in results:
        hit_str = "✓命中" if r["cache_hit"] else "✗未命中"
        saved_str = (
            f"，节省约{r['saved_chars']}字符的计算"
            if r["cache_hit"] else "，完整计算并缓存骨架"
        )
        print(f"  [{hit_str}] {r['template']}{saved_str}")

    stats = cache.get_stats()
    print()
    print(f"缓存统计：")
    print(f"  总查询：{stats['total_queries']}")
    print(f"  命中：{stats['cache_hits']}")
    print(f"  命中率：{stats['hit_rate']:.1%}")
    print(f"  已缓存骨架数：{stats['cached_skeletons']}")

    # ── 演示3：大批量请求的节省量化 ──────────────────────────
    print()
    print("=" * 60)
    print("【演示3：大批量请求节省量化（100条）】")
    print("=" * 60)
    print()

    # 模拟100条翻译请求（相同骨架，不同文本内容）
    bulk_cache = ParameterizedPrefixCache()
    bulk_templates = [
        f"将{{text_{i}}}翻译为{{lang}}，保持原文风格"
        if i > 0
        else "将{text}翻译为{lang}，保持原文风格"
        for i in range(100)
    ]

    # 前几条变量名不同但骨架完全一致
    # 实际上所有条目骨架均为"将___翻译为___，保持原文风格"
    bulk_results = simulate_requests(bulk_cache, bulk_templates)

    total_saved = sum(r["saved_chars"] for r in bulk_results)
    hit_count = sum(1 for r in bulk_results if r["cache_hit"])

    bulk_stats = bulk_cache.get_stats()
    print(f"  请求总数：{len(bulk_templates)}")
    print(f"  缓存命中：{hit_count} 条")
    print(f"  命中率：{bulk_stats['hit_rate']:.1%}")
    print(f"  节省字符总量：{total_saved}")
    print(
        f"  含义：{hit_count}条请求无需重复计算"
        f"骨架对应的Prefill"
    )

    # ── 演示4：should_use_cache判断 ───────────────────────────
    print()
    print("=" * 60)
    print("【演示4：should_use_cache预判断】")
    print("=" * 60)
    print()

    check_cache = ParameterizedPrefixCache()
    # 先存入一个骨架
    check_cache.store_skeleton(
        "将{text}翻译为{lang}", "mock_kv"
    )

    queries = [
        "将{article}翻译为{target}",  # 骨架相同，应返回True
        "请总结{content}（{max_words}字）",  # 新骨架，返回False
        "将{doc}翻译为{language}",    # 骨架相同，应返回True
    ]

    for q in queries:
        can_reuse = check_cache.should_use_cache(q)
        skeleton, _ = check_cache.extract_skeleton(q)
        print(
            f"  {'✓可复用' if can_reuse else '✗需计算'}  "
            f"骨架：{skeleton!r}"
        )

    print()
    print("=" * 60)
    print("【参数化缓存的适用场景】")
    print("=" * 60)
    print()
    print("高收益场景（骨架重复率高）：")
    print("  批量翻译（相同指令，不同文本）")
    print("  批量分类（相同Few-shot，不同查询）")
    print("  模板化内容生成（格式固定，内容可变）")
    print()
    print("低收益场景（骨架重复率低）：")
    print("  高度个性化的提示词")
    print("  动态构建的复杂多步骤提示")


if __name__ == "__main__":
    main()
