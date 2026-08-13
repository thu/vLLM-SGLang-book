"""
第3章 代码示例 07
主题：批处理与RadixAttention前缀缓存优化
场景：通过run_batch执行批量翻译请求，
      演示公共前缀KV Cache自动复用机制，
      并通过顺序执行与批量执行的耗时对比
      量化前缀缓存带来的实际性能收益

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

说明：
    系统提示词作为所有请求的公共前缀，
    run_batch执行时该部分的KV Cache只计算一次，
    后续请求直接复用，节省与前缀长度成正比的计算量。
"""

import time
from typing import List

import sglang as sgl

# 连接本地SGLang推理服务
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def batch_translate(s, text: str, target_lang: str):
    # 公共前缀：所有请求共享此系统指令
    # RadixAttention自动识别并复用该前缀的KV Cache
    s += sgl.system(
        f"你是专业翻译，将所有输入翻译为{target_lang}，"
        f"保持原文风格与语气。"
    )
    s += sgl.user(text)
    s += sgl.assistant(sgl.gen("translation", max_tokens=300))


@sgl.function
def few_shot_classify(s, text: str):
    """
    Few-shot分类示例：示例部分构成高比例公共前缀，
    批量处理时KV Cache复用收益尤为显著。
    """
    # 固定的Few-shot示例：所有请求共享，复用率极高
    s += (
        "以下是情感分类示例：\n\n"
        "输入：这款产品真的很好用！\n输出：积极\n\n"
        "输入：服务太差了，非常失望。\n输出：消极\n\n"
        "输入：产品一般，没什么特别。\n输出：中性\n\n"
    )
    s += f"输入：{text}\n输出："
    s += sgl.gen(
        "label",
        max_tokens=5,
        stop="\n",
        choices=["积极", "消极", "中性"],
    )


def sequential_translate(
    texts: List[str], target_lang: str
) -> List[str]:
    """顺序执行翻译，作为批量执行的对照组。"""
    results = []
    for text in texts:
        state = batch_translate.run(
            text=text, target_lang=target_lang
        )
        results.append(state["translation"])
    return results


def main():
    # ── 演示1：核心代码片段原样运行 ──────────────────────────
    print("=" * 60)
    print("【演示1：基础批量翻译（核心示例）】")
    print("=" * 60)
    print()

    # run_batch并发执行所有请求，公共前缀只计算一次
    texts = ["Hello world", "Good morning", "Thank you"]
    states = batch_translate.run_batch(
        [
            {"text": t, "target_lang": "中文"}
            for t in texts
        ],
        num_threads=4
    )
    for state in states:
        print(state["translation"])

    # ── 演示2：顺序 vs 批量性能对比 ──────────────────────────
    print()
    print("=" * 60)
    print("【演示2：顺序执行 vs 批量执行性能对比】")
    print("=" * 60)
    print()

    bench_texts = [
        "Deep learning has revolutionized computer vision.",
        "Natural language processing enables machines to understand text.",
        "Reinforcement learning trains agents through trial and error.",
        "Transfer learning reduces the need for large labeled datasets.",
        "Attention mechanisms allow models to focus on relevant context.",
    ]
    target_lang = "中文"

    print(f"待翻译文本数量：{len(bench_texts)} 条\n")

    # 顺序执行
    start = time.time()
    seq_results = sequential_translate(bench_texts, target_lang)
    seq_elapsed = (time.time() - start) * 1000
    print(f"顺序执行耗时：{seq_elapsed:.0f}ms")

    # 批量执行
    start = time.time()
    batch_states = batch_translate.run_batch(
        [{"text": t, "target_lang": target_lang}
         for t in bench_texts],
        num_threads=len(bench_texts),
    )
    batch_results = [st["translation"] for st in batch_states]
    batch_elapsed = (time.time() - start) * 1000
    print(f"批量执行耗时：{batch_elapsed:.0f}ms")

    if batch_elapsed > 0:
        speedup = seq_elapsed / batch_elapsed
        print(f"加速比：{speedup:.2f}x")

    print()
    print("翻译结果对比（前2条）：")
    for i in range(min(2, len(bench_texts))):
        print(f"  原文：{bench_texts[i][:50]}...")
        print(f"  顺序：{seq_results[i][:50]}...")
        print(f"  批量：{batch_results[i][:50]}...")
        print()

    # ── 演示3：Few-shot批量分类 ───────────────────────────────
    print("=" * 60)
    print("【演示3：Few-shot批量分类（高前缀复用率）】")
    print("=" * 60)
    print()

    reviews = [
        "这款手机拍照效果令人印象深刻，非常满意！",
        "物流太慢了，等了三周才收到，很失望。",
        "产品质量还可以，价格适中，没有特别惊喜。",
        "客服响应及时，问题很快得到解决，好评！",
        "包装有些简陋，产品本身功能正常。",
    ]

    print(f"并发分类{len(reviews)}条评论...\n")
    start = time.time()
    cls_states = few_shot_classify.run_batch(
        [{"text": r} for r in reviews],
        num_threads=len(reviews),
    )
    cls_elapsed = (time.time() - start) * 1000

    for review, st in zip(reviews, cls_states):
        print(f"  [{st['label']}] {review[:30]}...")

    print(f"\n批量分类耗时：{cls_elapsed:.0f}ms")
    print(f"平均每条：{cls_elapsed/len(reviews):.0f}ms")

    # ── RadixAttention原理说明 ────────────────────────────────
    print()
    print("=" * 60)
    print("【RadixAttention前缀缓存工作原理】")
    print("=" * 60)
    print()
    print("系统提示词（公共前缀）：")
    print("  批次中第1个请求：完整计算，结果存入基数树节点")
    print("  批次中第2-N个请求：直接复用节点KV Cache，")
    print("  仅对各自独立的用户输入部分执行Prefill计算")
    print()
    print("前缀越长、批量越大，缓存复用收益越显著：")
    print("  系统提示词100tokens + 批量100条")
    print("  → 节省约 99 × 100 = 9900 tokens 的Prefill计算")


if __name__ == "__main__":
    main()
