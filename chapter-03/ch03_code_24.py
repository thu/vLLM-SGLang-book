"""
第3章 代码示例 24
主题：命令式编程风格 vs SGLang声明式编程风格
场景：用翻译任务并排展示两种范式的本质差异，
      量化代码复杂度、性能和可维护性方面的差距

运行前提：
    pip install sglang openai
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
    export OPENAI_API_KEY="your-key"  # 命令式部分需要
"""

import os
import time
from typing import List

import sglang as sgl
from openai import OpenAI

# ── 命令式风格所需的OpenAI客户端 ─────────────────────────────
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "")
)

# ── SGLang后端 ────────────────────────────────────────────────
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


# ── 命令式风格（传统方式）────────────────────────────────────
def translate_imperative(text: str, target_lang: str) -> str:
    # 手动构造提示词：格式耦合在业务逻辑中
    prompt = f"将以下文本翻译为{target_lang}：\n\n{text}\n\n翻译："
    # 手动发起API调用：每次都重新发送完整提示词，无缓存
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    # 手动提取结果：了解响应对象的内部结构
    translation = response.choices[0].message.content.strip()
    if not translation:
        raise ValueError("翻译结果为空")
    return translation


# ── 声明式风格（SGLang方式）──────────────────────────────────
@sgl.function
def translate_declarative(s, text: str, target_lang: str):
    # 声明意图：翻译文本到目标语言
    # 框架自动处理：API调用、KV Cache复用、结果存储
    s += sgl.system("你是专业翻译，翻译准确自然，保持原文风格。")
    s += sgl.user(
        f"将以下文本翻译为{target_lang}：\n\n{text}"
    )
    # gen()声明生成意图，框架负责一切技术细节
    s += sgl.assistant(sgl.gen("translation", max_tokens=500))


# ── 命令式多步骤工作流（展示累积复杂度）─────────────────────
def multi_step_imperative(
    text: str, target_lang: str
) -> dict:
    """
    命令式三步骤：翻译 -> 摘要 -> 关键词提取
    每步独立API调用，上下文无法复用，
    代码随步骤数量线性膨胀。
    """
    # 步骤1：翻译
    prompt1 = f"将以下文本翻译为{target_lang}：\n{text}\n翻译："
    r1 = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt1}]
    )
    translation = r1.choices[0].message.content.strip()

    # 步骤2：对译文生成摘要
    prompt2 = f"为以下文本生成不超过50字的摘要：\n{translation}\n摘要："
    r2 = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt2}]
    )
    summary = r2.choices[0].message.content.strip()

    # 步骤3：提取关键词
    prompt3 = f"从以下文本提取5个关键词，逗号分隔：\n{text}\n关键词："
    r3 = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt3}]
    )
    keywords = r3.choices[0].message.content.strip()

    return {
        "translation": translation,
        "summary": summary,
        "keywords": keywords,
    }


# ── 声明式多步骤工作流 ────────────────────────────────────────
@sgl.function
def summarize_declarative(s, text: str, max_words: int = 50):
    """摘要原子函数。"""
    s += sgl.user(
        f"为以下文本生成不超过{max_words}字的摘要：\n{text}"
    )
    s += sgl.assistant(sgl.gen("summary", max_tokens=100))


@sgl.function
def extract_keywords_declarative(s, text: str):
    """关键词提取原子函数。"""
    s += sgl.user(
        f"从以下文本提取5个关键词，逗号分隔，只输出关键词：\n{text}"
    )
    s += sgl.assistant(
        sgl.gen("keywords", max_tokens=60, stop="\n")
    )


@sgl.function
def multi_step_declarative(s, text: str, target_lang: str):
    """
    声明式三步骤：每步调用独立原子函数，
    通过Python变量显式传递数据，逻辑清晰。
    """
    # 步骤1：翻译
    trans_state = translate_declarative.run(
        text=text, target_lang=target_lang
    )
    translation = trans_state["translation"]

    # 步骤2：对译文生成摘要
    summ_state = summarize_declarative.run(
        text=translation, max_words=50
    )
    summary = summ_state["summary"]

    # 步骤3：从原文提取关键词
    kw_state = extract_keywords_declarative.run(text=text)
    keywords = kw_state["keywords"]

    s += sgl.user("三步骤处理完成，请整合输出。")
    s += sgl.assistant(
        f"翻译：{translation[:80]}...\n"
        f"摘要：{summary}\n"
        f"关键词：{keywords}"
    )

    return {
        "translation": translation,
        "summary": summary,
        "keywords": keywords,
    }


def _print_paradigm_comparison():
    """打印两种范式的系统性对比。"""
    print()
    print("=" * 60)
    print("【命令式 vs 声明式编程范式对比】")
    print("=" * 60)
    print()
    print(f"  {'维度':<14} {'命令式（传统）':<20} {'声明式（SGLang）'}")
    print(f"  {'-'*14} {'-'*20} {'-'*18}")
    rows = [
        ("关注点",     "如何一步步实现",      "要什么结果"),
        ("API调用",    "手动管理每次调用",     "框架自动处理"),
        ("KV Cache",   "无，每次重新发送",     "RadixAttention自动复用"),
        ("结果提取",   "了解响应对象结构",     "变量名直接访问"),
        ("错误处理",   "手动try/except",       "框架内置容错"),
        ("性能优化",   "手动实现",             "自动批处理/缓存"),
        ("可测试性",   "端到端测试",           "每函数独立测试"),
    ]
    for dim, imp, dec in rows:
        print(f"  {dim:<14} {imp:<20} {dec}")


def main():
    sample_text = (
        "大语言模型通过在海量文本上进行自监督学习，"
        "获得了强大的语言理解和生成能力。"
    )

    # ── 演示1：单步翻译对比 ───────────────────────────────────
    print("=" * 60)
    print("【演示1：单步翻译——命令式 vs 声明式】")
    print("=" * 60)
    print()

    # 声明式（SGLang）
    print("声明式（SGLang）：")
    start = time.time()
    state = translate_declarative.run(
        text=sample_text, target_lang="英文"
    )
    sgl_elapsed = (time.time() - start) * 1000
    print(f"  结果：{state['translation'][:70]}...")
    print(f"  耗时：{sgl_elapsed:.0f}ms")
    print()

    # 命令式（如有API密钥）
    if os.environ.get("OPENAI_API_KEY"):
        print("命令式（OpenAI API）：")
        start = time.time()
        try:
            result = translate_imperative(sample_text, "英文")
            imp_elapsed = (time.time() - start) * 1000
            print(f"  结果：{result[:70]}...")
            print(f"  耗时：{imp_elapsed:.0f}ms")
        except Exception as e:
            print(f"  错误：{e}")
    else:
        print("命令式（未配置OPENAI_API_KEY，跳过）")

    # ── 演示2：三步骤工作流对比 ──────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：三步骤工作流——声明式实现】")
    print("=" * 60)
    print()

    start = time.time()
    state3 = multi_step_declarative.run(
        text=sample_text, target_lang="英文"
    )
    elapsed3 = (time.time() - start) * 1000

    result = state3.ret_value
    print(f"翻译（前60字）：{result['translation'][:60]}...")
    print(f"摘要：{result['summary']}")
    print(f"关键词：{result['keywords']}")
    print(f"三步骤总耗时：{elapsed3:.0f}ms")

    # ── 演示3：run_batch批量并发 ──────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：run_batch批量并发翻译】")
    print("=" * 60)
    print()

    texts = [
        "机器学习是人工智能的核心子领域。",
        "自然语言处理使计算机能够理解人类语言。",
        "深度学习通过多层神经网络学习特征表示。",
    ]

    print(f"并发翻译{len(texts)}条文本...\n")
    start = time.time()
    states = translate_declarative.run_batch(
        [{"text": t, "target_lang": "英文"} for t in texts],
        num_threads=3,
    )
    batch_elapsed = (time.time() - start) * 1000

    for text, st in zip(texts, states):
        print(f"  原：{text}")
        print(f"  译：{st['translation'][:60]}...")
        print()

    print(f"批量总耗时：{batch_elapsed:.0f}ms")
    print(f"平均每条：{batch_elapsed/len(texts):.0f}ms")

    _print_paradigm_comparison()


if __name__ == "__main__":
    main()
