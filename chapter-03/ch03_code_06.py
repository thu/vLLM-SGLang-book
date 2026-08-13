"""
第3章 代码示例 06
主题：SGLang函数组合与数据流转
场景：将翻译、摘要两个原子函数组合为工作流，
      演示子函数调用、结果提取、跨函数数据传递，
      以及函数作为可复用模块的工程价值

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time

import sglang as sgl

# 连接本地SGLang推理服务
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def translate(s, text: str, target_lang: str):
    # 独立的翻译模块，可单独测试，也可被任意工作流复用
    s += sgl.user(f"将以下文本翻译为{target_lang}：\n{text}")
    s += sgl.assistant(sgl.gen("translation", max_tokens=300))


@sgl.function
def summarize(s, text: str):
    s += sgl.user(f"为以下文本生成不超过50字的摘要：\n{text}")
    s += sgl.assistant(sgl.gen("summary", max_tokens=100))


@sgl.function
def translate_then_summarize(s, text: str, target_lang: str):
    # 运行子函数，同步等待结果
    trans_state = translate.run(text=text, target_lang=target_lang)
    translated = trans_state["translation"]

    # 将翻译结果传入摘要函数，实现函数间的数据流转
    summ_state = summarize.run(text=translated)

    s += sgl.user("翻译与摘要完成。")
    s += sgl.assistant(
        f"翻译：{translated}\n摘要：{summ_state['summary']}"
    )


@sgl.function
def extract_keywords(s, text: str, count: int = 5):
    """
    关键词提取模块：与翻译、摘要模块并列，
    可自由参与任意组合工作流。
    """
    s += sgl.user(
        f"从以下文本提取{count}个关键词，"
        f"以逗号分隔，只输出关键词：\n{text}"
    )
    s += sgl.assistant(
        sgl.gen("keywords", max_tokens=80, stop="\n")
    )


@sgl.function
def full_analysis(s, text: str, target_lang: str):
    """
    三步骤组合工作流：翻译 → 摘要 → 关键词提取。
    每个步骤均调用独立的原子函数，
    展示多函数串联的数据流编排能力。
    """
    # 步骤1：翻译
    trans_state = translate.run(
        text=text, target_lang=target_lang
    )
    translated = trans_state["translation"]

    # 步骤2：对译文生成摘要
    summ_state = summarize.run(text=translated)
    summary = summ_state["summary"]

    # 步骤3：从原文提取关键词
    kw_state = extract_keywords.run(text=text, count=5)
    keywords = kw_state["keywords"]

    s += sgl.user("全面分析完成，请整合输出。")
    s += sgl.assistant(
        f"【翻译】{translated[:100]}...\n"
        f"【摘要】{summary}\n"
        f"【关键词】{keywords}"
    )

    return {
        "translation": translated,
        "summary": summary,
        "keywords": keywords,
    }


def _print_composition_value():
    """打印函数组合的工程价值说明。"""
    print()
    print("=" * 60)
    print("【函数组合的工程价值】")
    print("=" * 60)
    print()
    print("可测试性：translate、summarize可独立单元测试，")
    print("  无需运行完整工作流即可验证各步骤正确性。")
    print()
    print("可复用性：translate函数同时服务于")
    print("  translate_then_summarize 和 full_analysis，")
    print("  修改一处，所有调用方自动受益。")
    print()
    print("可维护性：替换某个步骤的实现只需修改对应函数，")
    print("  组合逻辑代码无需改动。")
    print()
    print("数据流清晰：函数间通过Python变量显式传递数据，")
    print("  调试时可在任意步骤之间插入断点检查中间结果。")


def main():
    sample_text = (
        "大语言模型通过在海量文本上进行自监督学习，"
        "获得了强大的语言理解和生成能力。"
        "这些模型通过微调可以适应各种下游任务，"
        "包括文本分类、信息抽取和机器翻译等。"
    )

    print("=" * 60)
    print("【演示1：独立原子函数调用】")
    print("=" * 60)
    print()

    # 单独调用翻译函数
    start = time.time()
    t_state = translate.run(text=sample_text, target_lang="英文")
    print(f"翻译结果（前80字）：")
    print(f"  {t_state['translation'][:80]}...")
    print(f"  耗时：{(time.time()-start)*1000:.0f}ms")
    print()

    # 单独调用摘要函数
    start = time.time()
    s_state = summarize.run(text=sample_text)
    print(f"摘要结果：")
    print(f"  {s_state['summary']}")
    print(f"  耗时：{(time.time()-start)*1000:.0f}ms")
    print()

    print("=" * 60)
    print("【演示2：翻译+摘要组合工作流】")
    print("=" * 60)
    print()

    start = time.time()
    state = translate_then_summarize.run(
        text=sample_text, target_lang="英文"
    )
    elapsed = (time.time() - start) * 1000

    # 从组合函数状态中提取各步骤结果
    # assistant内容包含翻译和摘要，通过state.text()获取
    print(f"组合工作流输出：")
    print(f"  {state.text()[:200]}...")
    print(f"  总耗时：{elapsed:.0f}ms")
    print()

    print("=" * 60)
    print("【演示3：三步骤全面分析工作流】")
    print("=" * 60)
    print()

    start = time.time()
    state3 = full_analysis.run(
        text=sample_text, target_lang="英文"
    )
    elapsed3 = (time.time() - start) * 1000

    result = state3.ret_value
    print(f"翻译（前60字）：{result['translation'][:60]}...")
    print(f"摘要：{result['summary']}")
    print(f"关键词：{result['keywords']}")
    print(f"总耗时：{elapsed3:.0f}ms")

    print()
    print("=" * 60)
    print("【演示4：run_batch批量并发翻译】")
    print("=" * 60)
    print()

    texts = [
        "机器学习是人工智能的核心子领域。",
        "自然语言处理使计算机能够理解人类语言。",
        "深度学习通过多层神经网络学习特征表示。",
    ]

    print(f"并发翻译{len(texts)}条文本...\n")
    start = time.time()
    states = translate.run_batch(
        [{"text": t, "target_lang": "英文"} for t in texts],
        num_threads=3,
    )
    elapsed4 = (time.time() - start) * 1000

    for text, st in zip(texts, states):
        print(f"  原文：{text}")
        print(f"  译文：{st['translation'][:60]}...")
        print()

    print(f"批量总耗时：{elapsed4:.0f}ms")

    _print_composition_value()


if __name__ == "__main__":
    main()
