"""
第3章 代码示例 26
主题：通用处理管道与高阶函数模式
场景：将SGLang函数作为参数传入通用管道，
      实现策略可插拔的处理链，
      演示函数作为一等公民的工程价值

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time
from typing import Callable, Dict, List, Optional

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


# ── 原子处理函数（管道的基本单元）────────────────────────────

@sgl.function
def step_translate(s, text: str, target_lang: str = "英文"):
    """翻译步骤：输出存入约定键名 'result'。"""
    s += sgl.system("你是专业翻译，翻译准确自然。")
    s += sgl.user(
        f"将以下文本翻译为{target_lang}，只输出译文：\n{text}"
    )
    s += sgl.assistant(sgl.gen("result", max_tokens=400))


@sgl.function
def step_summarize(s, text: str, max_words: int = 50):
    """摘要步骤：输出存入约定键名 'result'。"""
    s += sgl.user(
        f"为以下文本生成不超过{max_words}字的摘要，"
        f"只输出摘要：\n{text}"
    )
    s += sgl.assistant(sgl.gen("result", max_tokens=150))


@sgl.function
def step_expand(s, text: str, expand_words: int = 150):
    """扩写步骤：输出存入约定键名 'result'。"""
    s += sgl.user(
        f"将以下内容扩写到约{expand_words}字，"
        f"保持核心意思，只输出扩写内容：\n{text}"
    )
    s += sgl.assistant(sgl.gen("result", max_tokens=300))


@sgl.function
def step_formalize(s, text: str):
    """正式化步骤：将口语化内容改为书面语。"""
    s += sgl.user(
        f"将以下内容改写为正式书面语，只输出改写结果：\n{text}"
    )
    s += sgl.assistant(sgl.gen("result", max_tokens=300))


# ── 通用处理管道 ──────────────────────────────────────────────

@sgl.function
def processing_pipeline(
    s, text: str,
    steps: List[Callable],  # SGLang函数列表
):
    """
    通用处理管道：将任意SGLang函数序列依次应用于输入文本。
    steps中的每个函数接收上一步的输出作为输入，
    形成有向数据流图。
    """
    current_text = text
    results = {}

    for i, step_fn in enumerate(steps):
        # 调用当前步骤函数
        step_state = step_fn.run(text=current_text)
        # 尝试提取约定的输出变量名
        for key in ["result", "output", "translation",
                    "summary", "analysis"]:
            if key in step_state.get_var_dict():
                current_text = step_state[key]
                results[f"step_{i}_{step_fn.__name__}"] = (
                    current_text
                )
                break

    s += sgl.user(f"处理完成，原始输入：{text[:50]}...")
    s += sgl.assistant(f"最终输出：{current_text}")
    return results


# ── 函数工厂（偏函数化）──────────────────────────────────────

def make_translator(target_lang: str):
    """
    函数工厂：生成针对特定目标语言的专用翻译函数。
    偏函数化：固定部分参数，生成特化版本。
    """
    @sgl.function
    def specialized_translator(s, text: str):
        s += sgl.system(
            f"你是专业的{target_lang}翻译，翻译准确自然。"
        )
        s += sgl.user(
            f"翻译为{target_lang}，只输出译文：\n{text}"
        )
        s += sgl.assistant(
            sgl.gen("result", max_tokens=400)
        )

    specialized_translator.__name__ = (
        f"translate_to_{target_lang.replace(' ', '_')}"
    )
    return specialized_translator


def make_summarizer(max_words: int):
    """生成固定字数上限的摘要函数。"""
    @sgl.function
    def specialized_summarizer(s, text: str):
        s += sgl.user(
            f"生成不超过{max_words}字的摘要，只输出摘要：\n{text}"
        )
        s += sgl.assistant(
            sgl.gen("result", max_tokens=200)
        )

    specialized_summarizer.__name__ = (
        f"summarize_{max_words}words"
    )
    return specialized_summarizer


def main():
    sample_text = (
        "大语言模型通过在海量文本上进行自监督学习，"
        "获得了强大的语言理解和生成能力。"
        "这些模型在各类自然语言处理任务上都表现出色，"
        "正在深刻改变人机交互的方式。"
    )

    # ── 演示1：单步管道 ───────────────────────────────────────
    print("=" * 60)
    print("【演示1：单步管道（仅翻译）】")
    print("=" * 60)
    print()

    translate_en = make_translator("英文")
    state = processing_pipeline.run(
        text=sample_text,
        steps=[translate_en],
    )
    results = state.ret_value
    print(f"输入：{sample_text[:40]}...")
    print()
    for step_name, output in results.items():
        print(f"[{step_name}]")
        print(f"  {output[:70]}...")
    print()

    # ── 演示2：两步管道（翻译→摘要）────────────────────────
    print("=" * 60)
    print("【演示2：两步管道（翻译→摘要）】")
    print("=" * 60)
    print()

    summarize_30 = make_summarizer(30)
    start = time.time()
    state2 = processing_pipeline.run(
        text=sample_text,
        steps=[translate_en, summarize_30],
    )
    elapsed2 = (time.time() - start) * 1000

    results2 = state2.ret_value
    print(f"管道步骤：翻译（→英文）→ 摘要（≤30字）")
    print(f"总耗时：{elapsed2:.0f}ms\n")
    for step_name, output in results2.items():
        print(f"[{step_name}]")
        print(f"  {output[:80]}...")
        print()

    # ── 演示3：三步管道（翻译→正式化→摘要）────────────────
    print("=" * 60)
    print("【演示3：三步管道（翻译→正式化→摘要）】")
    print("=" * 60)
    print()

    start = time.time()
    state3 = processing_pipeline.run(
        text=sample_text,
        steps=[translate_en, step_formalize, summarize_30],
    )
    elapsed3 = (time.time() - start) * 1000

    results3 = state3.ret_value
    print(f"管道步骤：翻译 → 正式化 → 摘要")
    print(f"总耗时：{elapsed3:.0f}ms\n")
    for step_name, output in results3.items():
        print(f"[{step_name}]")
        print(f"  {output[:80]}...")
        print()

    # ── 演示4：策略可替换性 ───────────────────────────────────
    print("=" * 60)
    print("【演示4：策略可替换性——同一管道不同语言】")
    print("=" * 60)
    print()

    short_text = "人工智能正在改变世界。"
    for lang in ["英文", "日文", "法文"]:
        translator = make_translator(lang)
        state_lang = processing_pipeline.run(
            text=short_text,
            steps=[translator],
        )
        r = state_lang.ret_value
        output = list(r.values())[0] if r else "N/A"
        print(f"  {lang}：{output[:50]}...")

    # ── 高阶函数的工程价值 ────────────────────────────────────
    print()
    print("=" * 60)
    print("【高阶函数管道的工程价值】")
    print("=" * 60)
    print()
    print("开闭原则：新增处理步骤只需传入新函数，")
    print("  无需修改 processing_pipeline 本身。")
    print()
    print("策略模式：运行时动态组合处理步骤，")
    print("  不同场景使用不同的 steps 列表。")
    print()
    print("函数工厂：偏函数化复用，减少重复参数传递，")
    print("  make_translator('日文') 生成专用版本。")
    print()
    print("可测试性：每个原子步骤可独立测试，")
    print("  管道测试只需验证步骤间的数据流转。")


if __name__ == "__main__":
    main()
