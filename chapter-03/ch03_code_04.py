"""
第3章 代码示例 04
主题：结构化输出与regex约束解码
场景：使用 build_regex_from_object 从数据类自动生成
      JSON结构约束，通过regex参数在生成阶段强制保证
      输出格式，彻底消除事后解析失败的可能性

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

注意：
    regex约束解码仅支持本地部署的模型，
    不支持通过OpenAI API调用的远程模型。
    build_regex_from_object 需要 outlines 库支持：
    pip install outlines
"""

import json
import time
from dataclasses import dataclass

import sglang as sgl
from sglang.lang.choices import build_regex_from_object

# 连接本地SGLang推理服务
sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@dataclass
class MovieInfo:
    title: str
    year: int
    rating: float


# 从数据类自动生成JSON结构约束的正则表达式
movie_regex = build_regex_from_object(MovieInfo)


@sgl.function
def recommend_movie(s, genre: str):
    s += sgl.user(
        f"推荐一部{genre}类型的经典电影，"
        f"以JSON格式返回片名、年份和评分："
    )
    s += sgl.assistant(
        # regex参数在每个生成步骤实时过滤非法token
        # 保证输出100%符合MovieInfo的JSON结构，无需事后验证
        sgl.gen("movie", max_tokens=100, regex=movie_regex)
    )
    # 结果直接解析，不再需要try/except防御
    return json.loads(s["movie"])


@sgl.function
def recommend_movie_batch_item(s, genre: str):
    """
    供run_batch调用的单条版本，
    返回raw字符串供外部统一解析。
    """
    s += sgl.user(
        f"推荐一部{genre}类型的经典电影，"
        f"以JSON格式返回片名、年份和评分："
    )
    s += sgl.assistant(
        sgl.gen("movie", max_tokens=100, regex=movie_regex)
    )


def _show_regex_value():
    """打印自动生成的regex，帮助读者理解其结构。"""
    print("自动生成的movie_regex：")
    # 截断显示，避免过长
    regex_str = str(movie_regex)
    if len(regex_str) > 120:
        print(f"  {regex_str[:120]}...")
    else:
        print(f"  {regex_str}")
    print()


def _show_contrast():
    """打印与传统方式的对比说明。"""
    print()
    print("=" * 60)
    print("【约束解码 vs 传统生成后解析对比】")
    print("=" * 60)
    print()
    print("传统方式（生成后解析）：")
    print("  1. 生成文本（可能包含markdown代码块）")
    print("  2. 剥离```json ... ```包裹")
    print("  3. json.loads()，格式偏差则抛异常")
    print("  4. 修改提示词重试，浪费计算资源")
    print("  成功率：通常65-80%")
    print()
    print("SGLang约束解码（生成即合法）：")
    print("  1. 生成时每个token经DFA实时过滤")
    print("  2. 输出100%符合目标JSON结构")
    print("  3. json.loads()直接调用，无需防御")
    print("  成功率：100%，零重试")


def main():
    print("=" * 60)
    print("【演示1：单条电影推荐（含regex值展示）】")
    print("=" * 60)
    print()
    _show_regex_value()

    genres = ["科幻", "悬疑", "喜剧"]
    for genre in genres:
        start = time.time()
        state = recommend_movie.run(genre=genre)
        elapsed = (time.time() - start) * 1000

        # ret_value为函数return的值（已解析的dict）
        movie = state.ret_value
        raw_json = state["movie"]

        print(f"类型：{genre}")
        print(f"  原始输出：{raw_json}")
        if isinstance(movie, dict):
            print(
                f"  解析结果：{movie.get('title', '?')}  "
                f"年份：{movie.get('year', '?')}  "
                f"评分：{movie.get('rating', '?')}"
            )
        print(f"  耗时：{elapsed:.0f}ms")
        print()

    print("=" * 60)
    print("【演示2：run_batch批量并发推荐】")
    print("=" * 60)
    print()

    batch_genres = ["动作", "爱情", "动画", "纪录片"]
    print(f"并发推荐{len(batch_genres)}个类型...\n")

    start = time.time()
    states = recommend_movie_batch_item.run_batch(
        [{"genre": g} for g in batch_genres],
        num_threads=4,
    )
    elapsed = (time.time() - start) * 1000

    for genre, st in zip(batch_genres, states):
        raw = st["movie"]
        try:
            movie = json.loads(raw)
            print(
                f"  {genre}：{movie.get('title', '?')} "
                f"({movie.get('year', '?')}) "
                f"★{movie.get('rating', '?')}"
            )
        except json.JSONDecodeError:
            print(f"  {genre}：解析失败 - {raw[:50]}")

    print(f"\n批量总耗时：{elapsed:.0f}ms")
    print(f"平均每条：{elapsed / len(batch_genres):.0f}ms")

    _show_contrast()


if __name__ == "__main__":
    main()
