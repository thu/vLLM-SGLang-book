"""
第3章 代码示例 29
主题：流式生成基础API——同步与异步两种消费模式
场景：演示SGLang的流式生成机制，
      对比批量模式与流式模式的首字延迟（TTFT）差异，
      并展示适用于FastAPI/WebSocket的异步流式用法

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

说明：
    流式生成通过将首字延迟（TTFT）与总完成延迟（TTC）
    解耦，在不改变实际计算量的情况下显著改善用户体验。
    人类感知响应的阈值约100-500ms，流式生成充分利用
    这一特性，让用户在生成完成前就看到内容。
"""

import asyncio
import sys
import time
from typing import Optional

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def streaming_response(s, question: str):
    s += sgl.system("你是一位知识渊博的助手。")
    s += sgl.user(question)
    # stream=True使生成结果以迭代器形式逐步输出
    # 首个token生成后即可开始传递，无需等待全部完成
    s += sgl.assistant(
        sgl.gen("answer", max_tokens=300, stream=True)
    )


# ── 同步流式消费 ─────────────────────────────────────────────
def sync_streaming_demo(question: str):
    state = streaming_response.run(
        question=question, stream=True
    )
    print("助手：", end="", flush=True)
    # text_iter()返回增量文本片段的迭代器
    # 每次迭代获取新生成的若干token对应的文本
    for chunk in state.text_iter("answer"):
        print(chunk, end="", flush=True)
    print()  # 换行


# ── 异步流式消费（适用于FastAPI/WebSocket）───────────────────
@sgl.function
async def async_streaming_response(s, prompt: str):
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen("response", max_tokens=300, stream=True)
    )


async def async_streaming_demo(prompt: str):
    state = await async_streaming_response.run_async(
        prompt=prompt, stream=True
    )
    # 异步迭代器：与Python async for语法无缝集成
    async for chunk in state.text_iter_async("response"):
        # 在实际应用中，此处将chunk推送给WebSocket客户端
        print(chunk, end="", flush=True)
    print()


# ── 非流式对照（用于延迟对比）────────────────────────────────

@sgl.function
def non_streaming_response(s, question: str):
    """非流式版本，用于与流式版本对比首字延迟。"""
    s += sgl.system("你是一位知识渊博的助手。")
    s += sgl.user(question)
    s += sgl.assistant(
        sgl.gen("answer", max_tokens=300, stream=False)
    )


def measure_ttft_and_ttc(
    question: str,
) -> dict:
    """
    测量流式生成的首字延迟（TTFT）和总完成延迟（TTC），
    同时测量非流式的完成延迟作为对照。
    """
    # 非流式：用户看到第一个字的时间 = 全部完成的时间
    start = time.time()
    non_stream_state = non_streaming_response.run(
        question=question
    )
    non_stream_ttc = (time.time() - start) * 1000
    answer_len = len(non_stream_state["answer"])

    # 流式：记录首字时间和全部完成时间
    start = time.time()
    first_token_time: Optional[float] = None
    chunks_collected = []

    stream_state = streaming_response.run(
        question=question, stream=True
    )
    for chunk in stream_state.text_iter("answer"):
        if first_token_time is None and chunk.strip():
            first_token_time = time.time()
        chunks_collected.append(chunk)

    stream_ttc = (time.time() - start) * 1000
    stream_ttft = (
        (first_token_time - start) * 1000
        if first_token_time else stream_ttc
    )

    return {
        "question_preview": question[:30] + "...",
        "answer_chars": answer_len,
        "non_stream_wait_ms": non_stream_ttc,
        "stream_ttft_ms": stream_ttft,
        "stream_ttc_ms": stream_ttc,
        "ttft_improvement": non_stream_ttc / stream_ttft
        if stream_ttft > 0 else 0,
    }


# ── SSE格式的异步流式推送（模拟Web服务场景）──────────────────

async def simulate_sse_push(prompt: str):
    """
    模拟Server-Sent Events（SSE）端点的推送行为。
    在真实FastAPI应用中，此函数对应StreamingResponse。
    """
    print(f"\n[SSE] 开始流式推送...")
    chunk_count = 0

    state = await async_streaming_response.run_async(
        prompt=prompt, stream=True
    )
    async for chunk in state.text_iter_async("response"):
        # SSE格式：每条消息以"data: "开头，双换行结尾
        # 前端EventSource API可直接解析此格式
        sse_line = f"data: {chunk}"
        print(sse_line, end="", flush=True)
        chunk_count += 1

    # 推送结束信号
    print("\ndata: [DONE]\n")
    print(f"[SSE] 共推送 {chunk_count} 个chunk")


def main():
    question = (
        "请简要解释Transformer中的自注意力机制，"
        "包括Query、Key、Value的作用。"
    )

    # ── 演示1：同步流式输出 ───────────────────────────────────
    print("=" * 60)
    print("【演示1：同步流式输出】")
    print("=" * 60)
    print(f"\n问：{question}\n")

    sync_streaming_demo(question)

    # ── 演示2：TTFT vs TTC 延迟对比 ──────────────────────────
    print()
    print("=" * 60)
    print("【演示2：首字延迟（TTFT）vs 完成延迟（TTC）对比】")
    print("=" * 60)
    print()

    metrics = measure_ttft_and_ttc(question)

    print(f"问题：{metrics['question_preview']}")
    print(f"回答字符数：{metrics['answer_chars']}")
    print()
    print(
        f"非流式模式：用户需等待 {metrics['non_stream_wait_ms']:.0f}ms "
        f"才看到第一个字"
    )
    print(
        f"流式模式  ：{metrics['stream_ttft_ms']:.0f}ms 看到首字，"
        f"{metrics['stream_ttc_ms']:.0f}ms 全部完成"
    )
    if metrics["ttft_improvement"] > 0:
        print(
            f"\n首字延迟改善：{metrics['ttft_improvement']:.1f}x"
            f"（{metrics['non_stream_wait_ms']:.0f}ms → "
            f"{metrics['stream_ttft_ms']:.0f}ms）"
        )

    # ── 演示3：异步流式（SSE格式模拟）────────────────────────
    print()
    print("=" * 60)
    print("【演示3：异步流式——SSE格式推送模拟】")
    print("=" * 60)

    asyncio.run(
        simulate_sse_push("用一句话解释什么是机器学习")
    )

    # ── 演示4：异步流式直接消费 ───────────────────────────────
    print()
    print("=" * 60)
    print("【演示4：异步流式直接消费】")
    print("=" * 60)
    print()

    print("异步流式输出：", end="", flush=True)
    asyncio.run(
        async_streaming_demo("用一句话介绍深度学习")
    )

    # ── 流式生成核心价值 ──────────────────────────────────────
    print()
    print("=" * 60)
    print("【流式生成核心价值】")
    print("=" * 60)
    print()
    print("TTFT（首字延迟）：影响用户感知响应速度")
    print("TTC （完成延迟）：影响任务完成时间")
    print()
    print("流式生成：TTFT大幅降低，TTC不变，")
    print("  在不改变实际计算量的前提下改善用户体验。")
    print()
    print("同步流式：适合命令行工具、批处理脚本")
    print("异步流式：适合FastAPI、WebSocket、SSE服务")
    print()
    print("text_iter()    ：同步增量文本迭代器")
    print("text_iter_async()：异步增量文本迭代器")


if __name__ == "__main__":
    main()
