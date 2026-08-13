"""
scripts/chapter8/benchmark_response_latency.py
响应延迟基准测试（三场景）。

场景一: 冷启动 - 服务重启后首批请求
场景二: 多轮热身 - 7 轮完整对话（缓存逐步生效）
场景三: 5 并发  - 5 个会话同时发起请求

测试环境: A100 80GB, ChatGLM-3-6B BF16, SGLang v0.1.17
通过标准: 所有场景 P95 首 Token 延迟 <500ms
用法: python scripts/chapter8/benchmark_response_latency.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.dialogue_manager import DialogueManager
from src.core.session_manager import SessionManager
from src.core.streaming_manager import StreamingManager
from src.tools import ToolManager

SGLANG_URL = "http://127.0.0.1:30000"


@dataclass
class LatencyMeasurement:
    query: str
    first_token_ms: float
    total_ms: float
    scenario: str


async def measure_single_request(
    streaming_mgr: StreamingManager,
    user_input: str,
    system_prompt: str = "你是旅游助手，请简洁回答用户问题。",
    history: Optional[list] = None,
) -> LatencyMeasurement:
    """测量单次请求的首 Token 延迟和总响应时间。"""
    start = time.perf_counter()
    first_token_time: Optional[float] = None

    try:
        async for _ in streaming_mgr.stream_response(
            system_prompt=system_prompt,
            user_input=user_input,
            history_messages=history or [],
            max_tokens=150,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            # 只需首 Token，不等待完整响应
            if first_token_time is not None:
                break
    except Exception:
        pass

    total = time.perf_counter() - start
    return LatencyMeasurement(
        query=user_input,
        first_token_ms=(first_token_time or total) * 1000,
        total_ms=total * 1000,
        scenario="",
    )


def _percentile_stats(values: List[float]) -> dict:
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


# ── 场景一: 冷启动 ────────────────────────────────────────────────────

async def benchmark_cold_start(
    streaming_mgr: StreamingManager,
    http_client: httpx.AsyncClient,
    n_trials: int = 3,
) -> dict:
    """
    冷启动场景：每次测量前清空 KV Cache，模拟服务重启后的首批请求。
    """
    print("\n" + "─" * 55)
    print("场景一: 冷启动（每次前清空 KV Cache）")
    print("─" * 55)

    test_queries = [
        "你好",
        "我想去旅游",
        "推荐一下云南的景点",
        "帮我查一下昆明的天气",
        "给我规划一个5天的云南行程",
    ]

    all_ttft: List[float] = []

    for trial in range(n_trials):
        for q in test_queries:
            # 冷启动：每次前清空缓存
            await http_client.post(f"{SGLANG_URL}/flush_cache", timeout=5)
            m = await measure_single_request(streaming_mgr, q)
            all_ttft.append(m.first_token_ms)
            print(f"  [{trial+1}/{n_trials}] {q[:30]:<30} TTFT={m.first_token_ms:.0f}ms")

    stats = _percentile_stats(all_ttft)
    target_met = stats["p95"] < 500
    print(f"\n  P50={stats['p50']:.0f}ms P95={stats['p95']:.0f}ms P99={stats['p99']:.0f}ms")
    print(f"  目标 P95<500ms: {'✓' if target_met else '✗'}")
    return {"scenario": "cold_start", **stats, "target_met": target_met}


# ── 场景二: 多轮热身 ──────────────────────────────────────────────────

async def benchmark_multi_turn(
    streaming_mgr: StreamingManager,
    dialogue_mgr: DialogueManager,
    session_mgr: SessionManager,
) -> dict:
    """多轮对话场景，观察缓存逐步生效后的延迟变化趋势。"""
    print("\n" + "─" * 55)
    print("场景二: 多轮对话（7轮，缓存逐步生效）")
    print("─" * 55)

    dialogue_turns = [
        "我想去云南旅游",
        "下个月初出发，玩五天",
        "两个人，预算一万",
        "住舒适型酒店",
        "帮我查一下天气",
        "推荐几个景点",
        "生成完整旅游方案",
    ]

    state = session_mgr.create_session(user_id="bench_multiturn")
    per_turn_ttft: List[float] = []

    for turn_idx, user_input in enumerate(dialogue_turns, 1):
        system_prompt = dialogue_mgr.build_system_prompt(state, "")
        window = dialogue_mgr.context_mgr.get_optimal_context(state)
        history = dialogue_mgr.context_mgr.build_history_messages(window)

        m = await measure_single_request(
            streaming_mgr, user_input, system_prompt, history
        )
        per_turn_ttft.append(m.first_token_ms)

        cold_flag = " (冷启动)" if turn_idx == 1 else ""
        print(f"  第{turn_idx}轮{cold_flag:<6} TTFT={m.first_token_ms:.0f}ms")

        # 更新状态
        state.add_turn("user", user_input)
        state.add_turn("assistant", "（测试响应）")
        session_mgr.save_state(state)

    warm_ttft = per_turn_ttft[1:]  # 排除第1轮冷启动
    stats = _percentile_stats(warm_ttft)
    improvement = (per_turn_ttft[0] - stats["mean"]) / per_turn_ttft[0] * 100

    print(f"\n  第1轮（冷启动）: {per_turn_ttft[0]:.0f}ms")
    print(f"  第2-7轮均值:     {stats['mean']:.0f}ms")
    print(f"  P95（第2-7轮）:  {stats['p95']:.0f}ms")
    print(f"  缓存带来的延迟降低: {improvement:.1f}%")

    target_met = stats["p95"] < 500
    print(f"  目标 P95<500ms: {'✓' if target_met else '✗'}")

    return {
        "scenario": "multi_turn",
        "cold_start_ms": per_turn_ttft[0],
        "warm_mean_ms": stats["mean"],
        "p95_ms": stats["p95"],
        "improvement_pct": improvement,
        "target_met": target_met,
    }


# ── 场景三: 5 并发 ────────────────────────────────────────────────────

async def benchmark_concurrent(
    streaming_mgr: StreamingManager,
    n_concurrent: int = 5,
    n_trials: int = 5,
) -> dict:
    """5 个并发请求，观察并发对首 Token 延迟的影响。"""
    print("\n" + "─" * 55)
    print(f"场景三: {n_concurrent} 并发请求（{n_trials} 次重复）")
    print("─" * 55)

    query = "推荐一下云南旅游的景点"
    all_ttft: List[float] = []

    for trial in range(n_trials):
        tasks = [
            measure_single_request(streaming_mgr, query)
            for _ in range(n_concurrent)
        ]
        measurements = await asyncio.gather(*tasks)
        ttfts = [m.first_token_ms for m in measurements]
        all_ttft.extend(ttfts)
        avg = np.mean(ttfts)
        max_v = max(ttfts)
        print(f"  试验 {trial+1}/{n_trials}: 均值={avg:.0f}ms 最大={max_v:.0f}ms")

    stats = _percentile_stats(all_ttft)
    target_met = stats["p95"] < 500

    print(f"\n  P50={stats['p50']:.0f}ms P95={stats['p95']:.0f}ms P99={stats['p99']:.0f}ms")
    print(f"  目标 P95<500ms: {'✓' if target_met else '✗'}")

    return {
        "scenario": f"concurrent_{n_concurrent}",
        **stats,
        "target_met": target_met,
        "n_concurrent": n_concurrent,
    }


async def main():
    print("=" * 55)
    print("  响应延迟基准测试")
    print("  测试环境: A100 80GB, ChatGLM-3-6B BF16, SGLang v0.1.17")
    print("=" * 55)

    tool_mgr = ToolManager(use_mock=True)
    session_mgr = SessionManager()
    dialogue_mgr = DialogueManager(tool_manager=tool_mgr)
    streaming_mgr = StreamingManager()

    async with httpx.AsyncClient(timeout=10.0) as http_client:
        try:
            resp = await http_client.get(f"{SGLANG_URL}/health")
            assert resp.status_code == 200
        except Exception as e:
            print(f"✗ SGLang 服务未就绪: {e}")
            sys.exit(1)
        print("✓ SGLang 服务正常")

        r1 = await benchmark_cold_start(streaming_mgr, http_client)
        r2 = await benchmark_multi_turn(streaming_mgr, dialogue_mgr, session_mgr)
        r3 = await benchmark_concurrent(streaming_mgr)

    # 汇总
    all_results = [r1, r2, r3]
    all_met = all(r.get("target_met", False) for r in all_results)

    print("\n" + "=" * 55)
    print("  基准测试汇总")
    print("=" * 55)
    p95_map = {
        "冷启动": r1.get("p95", 0),
        "多轮热身": r2.get("p95_ms", 0),
        "5并发": r3.get("p95", 0),
    }
    for name, p95 in p95_map.items():
        icon = "✓" if p95 < 500 else "✗"
        print(f"  {icon} {name}: P95 = {p95:.0f}ms")

    final = "✓ 全部达标" if all_met else "✗ 部分未达标"
    print(f"\n  综合结论（P95 <500ms）: {final}")

    output = Path("results/response_latency_benchmark.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {"cold_start": r1, "multi_turn": r2, "concurrent": r3},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n  详细结果已保存: {output}")
    await tool_mgr.close_all()


if __name__ == "__main__":
    asyncio.run(main())
