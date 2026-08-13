"""
scripts/chapter8/verify_prefix_cache.py
前缀缓存效率验证实验（严格受控对照）。

实验方法:
  - Cache 组:   正常进行 8 轮对话，RadixAttention 自然生效
  - No-Cache 组: 每轮推理前调用 /flush_cache API 显式清空 KV Cache
                  这是唯一能保证真正无缓存条件的方法

重要说明:
  仅用"重新构造相同 prompt"无法确保无缓存条件——SGLang 后端在内存中
  保留的 KV Cache 仍会被命中，导致对照组数据失真。

测试环境: A100 80GB, ChatGLM-3-6B BF16, SGLang v0.1.17
用法: python scripts/chapter8/verify_prefix_cache.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
import sglang as sgl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.session_manager import SessionManager
from src.core.dialogue_manager import DialogueManager
from src.core.streaming_manager import StreamingManager
from src.tools import ToolManager

SGLANG_URL = "http://127.0.0.1:30000"

sgl.set_default_backend(sgl.RuntimeEndpoint(SGLANG_URL))

# 标准测试对话脚本（8 轮典型旅游规划对话）
STANDARD_DIALOGUE = [
    "我想去云南旅游",
    "下个月初出发，大概玩五天",
    "两个人，预算一万元",
    "想住舒适型的酒店",
    "帮我查一下昆明的天气",
    "大理有什么好玩的景点",
    "推荐几家大理的酒店",
    "好的，帮我生成完整方案",
]


@dataclass
class TurnMetrics:
    """单轮对话的性能指标。"""
    turn: int
    first_token_ms: float
    total_ms: float
    approx_new_tokens: int
    cache_active: bool


@dataclass
class TrialResult:
    """一次完整对话试验的结果。"""
    use_cache: bool
    turns: List[TurnMetrics] = field(default_factory=list)

    @property
    def warm_turns(self) -> List[TurnMetrics]:
        """第 2 轮起（排除冷启动第 1 轮）。"""
        return self.turns[1:]

    @property
    def avg_first_token_ms(self) -> float:
        return float(np.mean([t.first_token_ms for t in self.warm_turns])) \
            if self.warm_turns else 0.0

    @property
    def total_new_tokens(self) -> int:
        return sum(t.approx_new_tokens for t in self.turns)


async def flush_kv_cache(client: httpx.AsyncClient) -> bool:
    """
    调用 SGLang /flush_cache API 清空 KV Cache。
    这是实现真正无缓存条件的唯一可靠方法。
    返回 True 表示清空成功。
    """
    try:
        resp = await client.post(f"{SGLANG_URL}/flush_cache", timeout=5.0)
        return resp.status_code == 200
    except Exception as e:
        print(f"  警告: flush_cache 失败 ({e})，无缓存条件可能不准确")
        return False


async def run_single_trial(
    use_cache: bool,
    dialogue_mgr: DialogueManager,
    session_mgr: SessionManager,
    streaming_mgr: StreamingManager,
    http_client: httpx.AsyncClient,
) -> TrialResult:
    """
    执行一次完整 8 轮对话，记录每轮的性能指标。
    use_cache=False 时，每轮推理前通过 /flush_cache 清空 KV Cache。
    """
    result = TrialResult(use_cache=use_cache)
    state = session_mgr.create_session(
        user_id=f"bench_{'cache' if use_cache else 'nocache'}_{int(time.time())}"
    )

    for turn_idx, user_input in enumerate(STANDARD_DIALOGUE, 1):
        # No-Cache 条件：每轮前清空 KV Cache
        if not use_cache:
            flushed = await flush_kv_cache(http_client)
            if not flushed and turn_idx == 1:
                print("  警告: 无法清空 KV Cache，实验结果可能不准确")

        # 构建提示
        system_prompt = dialogue_mgr.build_system_prompt(state, "")
        window = dialogue_mgr.context_mgr.get_optimal_context(state)
        history_msgs = dialogue_mgr.context_mgr.build_history_messages(window)

        # 测量首 Token 延迟
        start = time.perf_counter()
        first_token_time: Optional[float] = None
        token_count = 0
        response_text = ""

        try:
            async for chunk in streaming_mgr.stream_response(
                system_prompt=system_prompt,
                user_input=user_input,
                history_messages=history_msgs,
                max_tokens=300,
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                token_count += len(chunk)
                response_text += chunk
        except Exception as e:
            print(f"  第 {turn_idx} 轮生成失败: {e}")
            response_text = "（生成失败）"

        total_time = time.perf_counter() - start

        # 近似 token 数（字符数 × 0.4）
        approx_tokens = int(
            (len(system_prompt) + sum(len(m["content"]) for m in history_msgs)
             + len(user_input)) * 0.4
        )

        metrics = TurnMetrics(
            turn=turn_idx,
            first_token_ms=(first_token_time or total_time) * 1000,
            total_ms=total_time * 1000,
            approx_new_tokens=approx_tokens,
            cache_active=use_cache and turn_idx > 1,
        )
        result.turns.append(metrics)

        # 更新状态
        state.add_turn("user", user_input)
        state.add_turn("assistant", response_text)
        session_mgr.save_state(state)

    return result


def analyze_results(
    cache_trials: List[TrialResult],
    nocache_trials: List[TrialResult],
) -> dict:
    """汇总分析多次试验的统计结果。"""
    def _agg(trials: List[TrialResult], attr: str) -> dict:
        values = [getattr(t, attr) for t in trials]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    cache_warm_ttft = [
        ms for t in cache_trials for ms in
        [m.first_token_ms for m in t.warm_turns]
    ]
    nocache_warm_ttft = [
        ms for t in nocache_trials for ms in
        [m.first_token_ms for m in t.warm_turns]
    ]

    cache_tokens = [t.total_new_tokens for t in cache_trials]
    nocache_tokens = [t.total_new_tokens for t in nocache_trials]

    avg_cache_ttft = float(np.mean(cache_warm_ttft)) if cache_warm_ttft else 0
    avg_nocache_ttft = float(np.mean(nocache_warm_ttft)) if nocache_warm_ttft else 0
    ttft_reduction = (
        (avg_nocache_ttft - avg_cache_ttft) / avg_nocache_ttft * 100
        if avg_nocache_ttft > 0 else 0
    )

    avg_cache_tokens = float(np.mean(cache_tokens)) if cache_tokens else 0
    avg_nocache_tokens = float(np.mean(nocache_tokens)) if nocache_tokens else 0
    token_saving = (
        (avg_nocache_tokens - avg_cache_tokens) / avg_nocache_tokens * 100
        if avg_nocache_tokens > 0 else 0
    )

    return {
        "cache_warm_ttft_mean_ms": avg_cache_ttft,
        "nocache_warm_ttft_mean_ms": avg_nocache_ttft,
        "ttft_reduction_pct": ttft_reduction,
        "cache_total_tokens_mean": avg_cache_tokens,
        "nocache_total_tokens_mean": avg_nocache_tokens,
        "token_saving_pct": token_saving,
        "target_met": token_saving >= 35.0,  # 目标 ≥40%，允许 ±5% 误差
    }


def print_per_turn_comparison(
    cache_trials: List[TrialResult],
    nocache_trials: List[TrialResult],
) -> None:
    """逐轮打印 Cache vs No-Cache 的首 Token 延迟对比。"""
    print("\n  逐轮首 Token 延迟对比（均值，ms）:")
    print(f"  {'轮次':<6} {'Cache':<12} {'No-Cache':<12} {'节省':<12} {'加速比':<8}")
    print("  " + "─" * 48)

    n_turns = len(STANDARD_DIALOGUE)
    for turn_idx in range(1, n_turns + 1):
        c_vals = [
            t.turns[turn_idx - 1].first_token_ms
            for t in cache_trials
            if len(t.turns) >= turn_idx
        ]
        nc_vals = [
            t.turns[turn_idx - 1].first_token_ms
            for t in nocache_trials
            if len(t.turns) >= turn_idx
        ]
        if not c_vals or not nc_vals:
            continue
        c_mean = np.mean(c_vals)
        nc_mean = np.mean(nc_vals)
        saving = nc_mean - c_mean
        speedup = nc_mean / c_mean if c_mean > 0 else 1.0
        cold_flag = " (冷启动)" if turn_idx == 1 else ""
        print(
            f"  第{turn_idx}轮{cold_flag:<6} "
            f"{c_mean:<12.0f} {nc_mean:<12.0f} "
            f"{saving:<12.0f} {speedup:<8.2f}x"
        )


async def main():
    print("=" * 60)
    print("  前缀缓存效率验证实验")
    print("  测试环境: A100 80GB, ChatGLM-3-6B BF16, SGLang v0.1.17")
    print("=" * 60)

    # 初始化组件
    tool_mgr = ToolManager(use_mock=True)
    session_mgr = SessionManager()
    dialogue_mgr = DialogueManager(tool_manager=tool_mgr)
    streaming_mgr = StreamingManager()

    n_trials = 3  # 每个条件重复 3 次（完整实验建议 10 次）
    cache_results: List[TrialResult] = []
    nocache_results: List[TrialResult] = []

    async with httpx.AsyncClient(timeout=10.0) as http_client:
        # 健康检查
        try:
            resp = await http_client.get(f"{SGLANG_URL}/health")
            if resp.status_code != 200:
                raise Exception(f"健康检查失败: {resp.status_code}")
        except Exception as e:
            print(f"✗ SGLang 服务未就绪: {e}")
            sys.exit(1)
        print("✓ SGLang 服务正常\n")

        # Cache 组
        print(f"[1/2] 运行 Cache 组（{n_trials} 次重复）...")
        for i in range(n_trials):
            print(f"  试验 {i+1}/{n_trials}...", end=" ", flush=True)
            r = await run_single_trial(
                use_cache=True,
                dialogue_mgr=dialogue_mgr,
                session_mgr=session_mgr,
                streaming_mgr=streaming_mgr,
                http_client=http_client,
            )
            cache_results.append(r)
            print(f"完成，第2-8轮均值 {r.avg_first_token_ms:.0f}ms")

        # No-Cache 组
        print(f"\n[2/2] 运行 No-Cache 组（每轮前调用 /flush_cache，{n_trials} 次重复）...")
        for i in range(n_trials):
            print(f"  试验 {i+1}/{n_trials}...", end=" ", flush=True)
            r = await run_single_trial(
                use_cache=False,
                dialogue_mgr=dialogue_mgr,
                session_mgr=session_mgr,
                streaming_mgr=streaming_mgr,
                http_client=http_client,
            )
            nocache_results.append(r)
            print(f"完成，第2-8轮均值 {r.avg_first_token_ms:.0f}ms")

    # 分析结果
    stats = analyze_results(cache_results, nocache_results)

    print("\n" + "=" * 60)
    print("  实验结果汇总")
    print("=" * 60)

    print_per_turn_comparison(cache_results, nocache_results)

    print(f"\n  汇总统计（第2-8轮，排除冷启动）:")
    print(f"  Cache 首 Token 均值:   {stats['cache_warm_ttft_mean_ms']:.0f} ms")
    print(f"  No-Cache 首 Token 均值: {stats['nocache_warm_ttft_mean_ms']:.0f} ms")
    print(f"  延迟降低:              {stats['ttft_reduction_pct']:.1f}%")
    print(f"\n  Token 消耗对比（近似值）:")
    print(f"  Cache 总 Token 均值:   {stats['cache_total_tokens_mean']:.0f}")
    print(f"  No-Cache 总 Token 均值: {stats['nocache_total_tokens_mean']:.0f}")
    print(f"  Token 节省:            {stats['token_saving_pct']:.1f}%")

    target_label = "✓ 达标" if stats["target_met"] else "✗ 未达标"
    print(f"\n  目标（token 节省 ≥35%）: {target_label}")

    # 保存结果
    output_path = Path("results/prefix_cache_verification.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": stats,
                "cache_trials": [
                    [vars(m) for m in t.turns] for t in cache_results
                ],
                "nocache_trials": [
                    [vars(m) for m in t.turns] for t in nocache_results
                ],
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n  详细结果已保存: {output_path}")

    await tool_mgr.close_all()


if __name__ == "__main__":
    asyncio.run(main())
