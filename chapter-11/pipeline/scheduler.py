# pipeline/scheduler.py
# 两阶段流水线调度：预热阶段 + 批处理阶段（ch11_code_10）
#
# 设计目标：
#   在最大化缓存命中率的同时充分利用 GPU 计算资源。
#
# 核心调度策略：
#   - 文书内串行：同一文书的 6 项任务按序执行，
#     确保其 KV Cache 在所有任务完成前保持有效，
#     不会因 LRU 淘汰而被中途清除。
#   - 文书间并行：不同文书的处理可并发进行，
#     在 max_concurrent_docs 的约束下充分利用 GPU。
#   - 预热阶段：批处理正式开始前，并行触发所有文书
#     正文的 KV Cache 预计算，消除首次处理的冷启动延迟。
#
# 并发文书数（max_concurrent_docs）的选择：
#   过低（<4）：GPU 利用率不足，吞吐量未达上限；
#   过高（>12）：GPU 计算资源趋于饱和，显存压力增大；
#   推荐值 8 是在 4× A100 + Qwen2-72B 配置下的
#   计算资源饱和点与显存安全边界之间的平衡点。

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from config.model_config import DEPLOY_CONFIG, MODEL_CONFIG
from config.prompt_config import (
    TASK_TYPES,
    UNIFIED_SYSTEM_PROMPT,
    build_stable_prefix,
)
from data.dataset_builder import ContractDocument


@dataclass
class DocumentTaskResult:
    """单份文书单个任务的处理结果。"""
    doc_id: str
    task_type: str
    content: str
    # 首 token 延迟（毫秒）
    ttft_ms: float
    # 端到端延迟（毫秒）
    e2e_ms: float
    prompt_tokens: int
    completion_tokens: int
    # 是否经过缓存命中（由外部监控推断）
    cache_hit: bool = False
    error: str = ""


@dataclass
class PipelineStats:
    """流水线运行统计。"""
    total_docs: int = 0
    total_tasks: int = 0
    total_wall_time_s: float = 0.0
    warmup_time_s: float = 0.0
    results: list[DocumentTaskResult] = field(
        default_factory=list
    )


def _make_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key="EMPTY")


def _model_name() -> str:
    return MODEL_CONFIG["model_path"].split("/")[-1]


async def warmup_document_cache(
    docs: list[ContractDocument],
    base_url: str,
    warmup_concurrency: int = 16,
) -> float:
    """
    预热阶段：并行触发所有文书的 KV Cache 预计算。

    预热请求仅包含 L1（系统提示）+ L3（文书正文），
    不含 L2 任务指令和 L4 查询，以避免将任务指令的
    KV Cache 混入文书正文节点，保持缓存层级的清晰性。
    预热请求的输出极短（max_tokens=5），绝大部分计算
    时间用于 prefill，完成后文书正文的 KV Cache 即写入
    基数树，供后续批处理阶段的所有任务直接复用。

    参数
    ----
    docs : list[ContractDocument]
        需要预热的文书列表。
    base_url : str
        SGLang 或 vLLM 服务的 OpenAI 兼容接口基础 URL。
    warmup_concurrency : int
        预热并发数，可高于批处理并发数（预热请求 decode 极短）。

    返回
    ----
    float
        预热阶段耗时（秒）。
    """
    client = _make_client(base_url)
    sem = asyncio.Semaphore(warmup_concurrency)
    model = _model_name()

    async def warmup_single(doc: ContractDocument) -> None:
        async with sem:
            # 预热前缀：系统提示 + 文书正文（不含任务指令）
            warmup_content = (
                UNIFIED_SYSTEM_PROMPT
                + "\n\n"
                + doc.text
                + "\n\n请确认已读取以上文书内容，回复'已读取'即可。"
            )
            try:
                await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": warmup_content}
                    ],
                    # 极短输出，主要目的是触发 prefill 写入缓存
                    max_tokens=5,
                    temperature=0.0,
                )
            except Exception as exc:
                # 预热失败不阻断后续批处理，仅记录警告
                print(
                    f"[预热警告] 文书 {doc.doc_id} 预热失败：{exc}"
                )

    t0 = time.perf_counter()
    tasks = [warmup_single(doc) for doc in docs]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    print(
        f"预热完成：{len(docs)} 份文书，"
        f"耗时 {elapsed:.1f} 秒，"
        f"并发数 {warmup_concurrency}"
    )
    return elapsed


async def _process_single_document(
    doc: ContractDocument,
    task_types: list[str],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
) -> list[DocumentTaskResult]:
    """
    对单份文书按序执行所有任务（文书内串行）。

    文书内串行是保持 KV Cache 有效性的关键：
    同一文书的相邻两次请求之间，文书正文节点的引用计数
    始终大于零，不会被 LRU 淘汰。若改为跨文书交错执行，
    同一文书的两次请求之间可能插入大量其他文书的请求，
    导致文书正文节点因 LRU 被提前淘汰。
    """
    results: list[DocumentTaskResult] = []

    for task_type in task_types:
        prompt = build_stable_prefix(task_type, doc.text)
        t0 = time.perf_counter()
        first_token_time: float | None = None
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        error_msg = ""

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                delta = (
                    chunk.choices[0].delta.content
                    if chunk.choices
                    else None
                )
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks.append(delta)
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = (
                        chunk.usage.completion_tokens
                    )
        except Exception as exc:
            error_msg = str(exc)

        end = time.perf_counter()
        ttft_ms = (
            ((first_token_time or end) - t0) * 1000
        )
        results.append(DocumentTaskResult(
            doc_id=doc.doc_id,
            task_type=task_type,
            content="".join(chunks),
            ttft_ms=ttft_ms,
            e2e_ms=(end - t0) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error_msg,
        ))

    return results


async def batch_process_documents(
    docs: list[ContractDocument],
    base_url: str,
    task_types: list[str] | None = None,
    max_concurrent_docs: int = 8,
    max_tokens_per_task: int = 512,
) -> list[DocumentTaskResult]:
    """
    批处理阶段：文书内串行、文书间并行。

    参数
    ----
    docs : list[ContractDocument]
        待处理的文书列表。
    base_url : str
        推理服务的 OpenAI 兼容接口基础 URL。
    task_types : list[str], optional
        需要执行的任务类型列表，默认使用全部 6 种任务。
    max_concurrent_docs : int
        最大并发文书数，默认 8。
    max_tokens_per_task : int
        每个任务的最大输出 token 数，默认 512。

    返回
    ----
    list[DocumentTaskResult]
        所有文书、所有任务的结果列表，
        总条数 = len(docs) × len(task_types)。
    """
    if task_types is None:
        task_types = TASK_TYPES

    client = _make_client(base_url)
    model = _model_name()
    sem = asyncio.Semaphore(max_concurrent_docs)

    async def process_with_sem(
        doc: ContractDocument,
    ) -> list[DocumentTaskResult]:
        async with sem:
            return await _process_single_document(
                doc, task_types, client, model,
                max_tokens_per_task,
            )

    all_tasks = [process_with_sem(doc) for doc in docs]
    nested_results = await asyncio.gather(*all_tasks)
    # 展平嵌套列表
    return [r for doc_results in nested_results for r in doc_results]


async def run_full_pipeline(
    docs: list[ContractDocument],
    base_url: str,
    enable_warmup: bool = True,
    warmup_concurrency: int = 16,
    max_concurrent_docs: int = 8,
    task_types: list[str] | None = None,
    max_tokens_per_task: int = 512,
) -> PipelineStats:
    """
    完整两阶段流水线入口。

    阶段一（预热）：
        并行触发所有文书正文的 KV Cache 预计算。
        预热完成后，批处理阶段每份文书的首次任务
        即可直接命中 L3 缓存，消除冷启动延迟。

    阶段二（批处理）：
        文书内串行 + 文书间并行，在保持高缓存命中率
        的同时充分利用 GPU 计算资源。

    参数
    ----
    docs : list[ContractDocument]
        待处理的文书列表（须先经过 text_normalizer 处理）。
    base_url : str
        推理服务基础 URL。
    enable_warmup : bool
        是否执行预热阶段，默认 True。
        禁用可用于测试无预热情况下的冷启动性能。
    warmup_concurrency : int
        预热并发数，默认 16。
    max_concurrent_docs : int
        批处理阶段最大并发文书数，默认 8。
    task_types : list[str], optional
        任务类型列表。
    max_tokens_per_task : int
        每任务最大输出 token 数，默认 512。

    返回
    ----
    PipelineStats
        流水线统计信息，包含各请求的延迟详情。
    """
    if task_types is None:
        task_types = TASK_TYPES

    stats = PipelineStats(
        total_docs=len(docs),
        total_tasks=len(docs) * len(task_types),
    )
    wall_start = time.perf_counter()

    # 阶段一：预热
    warmup_time = 0.0
    if enable_warmup:
        print(
            f"阶段一（预热）：并行预热 {len(docs)} 份文书的 "
            "KV Cache..."
        )
        warmup_time = await warmup_document_cache(
            docs, base_url, warmup_concurrency
        )

    # 阶段二：批处理
    print(
        f"阶段二（批处理）：{len(docs)} 份文书，"
        f"{len(task_types)} 项任务，"
        f"并发文书数 {max_concurrent_docs}..."
    )
    results = await batch_process_documents(
        docs, base_url, task_types,
        max_concurrent_docs, max_tokens_per_task,
    )

    stats.warmup_time_s = warmup_time
    stats.total_wall_time_s = time.perf_counter() - wall_start
    stats.results = results

    print(
        f"流水线完成：总耗时 {stats.total_wall_time_s:.1f} 秒，"
        f"其中预热 {stats.warmup_time_s:.1f} 秒，"
        f"批处理 "
        f"{stats.total_wall_time_s - stats.warmup_time_s:.1f} 秒。"
    )
    return stats
