"""
第3章 代码示例 37
主题：事件驱动工作流——基于事件总线的异步编排
场景：构建事件驱动的文档处理工作流：
      文档到达 → 并发触发分析与质量评估 → 分析完成触发翻译
      各处理器通过事件类型解耦，新增处理逻辑无需修改已有代码

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@dataclass
class WorkflowEvent:
    """工作流事件：携带类型、数据和元信息。"""
    event_type: str
    data: Any
    source: str = ""
    timestamp: float = field(
        default_factory=lambda: __import__("time").time()
    )


class SGLangEventWorkflow:
    """
    基于事件总线的异步工作流引擎。
    各处理器通过注册事件类型声明感兴趣的事件，
    事件发生时引擎自动调度对应处理器执行，
    处理结果可以产生新事件形成处理链。
    """
    def __init__(self):
        # 事件类型 -> 处理器列表的映射
        self._handlers: Dict[str, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        # 收集所有处理结果，供外部查询
        self.results: Dict[str, Any] = {}
        self._event_log: List[Dict] = []

    def on(self, event_type: str):
        """装饰器：将函数注册为指定事件类型的处理器。"""
        def decorator(handler: Callable):
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler
        return decorator

    async def emit(self, event: WorkflowEvent):
        """发送事件到队列，由事件循环异步调度处理。"""
        await self._event_queue.put(event)
        self._event_log.append({
            "action": "emitted",
            "type": event.event_type,
            "source": event.source,
            "ts": event.timestamp,
        })

    async def run_until_empty(self):
        """处理队列中的所有事件直到队列为空。"""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=1.0
                )
                handlers = self._handlers.get(
                    event.event_type, []
                )
                # 并发执行所有该事件类型的处理器
                await asyncio.gather(*[
                    handler(event, self) for handler in handlers
                ])
                self._event_log.append({
                    "action": "processed",
                    "type": event.event_type,
                    "handlers": len(handlers),
                })
            except asyncio.TimeoutError:
                if self._event_queue.empty():
                    break
        self._running = False

    def get_event_log(self) -> List[Dict]:
        return self._event_log


# ── SGLang处理函数 ────────────────────────────────────────────

@sgl.function
def analyze_document(s, document: str):
    """提取文档的核心要点。"""
    s += sgl.system("你是文档分析专家。")
    s += sgl.user(
        f"请从以下文档提取3个核心要点：\n{document}"
    )
    s += sgl.assistant(sgl.gen("analysis", max_tokens=150))


@sgl.function
def assess_quality(s, document: str):
    """评估文档质量（1-10分）。"""
    s += sgl.user(
        f"评估以下文档的整体质量（1-10分，只输出数字）：\n"
        f"{document[:200]}..."
    )
    s += sgl.assistant(
        sgl.gen(
            "score",
            max_tokens=3,
            regex=r"([1-9]|10)",
            temperature=0.0,
        )
    )


@sgl.function
def translate_summary(s, summary: str, target_lang: str = "英文"):
    """将摘要翻译为目标语言。"""
    s += sgl.system(f"你是专业{target_lang}翻译。")
    s += sgl.user(
        f"将以下内容翻译为{target_lang}（只输出译文）：\n{summary}"
    )
    s += sgl.assistant(
        sgl.gen("translation", max_tokens=200)
    )


@sgl.function
def generate_executive_summary(
    s, analysis: str, quality_score: str
):
    """基于分析结果生成执行摘要。"""
    s += sgl.user(
        f"基于以下信息生成100字的执行摘要：\n"
        f"核心要点：{analysis}\n"
        f"质量评分：{quality_score}/10"
    )
    s += sgl.assistant(sgl.gen("summary", max_tokens=150))


# ── 构建事件驱动工作流 ────────────────────────────────────────

async def run_document_workflow(document: str) -> Dict:
    """
    事件驱动的文档处理工作流。

    事件流：
      document_received
        → analysis_requested（文档分析）
        → quality_check_requested（质量评估，与分析并行）
      analysis_completed
        → translation_requested（翻译摘要）
        → summary_requested（生成执行摘要）
    """
    workflow = SGLangEventWorkflow()
    results = {}

    # ── 注册事件处理器 ────────────────────────────────────────

    @workflow.on("document_received")
    async def on_document_received(
        event: WorkflowEvent, wf: SGLangEventWorkflow
    ):
        """文档到达：同时触发分析和质量评估两个独立任务。"""
        doc = event.data
        await asyncio.gather(
            wf.emit(WorkflowEvent(
                "analysis_requested", doc,
                source="document_receiver"
            )),
            wf.emit(WorkflowEvent(
                "quality_check_requested", doc,
                source="document_receiver"
            )),
        )

    @workflow.on("analysis_requested")
    async def on_analysis_requested(
        event: WorkflowEvent, wf: SGLangEventWorkflow
    ):
        """执行文档分析，完成后触发翻译和摘要生成。"""
        state = analyze_document.run(document=event.data)
        analysis = state["analysis"]
        results["analysis"] = analysis

        # 分析完成后触发下游任务
        await asyncio.gather(
            wf.emit(WorkflowEvent(
                "translation_requested",
                {"summary": analysis, "lang": "英文"},
                source="analyzer"
            )),
            wf.emit(WorkflowEvent(
                "summary_requested",
                {"analysis": analysis},
                source="analyzer"
            )),
        )

    @workflow.on("quality_check_requested")
    async def on_quality_check(
        event: WorkflowEvent, wf: SGLangEventWorkflow
    ):
        """独立的质量评估，与分析流程并行执行。"""
        state = assess_quality.run(document=event.data)
        results["quality_score"] = state["score"]

    @workflow.on("translation_requested")
    async def on_translation_requested(
        event: WorkflowEvent, wf: SGLangEventWorkflow
    ):
        """执行翻译。"""
        data = event.data
        state = translate_summary.run(
            summary=data["summary"],
            target_lang=data["lang"],
        )
        results[f"translation_{data['lang']}"] = (
            state["translation"]
        )

    @workflow.on("summary_requested")
    async def on_summary_requested(
        event: WorkflowEvent, wf: SGLangEventWorkflow
    ):
        """等待质量评分后生成执行摘要。"""
        # 等待质量评分完成（简单轮询，实际可用Event/Condition）
        for _ in range(10):
            if "quality_score" in results:
                break
            await asyncio.sleep(0.1)

        data = event.data
        state = generate_executive_summary.run(
            analysis=data["analysis"],
            quality_score=results.get("quality_score", "N/A"),
        )
        results["executive_summary"] = state["summary"]

    # ── 启动工作流 ────────────────────────────────────────────
    await workflow.emit(WorkflowEvent(
        "document_received", document, source="user"
    ))
    await workflow.run_until_empty(timeout_seconds=60)

    results["_event_log"] = workflow.get_event_log()
    return results


# ── run_until_empty扩展（添加timeout参数）───────────────────

async def _run_until_empty_with_timeout(
    self, timeout_seconds: float = 30.0
):
    self._running = True
    deadline = time.time() + timeout_seconds
    while self._running and time.time() < deadline:
        try:
            event = await asyncio.wait_for(
                self._event_queue.get(), timeout=1.0
            )
            handlers = self._handlers.get(event.event_type, [])
            await asyncio.gather(*[
                h(event, self) for h in handlers
            ], return_exceptions=True)
            self._event_log.append({
                "action": "processed",
                "type": event.event_type,
            })
        except asyncio.TimeoutError:
            if self._event_queue.empty():
                break
    self._running = False


# 动态绑定带超时参数的版本
SGLangEventWorkflow.run_until_empty = (
    _run_until_empty_with_timeout
)


def main():
    sample_document = (
        "大语言模型在2024年取得了里程碑式进展。"
        "GPT-4o、Claude 3.5等模型在多项基准测试中"
        "超越了人类专家水平，尤其在编程和数学推理方面。"
        "然而幻觉问题仍未根本解决，推理成本依然高昂。"
        "模型压缩和高效微调技术有望在2025年大幅降低部署门槛。"
    )

    # ── 演示1：运行事件驱动工作流 ─────────────────────────────
    print("=" * 60)
    print("【演示1：事件驱动文档处理工作流】")
    print("=" * 60)
    print(f"\n文档：{sample_document[:60]}...")
    print()
    print("事件流：document_received")
    print("  ├─ analysis_requested → translation + summary")
    print("  └─ quality_check_requested（与分析并行）")
    print()

    start = time.time()
    results = asyncio.run(
        run_document_workflow(sample_document)
    )
    elapsed = (time.time() - start) * 1000

    print(f"工作流完成，总耗时：{elapsed:.0f}ms\n")

    if "analysis" in results:
        print(f"核心要点：\n{results['analysis'][:150]}...")
    if "quality_score" in results:
        print(f"\n质量评分：{results['quality_score']}/10")
    if "translation_英文" in results:
        print(
            f"\n英文摘要：{results['translation_英文'][:80]}..."
        )
    if "executive_summary" in results:
        print(
            f"\n执行摘要：{results['executive_summary'][:100]}..."
        )

    # ── 演示2：事件处理日志 ───────────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：事件处理日志】")
    print("=" * 60)
    print()

    event_log = results.get("_event_log", [])
    emitted = [e for e in event_log if e["action"] == "emitted"]
    processed = [e for e in event_log if e["action"] == "processed"]

    print(f"发送事件数：{len(emitted)}")
    for e in emitted:
        print(f"  → [{e['type']}] 来自：{e.get('source', 'N/A')}")

    print(f"\n处理事件数：{len(processed)}")
    for e in processed:
        print(
            f"  ✓ [{e['type']}] "
            f"处理器数：{e.get('handlers', 0)}"
        )

    # ── 演示3：事件驱动vs状态机对比 ──────────────────────────
    print()
    print("=" * 60)
    print("【事件驱动 vs 状态机工作流对比】")
    print("=" * 60)
    print()
    print(f"{'维度':<12} {'状态机（FSM）':<22} {'事件驱动'}")
    print("-" * 55)
    rows = [
        ("耦合度",   "状态转换路径预先确定",   "处理器通过事件类型解耦"),
        ("扩展性",   "新增状态需修改转换图",   "新增处理器不改已有代码"),
        ("并发",     "通常顺序执行",           "多处理器自然并发"),
        ("调试",     "状态历史清晰",           "事件日志完整可追溯"),
        ("适用场景", "线性/树状流程",           "复杂网状依赖流程"),
    ]
    for dim, fsm, evt in rows:
        print(f"  {dim:<10} {fsm:<22} {evt}")

    print()
    print("开闭原则体现：")
    print("  新增「关键词提取」功能只需：")
    print("  1. 定义新处理器并注册到 analysis_completed 事件")
    print("  2. 无需修改任何已有处理器或引擎代码")


if __name__ == "__main__":
    main()
