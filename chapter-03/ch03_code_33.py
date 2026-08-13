"""
第3章 代码示例 33
主题：顺序链式执行工作流——三步骤文档分析管道
场景：步骤间存在严格数据依赖，必须按序完成：
      论点提取 → 可信度评估 → 综合报告生成
      每步依赖上一步的输出，代码结构直接反映执行顺序

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000
"""

import time
from dataclasses import dataclass, field
from typing import Dict

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


@sgl.function
def extract_key_points(s, document: str):
    """步骤1：从文档中提取关键论点"""
    s += sgl.system("你是一位专业的文档分析师。")
    s += sgl.user(
        f"请从以下文档中提取3-5个核心论点：\n{document}"
    )
    s += sgl.assistant(sgl.gen("key_points", max_tokens=200))


@sgl.function
def assess_credibility(s, key_points: str):
    """步骤2：评估论点的可信度"""
    s += sgl.user(
        f"请评估以下论点的可信度（高/中/低）：\n{key_points}"
    )
    s += sgl.assistant(
        sgl.gen("credibility",
                choices=["高可信度", "中等可信度", "低可信度"])
    )


@sgl.function
def generate_report(
    s, key_points: str, credibility: str, document: str
):
    """步骤3：生成综合分析报告"""
    s += sgl.user(
        f"基于以下信息生成分析报告：\n"
        f"核心论点：{key_points}\n"
        f"可信度评估：{credibility}\n"
        f"原文摘要：{document[:200]}..."
    )
    s += sgl.assistant(sgl.gen("report", max_tokens=400))


@sgl.function
def document_analysis_pipeline(s, document: str):
    """
    三步骤顺序管道：每步依赖前步输出，严格串行执行。
    代码结构直接反映执行顺序，无需额外注释。
    """
    # 步骤1 -> 步骤2 -> 步骤3，数据依赖关系清晰
    step1 = extract_key_points.run(document=document)
    key_points = step1["key_points"]

    step2 = assess_credibility.run(key_points=key_points)
    credibility = step2["credibility"]

    step3 = generate_report.run(
        key_points=key_points,
        credibility=credibility,
        document=document,
    )

    s += sgl.user("分析完成，请确认结果。")
    s += sgl.assistant(
        f"分析报告已生成。可信度：{credibility}\n"
        f"报告摘要：{step3['report'][:100]}..."
    )
    return {
        "key_points": key_points,
        "credibility": credibility,
        "report": step3["report"],
    }


def run_pipeline_with_timing(document: str) -> Dict:
    """
    执行管道并记录每个步骤的耗时，用于性能分析。
    """
    step_times = {}

    # 步骤1：独立执行，记录耗时
    t = time.time()
    s1 = extract_key_points.run(document=document)
    step_times["step1_extract"] = (time.time() - t) * 1000
    key_points = s1["key_points"]

    # 步骤2：依赖步骤1输出
    t = time.time()
    s2 = assess_credibility.run(key_points=key_points)
    step_times["step2_assess"] = (time.time() - t) * 1000
    credibility = s2["credibility"]

    # 步骤3：依赖步骤1和步骤2的输出
    t = time.time()
    s3 = generate_report.run(
        key_points=key_points,
        credibility=credibility,
        document=document,
    )
    step_times["step3_report"] = (time.time() - t) * 1000

    step_times["total"] = sum(step_times.values())
    return {
        "key_points": key_points,
        "credibility": credibility,
        "report": s3["report"],
        "step_times": step_times,
    }


def main():
    sample_document = """
    人工智能大语言模型在2024年取得了里程碑式进展。
    GPT-4o、Claude 3.5等模型在多项基准测试中超越了人类专家水平，
    尤其在编程、数学推理和科学问答方面表现突出。

    然而这些模型仍存在明显局限性：幻觉问题未得到根本解决，
    模型有时会以高置信度生成错误信息；推理成本高昂，
    部署大型模型需要昂贵的GPU基础设施，限制了其在
    资源受限场景中的应用。

    展望未来，模型压缩与高效微调技术有望降低部署门槛，
    使大语言模型在更广泛的场景中得到应用。
    """.strip()

    # ── 演示1：完整管道执行 ───────────────────────────────────
    print("=" * 60)
    print("【演示1：三步骤顺序文档分析管道】")
    print("=" * 60)
    print(f"\n文档预览：{sample_document[:60]}...\n")

    start = time.time()
    state = document_analysis_pipeline.run(
        document=sample_document
    )
    total_elapsed = (time.time() - start) * 1000

    result = state.ret_value
    print(f"【步骤1：关键论点提取】")
    print(f"  {result['key_points'][:150]}...")
    print(f"\n【步骤2：可信度评估】")
    print(f"  评级：{result['credibility']}")
    print(f"\n【步骤3：综合分析报告】")
    print(f"  {result['report'][:200]}...")
    print(f"\n总耗时：{total_elapsed:.0f}ms")

    # ── 演示2：分步执行与耗时分析 ─────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：分步执行与各步骤耗时分析】")
    print("=" * 60)
    print()

    timed_result = run_pipeline_with_timing(sample_document)
    times = timed_result["step_times"]

    print(f"可信度评级：{timed_result['credibility']}")
    print()
    print("各步骤耗时：")
    for step, ms in times.items():
        if step != "total":
            pct = ms / times["total"] * 100
            print(f"  {step:<20}：{ms:>6.0f}ms  ({pct:.0f}%)")
    print(f"  {'总计':<20}：{times['total']:>6.0f}ms")

    # ── 演示3：分步调试——单独运行各步骤 ─────────────────────
    print()
    print("=" * 60)
    print("【演示3：分步调试演示——单独验证各步骤】")
    print("=" * 60)
    print()

    # 步骤1可独立测试
    print("单独运行步骤1（论点提取）：")
    s1 = extract_key_points.run(document=sample_document)
    print(f"  论点预览：{s1['key_points'][:80]}...")

    # 步骤2可独立测试（传入任意论点文本）
    print("\n单独运行步骤2（可信度评估）：")
    test_points = "AI模型性能超越人类专家。AI存在幻觉问题。"
    s2 = assess_credibility.run(key_points=test_points)
    print(f"  可信度：{s2['credibility']}")

    # 演示修复某步骤后无需重跑其他步骤
    print()
    print("调试优势：")
    print("  步骤1输出不满意时，只需修改extract_key_points")
    print("  步骤2/3无需重新运行，节省时间和计算成本")

    # ── 多文档批量处理 ────────────────────────────────────────
    print()
    print("=" * 60)
    print("【演示4：多文档批量顺序处理】")
    print("=" * 60)
    print()

    docs = [
        "量子计算利用量子力学原理处理信息，"
        "量子比特可同时处于叠加态，理论上能以指数级速度"
        "解决特定类型的问题。目前量子纠错仍是主要挑战。",

        "区块链是分布式账本技术，通过密码学和共识机制"
        "保证数据不可篡改。比特币和以太坊是主要应用，"
        "但能耗和扩展性问题限制了其大规模推广。",
    ]

    for i, doc in enumerate(docs, 1):
        print(f"文档{i}：{doc[:35]}...")
        st = document_analysis_pipeline.run(document=doc)
        r = st.ret_value
        print(f"  可信度：{r['credibility']}")
        print(f"  报告：{r['report'][:60]}...")
        print()

    print("=" * 60)
    print("【顺序管道的工程价值】")
    print("=" * 60)
    print()
    print("代码即文档：执行顺序与代码结构完全一致")
    print("可调试性  ：任意步骤可独立运行和验证")
    print("失败定位  ：异常精确归属到某个步骤")
    print("适用场景  ：步骤间存在严格数据依赖的工作流")


if __name__ == "__main__":
    main()
