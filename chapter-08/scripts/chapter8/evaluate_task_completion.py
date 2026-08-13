"""
scripts/chapter8/evaluate_task_completion.py
任务完成率评估框架（10 个标准化测试场景）。

完成判定标准（四条件，全部满足）:
  1. 四个必填槽位（destination/start_date/end_date/num_travelers）全部填充
  2. 系统输出了包含'按天行程'和'推荐住宿'的方案文本（≥300字）
  3. 对话轮次不超过 10 轮
  4. 无明显槽位错误（num_travelers 在 1-20 之间）

用法: python scripts/chapter8/evaluate_task_completion.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.dialogue_manager import DialogueManager
from src.core.session_manager import SessionManager
from src.tools import ToolManager


@dataclass
class TaskScenario:
    """单个测试场景的定义。"""
    scenario_id: str
    description: str
    user_turns: List[str]
    max_turns: int = 10
    # 宽松模式：某些复杂场景只要求部分槽位
    required_slots_override: Optional[List[str]] = None


@dataclass
class TaskResult:
    """单个场景的执行结果。"""
    scenario_id: str
    description: str
    status: str          # "完成" | "未完成" | "超时"
    turns_used: int
    filled_slots: List[str]
    missing_slots: List[str]
    plan_generated: bool
    execution_time_s: float
    error: Optional[str] = None


# ── 10 个标准化测试场景 ───────────────────────────────────────────────

TEST_SCENARIOS = [
    TaskScenario(
        scenario_id="S001",
        description="标准顺畅流程",
        user_turns=[
            "我想去云南旅游",
            "下个月初出发，大概玩五天",
            "两个人，预算一万元",
            "想住舒适型酒店",
            "好的，帮我生成完整方案",
        ],
    ),
    TaskScenario(
        scenario_id="S002",
        description="信息分散（7轮）",
        user_turns=[
            "想去旅游",
            "云南吧",
            "3月1号出发",
            "5号回来",
            "两个人",
            "预算一万",
            "生成方案",
        ],
    ),
    TaskScenario(
        scenario_id="S003",
        description="工具调用密集",
        user_turns=[
            "我想去大理旅游，3月5号出发，玩三天，两个人",
            "先帮我查一下大理的天气",
            "推荐几个景点",
            "找几家酒店",
            "好的，生成完整方案",
        ],
    ),
    TaskScenario(
        scenario_id="S004",
        description="复杂需求+方案调整",
        user_turns=[
            "想带父母去丽江，3月中旬，5天，三个人，预算15000",
            "住舒适型酒店",
            "生成方案吧",
            "能不能多安排一天在古城",
            "预算可以加到18000",
            "好，就这样了",
        ],
    ),
    TaskScenario(
        scenario_id="S005",
        description="一次性完整表达",
        user_turns=[
            "帮我规划一下，3月去云南玩五天，两个人，预算一万，住舒适型酒店",
        ],
        max_turns=5,
    ),
    TaskScenario(
        scenario_id="S006",
        description="极度模糊表达",
        user_turns=[
            "想出去玩",
            "随便，南方比较好",
            "云南吧，听说不错",
            "下个月，具体不确定",
            "差不多五天",
            "两个人",
            "预算灵活，一万到一万五都行",
            "可以了，帮我规划一下",
        ],
        # 极度模糊场景放宽：只要求目的地和人数
        required_slots_override=["destination", "num_travelers"],
    ),
    TaskScenario(
        scenario_id="S007",
        description="包含闲聊干扰",
        user_turns=[
            "你好",
            "我想去旅游",
            "谢谢",
            "去云南，下个月初",
            "玩五天",
            "两个人",
            "预算一万",
            "生成方案",
        ],
    ),
    TaskScenario(
        scenario_id="S008",
        description="目的地变更",
        user_turns=[
            "我想去北京旅游，3月初，五天，两个人",
            "不，还是去云南吧，更适合春天",
            "预算一万",
            "生成方案",
        ],
    ),
    TaskScenario(
        scenario_id="S009",
        description="最少信息边界",
        user_turns=[
            "去云南",
            "下个月",
            "五天",
            "2个人",
            "生成方案",
        ],
    ),
    TaskScenario(
        scenario_id="S010",
        description="详细偏好（含可选槽位）",
        user_turns=[
            "我想3月1号去云南旅游，3月5号回来，两个人，预算一万",
            "对自然风光很感兴趣，想住精品民宿",
            "先帮我查昆明和大理的天气",
            "推荐景点",
            "生成完整的旅游方案",
        ],
    ),
]


def _check_plan_generated(response: str) -> bool:
    """
    检测响应是否包含完整方案文本。
    要求: 长度≥300字 且 包含行程关键词 且 包含住宿关键词。
    """
    if len(response) < 300:
        return False
    has_itinerary = any(
        kw in response
        for kw in ["第一天", "第1天", "Day 1", "行程", "第二天", "第2天"]
    )
    has_accommodation = any(
        kw in response
        for kw in ["酒店", "民宿", "住宿", "客栈"]
    )
    return has_itinerary and has_accommodation


def _check_slot_validity(prefs) -> bool:
    """检查已填充槽位是否合理（无明显错误）。"""
    if prefs.num_travelers is not None:
        if not (1 <= prefs.num_travelers <= 20):
            return False
    return True


async def run_scenario(
    scenario: TaskScenario,
    dialogue_mgr: DialogueManager,
    session_mgr: SessionManager,
) -> TaskResult:
    """执行单个测试场景，返回评估结果。"""
    start_time = time.perf_counter()
    state = session_mgr.create_session(
        user_id=f"eval_{scenario.scenario_id}"
    )
    plan_generated = False
    error = None

    print(f"\n  [{scenario.scenario_id}] {scenario.description}")

    try:
        for turn_idx, user_input in enumerate(scenario.user_turns, 1):
            if turn_idx > scenario.max_turns:
                break
            print(f"    第{turn_idx}轮 | 用户: {user_input[:40]}")

            response = await dialogue_mgr.process_user_input(
                user_input, state
            )
            print(f"    第{turn_idx}轮 | 助手: {response[:60]}...")

            if _check_plan_generated(response):
                plan_generated = True
                break  # 方案已生成，提前结束

            session_mgr.save_state(state)

    except Exception as e:
        error = str(e)
        print(f"    ✗ 异常: {e}")

    elapsed = time.perf_counter() - start_time
    prefs = state.preferences

    # 评估槽位完成情况
    required = scenario.required_slots_override or [
        "destination", "start_date", "end_date", "num_travelers"
    ]
    filled = [s for s in required if getattr(prefs, s) is not None]
    missing = [s for s in required if getattr(prefs, s) is None]

    # 判断完成状态
    slots_ok = len(missing) == 0
    validity_ok = _check_slot_validity(prefs)
    within_limit = state.turn_count <= scenario.max_turns

    if slots_ok and plan_generated and validity_ok and within_limit:
        status = "完成"
    elif state.turn_count > scenario.max_turns:
        status = "超时"
    else:
        status = "未完成"

    result = TaskResult(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        status=status,
        turns_used=state.turn_count,
        filled_slots=filled,
        missing_slots=missing,
        plan_generated=plan_generated,
        execution_time_s=elapsed,
        error=error,
    )

    status_icon = "✓" if status == "完成" else "✗"
    print(
        f"    {status_icon} 状态: {status} | "
        f"轮次: {state.turn_count} | "
        f"槽位: {len(filled)}/{len(required)} | "
        f"方案: {'是' if plan_generated else '否'}"
    )
    return result


def print_summary(results: List[TaskResult]) -> None:
    """打印评估汇总报告。"""
    total = len(results)
    completed = sum(1 for r in results if r.status == "完成")
    rate = completed / total * 100

    print("\n" + "=" * 65)
    print("  任务完成率评估报告")
    print("=" * 65)
    print(
        f"\n  {'场景ID':<8} {'描述':<22} {'状态':<8} "
        f"{'轮次':<6} {'槽位':<8} {'方案':<6}"
    )
    print("  " + "─" * 58)

    for r in results:
        slot_str = f"{len(r.filled_slots)}/{len(r.filled_slots)+len(r.missing_slots)}"
        print(
            f"  {r.scenario_id:<8} {r.description[:20]:<22} "
            f"{r.status:<8} {r.turns_used:<6} "
            f"{slot_str:<8} {'是' if r.plan_generated else '否':<6}"
        )

    print(f"\n  总计: {completed}/{total} 完成 ({rate:.0f}%)")
    print(f"  目标: ≥90%（{'✓ 达标' if rate >= 90 else '✗ 未达标'}）")

    avg_turns = sum(r.turns_used for r in results if r.status == "完成") / max(completed, 1)
    print(f"  完成场景平均轮次: {avg_turns:.1f}")

    if any(r.status == "未完成" or r.status == "超时" for r in results):
        print("\n  未完成场景分析:")
        for r in results:
            if r.status != "完成":
                print(
                    f"  - {r.scenario_id} ({r.description}): "
                    f"缺失槽位={r.missing_slots}, "
                    f"方案生成={'是' if r.plan_generated else '否'}"
                )
                if r.error:
                    print(f"    错误: {r.error}")


async def main():
    print("=" * 65)
    print("  多轮对话 Agent 任务完成率评估")
    print("  10 个标准化场景，通过线 ≥90%")
    print("=" * 65)

    tool_mgr = ToolManager(use_mock=True)
    session_mgr = SessionManager()
    dialogue_mgr = DialogueManager(tool_manager=tool_mgr)

    print(f"\n共 {len(TEST_SCENARIOS)} 个测试场景，开始评估...\n")

    results: List[TaskResult] = []
    for scenario in TEST_SCENARIOS:
        result = await run_scenario(scenario, dialogue_mgr, session_mgr)
        results.append(result)
        await asyncio.sleep(0.5)  # 避免请求过于密集

    print_summary(results)

    # 保存结果
    output = Path("results/task_completion_evaluation.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "scenario_id": r.scenario_id,
                    "description": r.description,
                    "status": r.status,
                    "turns_used": r.turns_used,
                    "filled_slots": r.filled_slots,
                    "missing_slots": r.missing_slots,
                    "plan_generated": r.plan_generated,
                    "execution_time_s": round(r.execution_time_s, 2),
                    "error": r.error,
                }
                for r in results
            ],
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n  详细结果已保存: {output}")
    await tool_mgr.close_all()


if __name__ == "__main__":
    asyncio.run(main())
