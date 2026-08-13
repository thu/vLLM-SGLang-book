"""
scripts/chapter8/verify_function_calling.py
验证 ChatGLM-3-6B 的 Function Calling 能力（三层测试）。

层次一: 单工具调用 - 模型能否识别工具调用时机并输出正确格式
层次二: 工具结果整合 - 工具结果能否以自然语言融入回答
层次三: 工具失败降级 - 工具失败时能否给出有价值的降级回答

运行前置条件: SGLang Runtime 已在 127.0.0.1:30000 启动
用法: python scripts/chapter8/verify_function_calling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import sglang as sgl

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sgl.set_default_backend(sgl.RuntimeEndpoint("http://127.0.0.1:30000"))

# ── 工具定义（供测试使用）────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "date": {"type": "string", "description": "查询日期"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_hotel",
        "description": "搜索指定城市的酒店",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "check_in": {"type": "string"},
                "check_out": {"type": "string"},
                "price_range": {
                    "type": "string",
                    "enum": ["经济型", "舒适型", "豪华型"],
                },
            },
            "required": ["city", "check_in", "check_out"],
        },
    },
]


# ── 测试用 SGLang 函数（模块级，符合约束）────────────────────────────

@sgl.function
def test_tier1_tool_recognition(s, user_query: str, tools_desc: str):
    """层次一：验证模型能识别工具调用时机并输出结构化调用指令。"""
    s += sgl.system(
        "你是旅游助手，可以调用以下工具获取实时信息:\n"
        f"{tools_desc}\n\n"
        "当需要查询实时信息时，以 JSON 格式输出工具调用指令:\n"
        '{"tool": "工具名", "arguments": {参数}}\n'
        "如果不需要工具，直接回答。"
    )
    s += sgl.user(user_query)
    s += sgl.assistant(sgl.gen("output", max_tokens=200))


@sgl.function
def test_tier2_result_integration(
    s, original_query: str, tool_name: str, tool_result_json: str
):
    """层次二：验证工具结果能以自然语言融入回答（不直接输出 JSON）。"""
    s += sgl.system(
        "你是旅游助手。以下是工具查询结果，请用自然语言回答用户的问题。\n"
        "要求: 不要直接输出 JSON，将信息融入流畅的回答中。"
    )
    s += sgl.user(
        f"原始问题: {original_query}\n"
        f"工具 '{tool_name}' 返回结果: {tool_result_json}"
    )
    s += sgl.assistant(sgl.gen("integrated_answer", max_tokens=250))


@sgl.function
def test_tier3_graceful_degradation(
    s, original_query: str, failure_reason: str
):
    """层次三：验证工具失败时能给出基于历史知识的有价值降级回答。"""
    s += sgl.system(
        "你是旅游助手。工具查询失败，请基于你的知识给出有参考价值的回答。\n"
        "要求: 说明无法获取实时数据，但提供历史/经验参考信息。\n"
        "禁止: 只说'查询失败，请重试'等无实质内容的回答。"
    )
    s += sgl.user(
        f"用户问题: {original_query}\n"
        f"工具失败原因: {failure_reason}"
    )
    s += sgl.assistant(sgl.gen("degraded_answer", max_tokens=200))


# ── 辅助函数 ─────────────────────────────────────────────────────────

def _check_sglang_health() -> bool:
    """检查 SGLang 服务是否就绪。"""
    try:
        resp = httpx.get("http://127.0.0.1:30000/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _format_tools_desc(tools: list) -> str:
    return "\n".join(
        f"- {t['name']}: {t['description']}" for t in tools
    )


def _try_parse_tool_call(output: str) -> dict | None:
    """尝试从模型输出中提取 JSON 工具调用。"""
    import re
    # 提取 JSON 块
    pattern = r'\{[^{}]*"tool"[^{}]*\}'
    matches = re.findall(pattern, output, re.DOTALL)
    for m in matches:
        try:
            data = json.loads(m)
            if "tool" in data and "arguments" in data:
                return data
        except json.JSONDecodeError:
            pass
    return None


# ── 三层测试 ─────────────────────────────────────────────────────────

def run_tier1_tests() -> tuple[int, int]:
    """层次一：工具调用识别，期望成功率 ≥80%。"""
    print("\n" + "─" * 55)
    print("【层次一】工具调用识别测试")
    print("─" * 55)

    tools_desc = _format_tools_desc(TOOL_DEFINITIONS)
    test_cases = [
        ("帮我查一下昆明明天的天气", True, "get_weather"),
        ("推荐大理的舒适型酒店，3月1号入住，5号退房", True, "search_hotel"),
        ("你好，我想去旅游", False, None),
        ("云南旅游一般花多少钱", False, None),
        ("帮我查丽江的天气", True, "get_weather"),
    ]

    passed = 0
    for user_input, expect_tool_call, expected_tool in test_cases:
        result = test_tier1_tool_recognition.run(
            user_query=user_input,
            tools_desc=tools_desc,
        )
        output = result.get("output", "").strip()
        tool_call = _try_parse_tool_call(output)
        actually_called = tool_call is not None

        # 判断是否符合预期
        if expect_tool_call:
            ok = actually_called and (
                expected_tool is None
                or (tool_call and tool_call.get("tool") == expected_tool)
            )
        else:
            ok = not actually_called

        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(
            f"  {status} [{user_input[:30]:<30}] "
            f"期望={'调用' if expect_tool_call else '不调用'} "
            f"实际={'调用 ' + (tool_call['tool'] if tool_call else '?') if actually_called else '不调用'}"
        )

    total = len(test_cases)
    print(f"\n  结果: {passed}/{total} 通过 ({passed/total*100:.0f}%)")
    return passed, total


def run_tier2_tests() -> tuple[int, int]:
    """层次二：工具结果自然语言整合，人工评估模型输出质量。"""
    print("\n" + "─" * 55)
    print("【层次二】工具结果整合测试")
    print("─" * 55)

    test_cases = [
        {
            "query": "帮我查昆明的天气",
            "tool": "get_weather",
            "result": json.dumps({
                "city": "昆明", "date": "近期",
                "weather": "晴天", "temperature": "15-23°C",
                "humidity": "55%", "suggestion": "适合出行",
            }, ensure_ascii=False),
            "check": lambda ans: "昆明" in ans and ("°C" in ans or "温度" in ans or "晴" in ans),
        },
        {
            "query": "推荐大理的酒店",
            "tool": "search_hotel",
            "result": json.dumps({
                "city": "大理", "price_range": "舒适型",
                "hotels": [
                    {"name": "洱海边客栈", "rating": 4.6,
                     "price": 468, "total_price": 1404},
                ]
            }, ensure_ascii=False),
            "check": lambda ans: "洱海" in ans or "大理" in ans,
        },
    ]

    passed = 0
    for case in test_cases:
        result = test_tier2_result_integration.run(
            original_query=case["query"],
            tool_name=case["tool"],
            tool_result_json=case["result"],
        )
        answer = result.get("integrated_answer", "").strip()
        ok = case["check"](answer)
        if ok:
            passed += 1
        status = "✓" if ok else "✗"
        print(f"  {status} [{case['query']}]")
        print(f"    模型输出（前100字）: {answer[:100]}...")

    total = len(test_cases)
    print(f"\n  结果: {passed}/{total} 通过")
    return passed, total


def run_tier3_tests() -> tuple[int, int]:
    """层次三：工具失败降级，确保不输出无意义的'请重试'。"""
    print("\n" + "─" * 55)
    print("【层次三】工具失败降级测试")
    print("─" * 55)

    test_cases = [
        {
            "query": "昆明三月天气怎么样",
            "failure": "网络超时，无法获取实时天气数据",
            # 验证：回答中包含有价值的信息，不只是'请重试'
            "check": lambda ans: len(ans) > 50 and "重试" not in ans[:30],
        },
        {
            "query": "大理有什么好玩的",
            "failure": "景点 API 暂时不可用",
            "check": lambda ans: len(ans) > 60 and (
                "洱海" in ans or "古城" in ans or "景点" in ans
            ),
        },
    ]

    passed = 0
    for case in test_cases:
        result = test_tier3_graceful_degradation.run(
            original_query=case["query"],
            failure_reason=case["failure"],
        )
        answer = result.get("degraded_answer", "").strip()
        ok = case["check"](answer)
        if ok:
            passed += 1
        status = "✓" if ok else "✗"
        print(f"  {status} [{case['query']}]")
        print(f"    降级回答（前100字）: {answer[:100]}...")

    total = len(test_cases)
    print(f"\n  结果: {passed}/{total} 通过")
    return passed, total


def main():
    print("=" * 55)
    print("  ChatGLM-3-6B Function Calling 能力验证")
    print("=" * 55)

    # 健康检查
    print("\n检查 SGLang 服务状态...")
    if not _check_sglang_health():
        print("✗ SGLang 服务未就绪，请先运行:")
        print("  bash scripts/chapter8/start_sglang_server.sh")
        sys.exit(1)
    print("✓ SGLang 服务正常")

    # 运行三层测试
    r1_pass, r1_total = run_tier1_tests()
    r2_pass, r2_total = run_tier2_tests()
    r3_pass, r3_total = run_tier3_tests()

    # 汇总
    total_pass = r1_pass + r2_pass + r3_pass
    total_all = r1_total + r2_total + r3_total
    rate = total_pass / total_all * 100

    print("\n" + "=" * 55)
    print("  验证汇总")
    print("=" * 55)
    print(f"  层次一（工具识别）: {r1_pass}/{r1_total}")
    print(f"  层次二（结果整合）: {r2_pass}/{r2_total}")
    print(f"  层次三（失败降级）: {r3_pass}/{r3_total}")
    print(f"  总体通过率: {total_pass}/{total_all} ({rate:.0f}%)")

    # 通过标准：层次一 ≥80%，整体 ≥75%
    tier1_rate = r1_pass / r1_total * 100
    if tier1_rate >= 80 and rate >= 75:
        print("\n  ✓ Function Calling 能力达标，可用于后续开发")
        sys.exit(0)
    else:
        print(
            f"\n  ✗ 未达标（层次一需 ≥80%，当前 {tier1_rate:.0f}%）"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
