"""
第3章 代码示例 36
主题：基于有限状态机（FSM）的复杂对话工作流
场景：客服对话状态机，演示显式状态管理、
      select驱动的状态转换和多阶段交互逻辑

运行前提：
    pip install sglang
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000

状态转换图：
    GREETING → INTENT_DETECTION
    INTENT_DETECTION → INFO_GATHERING（查询订单/申请退款）
    INTENT_DETECTION → SOLUTION（产品咨询/投诉建议）
    INFO_GATHERING → INFO_GATHERING（信息不完整，自循环）
    INFO_GATHERING → SOLUTION（信息收集完整）
    SOLUTION → CONFIRMATION
    CONFIRMATION → CLOSING（用户满意）
    CONFIRMATION → SOLUTION（用户不满意，重新处理）
    CLOSING → None（对话终止）
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)


class ServiceState(Enum):
    """客服对话的状态枚举。每个状态对应一个明确的处理阶段。"""
    GREETING         = "问候"
    INTENT_DETECTION = "意图识别"
    INFO_GATHERING   = "信息收集"
    SOLUTION         = "方案提供"
    CONFIRMATION     = "确认"
    CLOSING          = "结束"


@sgl.function
def customer_service_fsm(
    s, user_input: str, current_state: ServiceState
) -> Tuple[str, Optional[ServiceState]]:
    """
    客服状态机的单步处理函数。
    每次调用处理一个状态，返回响应文本和下一个状态。
    状态转换逻辑由select结果或业务规则决定。
    """
    s += sgl.system(
        "你是一位专业的电商客服，处理用户的各类问题。"
    )
    s += sgl.user(
        f"当前状态：{current_state.value}\n"
        f"用户输入：{user_input}"
    )

    if current_state == ServiceState.GREETING:
        s += sgl.assistant(
            "您好！欢迎联系客服，请问有什么可以帮您的？"
        )
        return s.text(), ServiceState.INTENT_DETECTION

    elif current_state == ServiceState.INTENT_DETECTION:
        s += sgl.user("请识别用户意图：")
        s += sgl.assistant(
            sgl.gen("intent",
                    choices=["查询订单", "申请退款",
                             "产品咨询", "投诉建议"])
        )
        # 意图决定下一状态：需要信息的意图转入信息收集
        if s["intent"] in ["查询订单", "申请退款"]:
            next_state = ServiceState.INFO_GATHERING
        else:
            next_state = ServiceState.SOLUTION
        return s["intent"], next_state

    elif current_state == ServiceState.INFO_GATHERING:
        s += sgl.user("请向用户收集必要信息：")
        s += sgl.assistant(
            sgl.gen("questions", max_tokens=100)
        )
        s += sgl.user("信息是否已收集完整？")
        s += sgl.assistant(
            sgl.gen("complete", choices=["是", "否"])
        )
        next_state = (ServiceState.SOLUTION
                      if s["complete"] == "是"
                      else ServiceState.INFO_GATHERING)
        return s["questions"], next_state

    elif current_state == ServiceState.SOLUTION:
        s += sgl.user("请提供解决方案：")
        s += sgl.assistant(
            sgl.gen("solution", max_tokens=200)
        )
        return s["solution"], ServiceState.CONFIRMATION

    elif current_state == ServiceState.CONFIRMATION:
        s += sgl.user("请询问用户是否满意：")
        s += sgl.assistant(
            sgl.gen("confirm", choices=["满意", "不满意"])
        )
        if s["confirm"] == "满意":
            return "感谢您的反馈！", ServiceState.CLOSING
        else:
            return "非常抱歉，让我重新为您处理。", ServiceState.SOLUTION

    else:  # CLOSING
        s += sgl.assistant(
            "感谢您的惠顾，祝您生活愉快！如有其他问题随时联系我们。"
        )
        return s.text(), None  # None表示对话终止


# ── 状态机控制器 ──────────────────────────────────────────────

class CustomerServiceController:
    """
    客服状态机控制器。
    封装状态转换调度逻辑，对外提供统一的 step() 接口，
    维护对话历史日志。
    """

    def __init__(self):
        self.current_state = ServiceState.GREETING
        self.log: List[Dict] = []
        self._step_count = 0

    def step(
        self, user_input: str = ""
    ) -> Tuple[str, bool]:
        """
        执行一步状态机处理。
        返回 (助手回复, 对话是否已终止)。
        """
        state = self.current_state
        state_start = time.time()

        state_obj = customer_service_fsm.run(
            user_input=user_input,
            current_state=state,
        )
        response, next_state = state_obj.ret_value
        elapsed_ms = (time.time() - state_start) * 1000

        self.log.append({
            "step": self._step_count,
            "state": state.value,
            "user_input": user_input[:40] if user_input else "",
            "response_preview": str(response)[:60],
            "next_state": next_state.value if next_state else "终止",
            "latency_ms": elapsed_ms,
        })
        self._step_count += 1

        is_terminal = next_state is None
        if not is_terminal:
            self.current_state = next_state

        return str(response), is_terminal

    def print_log(self):
        """打印状态转换历史日志。"""
        print(f"\n{'步骤':>4}  {'当前状态':>10}  "
              f"{'下一状态':>10}  {'耗时':>8}  回复预览")
        print("-" * 70)
        for entry in self.log:
            print(
                f"  {entry['step']:>4}  "
                f"{entry['state']:>10}  "
                f"{entry['next_state']:>10}  "
                f"{entry['latency_ms']:>6.0f}ms  "
                f"{entry['response_preview'][:25]}..."
            )


def main():
    # ── 演示1：完整对话流程模拟 ───────────────────────────────
    print("=" * 60)
    print("【演示1：客服FSM完整对话流程】")
    print("=" * 60)
    print()

    controller = CustomerServiceController()

    # 模拟用户输入序列（对应各状态的处理）
    dialogue_script = [
        "",                              # GREETING：发起联系
        "我的订单一直没有发货，已经等了两周了",  # INTENT_DETECTION
        "订单号是2024031500001",           # INFO_GATHERING
        "",                              # SOLUTION（信息已收集）
        "",                              # CONFIRMATION
        "",                              # CLOSING
    ]

    step = 0
    print("开始模拟对话：\n")

    while True:
        user_input = (
            dialogue_script[step]
            if step < len(dialogue_script)
            else ""
        )

        if user_input:
            print(f"👤 用户：{user_input}")

        response, is_terminal = controller.step(
            user_input=user_input
        )

        state_arrow = (
            f"{controller.log[-1]['state']} → "
            f"{controller.log[-1]['next_state']}"
        )
        print(f"🤖 客服（{state_arrow}）：")
        response_text = str(response)
        print(
            f"   {response_text[:100]}"
            f"{'...' if len(response_text) > 100 else ''}"
        )
        print()

        step += 1
        if is_terminal:
            print("对话已结束。")
            break
        if step >= len(dialogue_script) + 3:
            # 安全边界，防止测试时无限循环
            print("（达到最大步骤数，停止模拟）")
            break

    # ── 演示2：状态转换历史日志 ───────────────────────────────
    print()
    print("=" * 60)
    print("【演示2：状态转换历史日志】")
    print("=" * 60)
    controller.print_log()

    total_ms = sum(e["latency_ms"] for e in controller.log)
    print(f"\n总步骤数：{len(controller.log)}")
    print(f"总耗时：{total_ms:.0f}ms")
    print(f"平均每步：{total_ms/len(controller.log):.0f}ms")

    # ── 演示3：不同意图路径对比 ───────────────────────────────
    print()
    print("=" * 60)
    print("【演示3：不同意图的状态路径对比】")
    print("=" * 60)
    print()

    test_cases = [
        ("我的包裹三天了还没到，能查一下吗？", "查询订单→信息收集→方案"),
        ("这款产品有没有红色款？", "产品咨询→直接方案"),
    ]

    for user_msg, expected_path in test_cases:
        ctrl = CustomerServiceController()
        # 跳过问候直接进入意图识别
        ctrl.current_state = ServiceState.INTENT_DETECTION
        response, _ = ctrl.step(user_input=user_msg)
        intent = ctrl.log[-1]["next_state"]
        print(f"用户：{user_msg[:35]}...")
        print(f"  期望路径：{expected_path}")
        print(f"  实际下一状态：{intent}")
        print()

    # ── FSM价值说明 ───────────────────────────────────────────
    print("=" * 60)
    print("【状态机工作流的工程价值】")
    print("=" * 60)
    print()
    print("可预测性：任何时刻系统状态明确，行为可预期")
    print("可调试性：状态转换历史清晰，问题定位精确到状态")
    print("合法性  ：非法转换在设计阶段即被排除")
    print("可扩展性：新增状态只需添加枚举值和处理分支")
    print()
    print("适用场景：客服对话、审批流程、教学互动、游戏对话")


if __name__ == "__main__":
    main()
