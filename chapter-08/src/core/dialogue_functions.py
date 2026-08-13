"""
src/core/dialogue_functions.py
所有 SGLang 函数的定义。

重要约束:
  @sgl.function 只能装饰模块级函数，不能用于类的实例方法。
  DialogueManager 等类通过直接调用本模块中的函数来编排流程。

函数列表:
  recognize_intent        - 意图识别（9类）
  extract_slots           - 槽位提取
  ask_missing_slot        - 询问缺失槽位
  decide_tool_call        - 工具调用需求判断
  select_tool             - 工具选择
  extract_tool_args       - 工具参数提取
  generate_travel_plan    - 旅游方案生成
  generate_direct_reply   - 通用对话响应
  generate_context_summary - 上下文历史摘要（供 ContextWindowManager 使用）
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import sglang as sgl

# ────────────────────────────────────────────────────────────────────
# Few-shot 示例（供意图识别和槽位提取使用）
# ────────────────────────────────────────────────────────────────────

_INTENT_FEW_SHOT = [
    # 难例1: 隐含意图
    {
        "input": "有什么好玩的地方",
        "context": "（无历史）",
        "intent": "search_attraction",
        "reason": "询问'好玩的地方'是对景点的隐式查询，不是闲聊",
    },
    {
        "input": "云南吃什么比较有特色",
        "context": "前轮: 您想去云南旅游",
        "intent": "search_attraction",
        "reason": "结合上下文，询问特色美食属于景点/美食搜索",
    },
    # 难例2: 片段式输入
    {
        "input": "下个月初",
        "context": "前轮助手: 请问您计划什么时候出发？",
        "intent": "plan_trip",
        "reason": "结合上下文，这是对出发日期询问的回答，属于 plan_trip 流程",
    },
    {
        "input": "两个人",
        "context": "前轮助手: 有几个人一起去？",
        "intent": "plan_trip",
        "reason": "片段回答，结合上下文是人数信息的提供",
    },
    # 难例3: 意图切换
    {
        "input": "等等，云南现在冷不冷",
        "context": "前轮助手: 为您生成了初步行程方案……",
        "intent": "query_weather",
        "reason": "明确切换为天气查询，应优先响应新意图",
    },
    {
        "input": "先帮我查一下大理的酒店",
        "context": "前轮助手: 请问您有什么旅行风格偏好？",
        "intent": "search_hotel",
        "reason": "用户跳过了当前问题，意图明确切换为酒店查询",
    },
]

_SLOT_NUM_TRAVELERS_EXAMPLES = [
    ("我和老婆两个人", 2),
    ("一家三口", 3),
    ("我们几个朋友，总共四个", 4),
    ("就我自己", 1),
    ("带上父母一共五个", 5),
]

_SLOT_INTERESTS_EXAMPLES = [
    ("喜欢自然风光", ["自然风光"]),
    ("想看古建筑，也想尝尝当地美食", ["古建筑", "地方美食"]),
    ("对什么都感兴趣", ["自然风光", "历史文化", "地方美食"]),
]


def _build_intent_few_shot_text() -> str:
    lines = []
    for i, ex in enumerate(_INTENT_FEW_SHOT, 1):
        lines.append(
            f"示例{i}:\n"
            f"  输入: {ex['input']}\n"
            f"  上下文: {ex['context']}\n"
            f"  意图: {ex['intent']}\n"
            f"  原因: {ex['reason']}"
        )
    return "\n\n".join(lines)


def _build_slot_type_hint_text() -> str:
    num_lines = "\n".join(
        f"  '{src}' → {tgt}" for src, tgt in _SLOT_NUM_TRAVELERS_EXAMPLES
    )
    int_lines = "\n".join(
        f"  '{src}' → {tgt}" for src, tgt in _SLOT_INTERESTS_EXAMPLES
    )
    return (
        f"num_travelers 必须转换为整数，示例:\n{num_lines}\n\n"
        f"interests 必须为列表，示例:\n{int_lines}"
    )


# ────────────────────────────────────────────────────────────────────
# 函数1: 意图识别
# ────────────────────────────────────────────────────────────────────

@sgl.function
def recognize_intent(s, user_input: str, recent_context: List[str]):
    """
    9 类意图识别，输出带置信度的 JSON。
    recent_context: 最近 3 轮对话摘要列表，格式为 ["用户: xxx", "助手: xxx"]。
    输出: {"intent": "类别", "confidence": 0.9, "reason": "简述"}
    """
    ctx_text = (
        "\n".join(f"  - {c}" for c in recent_context)
        if recent_context
        else "  （无历史对话）"
    )
    few_shot_text = _build_intent_few_shot_text()

    s += sgl.system(
        "你是意图识别助手。从以下 9 类中选择最匹配的意图:\n"
        "plan_trip | query_weather | search_attraction | search_hotel | "
        "modify_plan | confirm_plan | help | chitchat | unknown\n\n"
        "特别注意:\n"
        "- 片段式输入（如'下个月'、'两个人'）需结合上下文判断，"
        "通常是 plan_trip 流程中的回答\n"
        "- 隐含意图（如'有什么好吃的'）应识别为 search_attraction 而非 chitchat\n"
        "- 意图切换（如方案讨论中突然问天气）应优先响应新意图\n\n"
        f"Few-shot 参考:\n{few_shot_text}\n\n"
        f"最近 3 轮对话上下文:\n{ctx_text}\n\n"
        '严格输出 JSON（不加任何前缀或解释）: '
        '{"intent":"类别","confidence":0.9,"reason":"简述"}'
    )
    s += sgl.user(f"用户输入: {user_input}")
    s += sgl.assistant(sgl.gen("intent_json", max_tokens=100))


# ────────────────────────────────────────────────────────────────────
# 函数2: 槽位提取
# ────────────────────────────────────────────────────────────────────

@sgl.function
def extract_slots(
    s,
    user_input: str,
    current_prefs_json: str,
    target_slots: Optional[List[str]] = None,
):
    """
    从用户输入中提取旅行相关槽位信息。
    current_prefs_json: 当前槽位状态 JSON，避免重复提取已有信息。
    target_slots: 可选，限定重点提取的槽位，减少注意力分散。
    输出: JSON 对象，只含新提取的槽位；无新信息则输出 {}
    """
    focus_hint = (
        f"\n当前优先提取: {', '.join(target_slots)}" if target_slots else ""
    )
    type_hint = _build_slot_type_hint_text()

    s += sgl.system(
        "你是信息提取助手，从用户输入中提取旅行相关槽位。\n\n"
        f"类型规则（重要）:\n{type_hint}\n\n"
        "start_date 和 end_date 保留原始描述，不要转换为 YYYY-MM-DD 格式。\n\n"
        f"当前已有信息:\n{current_prefs_json}{focus_hint}\n\n"
        "提取规则:\n"
        "1. 只提取有新信息的槽位，不重复已有信息\n"
        "2. 一句话中可能包含多个槽位，全部提取\n"
        "3. 无新信息时输出空对象 {}\n"
        "只输出 JSON，不加任何解释。"
    )
    s += sgl.user(f"用户输入: {user_input}")
    s += sgl.assistant(sgl.gen("slots_json", max_tokens=200))


# ────────────────────────────────────────────────────────────────────
# 函数3: 询问缺失槽位
# ────────────────────────────────────────────────────────────────────

@sgl.function
def ask_missing_slot(
    s,
    slot_name: str,
    known_info_text: str,
    dialogue_context: str,
):
    """
    生成自然的槽位询问语句。
    known_info_text: 已知信息的文本摘要（来自 TravelPreferences.to_summary_text()）。
    dialogue_context: 最近 2 轮对话摘要，提供对话连贯性上下文。
    设计目标: 基于已知信息生成上下文相关询问，而非固定模板。
    """
    slot_desc_map = {
        "destination": "目的地",
        "start_date": "出发日期",
        "end_date": "返回/行程天数",
        "num_travelers": "出行人数",
        "budget": "预算范围",
        "accommodation_level": "住宿偏好",
        "travel_style": "旅行风格",
        "interests": "主要兴趣点",
    }
    slot_desc = slot_desc_map.get(slot_name, slot_name)

    s += sgl.system(
        f"你是旅游规划助手，需要询问用户的{slot_desc}信息。\n\n"
        f"已知信息: {known_info_text}\n"
        f"对话上下文: {dialogue_context}\n\n"
        "要求:\n"
        "1. 用自然口语询问，避免填表式生硬问法\n"
        "2. 结合已知信息使问题更有针对性\n"
        "   （例如目的地已知为云南时，询问日期可说'云南四季都适合，"
        "您打算哪个月去？'）\n"
        "3. 对于有明显选项的槽位（如住宿档次），可以列举选项供选择\n"
        "4. 控制在 2 句话以内，简洁自然"
    )
    s += sgl.user(f"请询问用户的{slot_desc}")
    s += sgl.assistant(sgl.gen("question", max_tokens=120))


# ────────────────────────────────────────────────────────────────────
# 函数4: 工具调用需求判断
# ────────────────────────────────────────────────────────────────────

@sgl.function
def should_call_tool(
    s,
    user_input: str,
    context_summary: str,
    available_tools_desc: str,
):
    """
    判断当前用户输入是否需要调用外部工具。
    倾向策略: 不确定时输出 YES（调用工具比不调用更安全，避免给出过时信息）。
    输出: YES 或 NO（只输出这两个词，不加解释）
    """
    s += sgl.system(
        "判断是否需要调用外部工具获取实时信息。\n\n"
        f"可用工具:\n{available_tools_desc}\n\n"
        "需要工具的情况（倾向 YES）:\n"
        "- 查询天气、气温、降雨等实时气象信息\n"
        "- 搜索景点、餐厅、购物等地点信息\n"
        "- 查询酒店、住宿的价格和可用性\n\n"
        "不需要工具的情况:\n"
        "- 一般性旅行建议或常识性问题\n"
        "- 用户只是确认或闲聊\n"
        "- 信息收集阶段（只是询问用户偏好，不需要实时数据）\n\n"
        f"对话上下文: {context_summary}\n\n"
        "只输出 YES 或 NO，不加任何解释。"
    )
    s += sgl.user(f"用户输入: {user_input}")
    s += sgl.assistant(sgl.gen("decision", max_tokens=5))


# ────────────────────────────────────────────────────────────────────
# 函数5: 工具选择
# ────────────────────────────────────────────────────────────────────

@sgl.function
def select_tool(
    s,
    user_input: str,
    available_tools_json: str,
    context_summary: str,
):
    """
    从可用工具中选择最合适的一个。
    available_tools_json: 所有工具 Schema 的 JSON 列表。
    输出: 工具名称字符串（如 get_weather），不加任何解释。
    """
    s += sgl.system(
        "从可用工具中选择最适合的一个来响应用户需求。\n\n"
        f"可用工具（JSON 格式）:\n{available_tools_json}\n\n"
        f"对话上下文: {context_summary}\n\n"
        "只输出工具的 name 字段值，不加任何解释。\n"
        "例如: get_weather"
    )
    s += sgl.user(f"用户需求: {user_input}")
    s += sgl.assistant(sgl.gen("tool_name", max_tokens=30))


# ────────────────────────────────────────────────────────────────────
# 函数6: 工具参数提取
# ────────────────────────────────────────────────────────────────────

@sgl.function
def extract_tool_args(
    s,
    tool_name: str,
    tool_schema_json: str,
    user_input: str,
    context_state_json: str,
):
    """
    从用户输入和对话上下文中提取工具调用参数。
    context_state_json: 当前槽位状态 JSON，用于自动填充可推断的参数
    （如目的地已知时，天气查询的 city 参数可直接从 destination 获取）。
    输出: JSON 参数对象；确实无法确定的参数直接省略，不要伪造。
    """
    s += sgl.system(
        f"为工具 '{tool_name}' 提取调用参数。\n\n"
        f"工具参数说明:\n{tool_schema_json}\n\n"
        f"当前已知信息（可用于推断参数）:\n{context_state_json}\n\n"
        "提取规则:\n"
        "1. 优先从用户输入中直接提取\n"
        "2. 用户输入中缺少的必填参数，尝试从已知信息中推断\n"
        "   （如 search_hotel 需要 check_in，若已知 start_date，则使用之）\n"
        "3. 确实无法确定的参数，直接省略，不要伪造或猜测\n"
        "只输出 JSON 参数对象，不加任何解释。"
    )
    s += sgl.user(f"用户输入: {user_input}\n请提取参数:")
    s += sgl.assistant(sgl.gen("args_json", max_tokens=250))


# ────────────────────────────────────────────────────────────────────
# 函数7: 旅游方案生成
# ────────────────────────────────────────────────────────────────────

@sgl.function
def generate_travel_plan(
    s,
    preferences_json: str,
    tool_results_json: str,
    dialogue_context: str,
):
    """
    基于完整槽位信息和工具查询结果生成个性化旅游方案。
    preferences_json: TravelPreferences 的 JSON 表示。
    tool_results_json: 天气/景点/酒店查询结果的 JSON 表示。
    dialogue_context: 最近几轮对话摘要，了解用户的特殊需求表达。
    """
    s += sgl.system(
        "你是专业旅游规划师，根据用户偏好和实时查询数据生成个性化方案。\n\n"
        f"用户偏好:\n{preferences_json}\n\n"
        f"实时查询数据:\n{tool_results_json}\n\n"
        f"对话上下文（了解特殊需求）:\n{dialogue_context}\n\n"
        "方案格式要求:\n"
        "1. 按天列出行程（第X天: 地点 + 主要活动，各天内容不重复）\n"
        "2. 推荐 2-3 家酒店（含价格范围和核心特色）\n"
        "3. 预算分配建议（交通/住宿/餐饮/景点票务）\n"
        "4. 2-3 条实用注意事项\n"
        "语言: 亲切自然，避免过于正式的模板感"
    )
    s += sgl.user("请生成完整的旅游规划方案")
    s += sgl.assistant(sgl.gen("plan", max_tokens=1500))


# ────────────────────────────────────────────────────────────────────
# 函数8: 通用对话响应
# ────────────────────────────────────────────────────────────────────

@sgl.function
def generate_direct_reply(
    s,
    system_prompt: str,
    user_input: str,
    history_messages: List[Dict[str, str]],
):
    """
    通用对话响应函数，用于闲聊、帮助、简单确认等非规划意图。
    history_messages: [{"role": "user"/"assistant", "content": "..."}] 格式。
    """
    s += sgl.system(system_prompt)
    for msg in history_messages:
        if msg["role"] == "user":
            s += sgl.user(msg["content"])
        else:
            s += sgl.assistant(msg["content"])
    s += sgl.user(user_input)
    s += sgl.assistant(sgl.gen("response", max_tokens=500))


# ────────────────────────────────────────────────────────────────────
# 函数9: 上下文历史摘要（供 ContextWindowManager 使用）
# ────────────────────────────────────────────────────────────────────

@sgl.function
def generate_context_summary(
    s,
    early_dialogue_text: str,
    current_slots_json: str,
):
    """
    为滑出上下文窗口的早期对话历史生成压缩摘要。
    early_dialogue_text: 需要压缩的早期对话文本（格式: "用户: xxx\n助手: xxx\n..."）。
    current_slots_json: 当前槽位状态，避免在摘要中重复已记录的槽位信息。
    输出目标: 150 字以内，保留槽位外的特殊需求/用户拒绝项/工具关键结论。
    """
    s += sgl.system(
        "你是对话摘要助手，将历史对话压缩为简洁摘要。\n\n"
        f"已确认的槽位信息（不需要在摘要中重复）:\n{current_slots_json}\n\n"
        "摘要内容应包含:\n"
        "1. 用户表达的特殊需求或偏好（槽位之外的信息）\n"
        "2. 工具查询的关键结论（如天气好坏、景点评价）\n"
        "3. 用户明确拒绝的选项（避免重复推荐）\n\n"
        "摘要内容不应包含:\n"
        "- 已在槽位中记录的信息（目的地/日期/人数/预算等）\n"
        "- 礼貌性寒暄\n"
        "- 被后续轮次覆盖的早期意见\n\n"
        "控制在 150 字以内，用简洁的第三人称描述。"
    )
    s += sgl.user(f"待压缩的历史对话:\n{early_dialogue_text}")
    s += sgl.assistant(sgl.gen("summary", max_tokens=200))
