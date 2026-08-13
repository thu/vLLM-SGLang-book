"""
src/assistants/knowledge_base_assistant.py
企业知识库问答助手：基于 RAG 的检索增强生成。

与旅游规划助手的架构差异:
  - 新增 VectorSearchTool（向量检索）注册到 ToolManager
  - answer_with_retrieved_context: 严格限制只基于检索文档回答
  - disambiguate_query: 问题存在多种解释时请用户澄清
  - 追问链（Follow-up Chain）: 通过对话历史解析指代关系

安全原则: 模型不得使用检索文档以外的信息回答，确保答案可追溯。
"""

from __future__ import annotations

from typing import List, Optional

import sglang as sgl


# ────────────────────────────────────────────────────────────────────
# 企业知识库 SGLang 函数（模块级）
# ────────────────────────────────────────────────────────────────────

@sgl.function
def answer_with_retrieved_context(
    s,
    user_question: str,
    retrieved_docs: str,
    user_dept: str,
    conversation_context: str,
):
    """
    基于检索结果回答问题。

    安全约束（严格执行）:
    - 答案必须完全基于 retrieved_docs，不得引用文档外信息
    - 文档中无相关信息时，明确告知并推荐联系对应部门

    conversation_context: 最近 2 轮对话摘要，用于解析追问中的指代关系
    （如"那病假呢"→ 根据上下文理解为"病假的提前申请天数"）
    """
    s += sgl.system(
        f"你是企业内部知识库助手，为{user_dept}部门员工回答问题。\n\n"
        f"最近对话上下文（用于理解追问的指代关系）:\n{conversation_context}\n\n"
        "回答规则（严格遵守，不可违反）:\n"
        "1. 答案必须完全基于以下检索文档，不得使用文档外的任何信息\n"
        "2. 文档中无相关信息时，说明'该信息暂无记录'并推荐联系哪个部门\n"
        "3. 流程类问题用编号步骤列出\n"
        "4. 引用政策/规定时注明来源文档名称\n"
        "5. 不得揣测或扩展文档中未明确说明的内容\n\n"
        f"检索到的相关文档（按相关度排序）:\n{retrieved_docs}"
    )
    s += sgl.user(f"问题: {user_question}")
    s += sgl.assistant(sgl.gen("answer", max_tokens=800))


@sgl.function
def disambiguate_query(
    s,
    ambiguous_question: str,
    recent_topic: str,
    possible_interpretations: List[str],
):
    """
    问题澄清函数。
    当检索系统无法确定用户具体意图时，生成礼貌的澄清请求。
    提供 2-3 个具体选项供用户确认，避免检索错误方向的文档。
    """
    interps_text = "\n".join(
        f"{i+1}. {interp}"
        for i, interp in enumerate(possible_interpretations[:3])
    )
    s += sgl.system(
        "你是知识库助手，需要在检索前确认用户的具体问题方向。\n\n"
        f"用户最近在询问的主题: {recent_topic}\n\n"
        "澄清请求要求:\n"
        "1. 解释问题可能有多种理解\n"
        "2. 列出选项，每个用一句话说明对应的信息类型\n"
        "3. 选项不超过 3 个，避免过多选择\n"
        "4. 措辞自然，不显机械"
    )
    s += sgl.user(
        f"用户问题: {ambiguous_question}\n\n"
        f"可能的理解:\n{interps_text}\n\n"
        "请生成澄清请求:"
    )
    s += sgl.assistant(sgl.gen("clarification", max_tokens=200))


@sgl.function
def answer_without_docs(
    s,
    user_question: str,
    reason: str,
    suggested_contact: str,
):
    """
    无文档匹配时的降级响应。
    给出有帮助的指引，而非空洞的"无相关信息"。
    reason: 无法回答的原因（文档未入库/权限不足/问题超出范围）
    suggested_contact: 推荐用户联系的部门或人员
    """
    s += sgl.system(
        "你是知识库助手，本次检索未找到相关文档。\n\n"
        "回复要求:\n"
        "1. 诚实说明暂时无法提供该信息\n"
        "2. 明确推荐用户联系谁（部门/负责人）\n"
        "3. 如有相关替代信息来源（如外部官网），可以提及\n"
        "4. 语气诚恳，不推卸责任，不显敷衍\n"
        "控制在 3-4 句话内"
    )
    s += sgl.user(
        f"用户问题: {user_question}\n"
        f"无法回答原因: {reason}\n"
        f"推荐联系: {suggested_contact}"
    )
    s += sgl.assistant(sgl.gen("no_docs_response", max_tokens=200))
