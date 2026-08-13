"""
src/assistants/education_bot.py
教育陪伴机器人：个性化概念解释、自适应练习生成、建设性反馈。

架构特点（与旅游规划助手的差异）:
  - 学生知识掌握状态需跨会话持久化（TTL 30天，而非30分钟）
  - 采用"三明治反馈"结构（肯定→改进→鼓励），避免直接否定
  - 苏格拉底式追问模式：解释后主动提出思考问题等待回答

复用: SessionManager, ContextWindowManager, StreamingManager
定制: LearningSession 数据模型 + 4 个领域 @sgl.function
"""

from __future__ import annotations

from typing import List, Optional

import sglang as sgl


# ────────────────────────────────────────────────────────────────────
# 教育机器人 SGLang 函数（模块级）
# ────────────────────────────────────────────────────────────────────

@sgl.function
def explain_concept(
    s,
    concept: str,
    subject: str,
    student_level: str,
    mastery_history: str,
    learning_style: str = "混合型",
):
    """
    个性化概念解释。
    mastery_history: 学生对相关知识点的历史掌握情况（JSON字符串），
                     决定讲解起点和深度（从已掌握的知识出发建立连接）。
    learning_style: 视觉型/听觉型/动觉型/阅读型，影响例子选择方式。
    """
    style_hint = {
        "视觉型": "多使用类比、图示描述和空间关系解释",
        "听觉型": "多使用节奏、韵律和声音类比解释",
        "动觉型": "多使用动手操作的例子，强调步骤和过程",
        "阅读型": "多使用精确的文字定义和分类说明",
        "混合型": "综合使用多种方式，注重全面性",
    }.get(learning_style, "综合使用多种方式")

    s += sgl.system(
        f"你是耐心的{subject}辅导老师，面向{student_level}学生。\n\n"
        f"学生对相关知识点的掌握情况:\n{mastery_history}\n\n"
        f"讲解风格要求: {style_hint}\n\n"
        "讲解结构（严格遵守）:\n"
        "1. 【核心概念】: 用 1-2 句话定义（从学生已掌握的知识出发）\n"
        "2. 【生活类比】: 用一个贴近该年级生活的例子说明\n"
        "3. 【应用场景】: 这个概念在哪些实际情况中有用\n"
        "4. 【常见误区】: 同龄学生最容易理解错的地方\n"
        "5. 【思考问题】: 提出 1 个检验理解的问题，等待学生回答\n\n"
        "语气: 温和鼓励，避免'这很简单'等打击性表达，"
        "避免'你必须'等命令性语气"
    )
    s += sgl.user(f"请解释{subject}中的: {concept}")
    s += sgl.assistant(sgl.gen("explanation", max_tokens=800))


@sgl.function
def evaluate_student_answer(
    s,
    question: str,
    student_answer: str,
    reference_answer: str,
    student_level: str,
):
    """
    三明治反馈结构: 肯定→改进→鼓励。
    严禁使用直接否定表述（'错了'、'不对'等），
    改用'可以这样理解……'等引导性语言。
    """
    s += sgl.system(
        f"你是温和的{student_level}辅导老师，为学生的回答提供建设性反馈。\n\n"
        "反馈结构（严格遵守，不省略任何一段）:\n"
        "1. 【肯定】: 先指出回答中正确或有价值的部分（至少找到 1 点）\n"
        "2. 【改进】: 用'可以这样理解……'而非'你错了……'指出需要改进处\n"
        "3. 【正确思路】: 给出完整正确的理解（注重思路，不只是答案）\n"
        "4. 【鼓励】: 以一句鼓励性的话结束\n\n"
        "严格禁止: '错了'、'不对'、'这是错误的'、'完全错误'等直接否定表述"
    )
    s += sgl.user(
        f"题目: {question}\n\n"
        f"学生回答: {student_answer}\n\n"
        f"参考答案: {reference_answer}\n\n"
        "请提供反馈:"
    )
    s += sgl.assistant(sgl.gen("feedback", max_tokens=600))


@sgl.function
def generate_adaptive_exercise(
    s,
    topic: str,
    difficulty: str,
    weak_points: List[str],
    num_questions: int = 3,
):
    """
    自适应练习生成。
    weak_points: 学生薄弱知识点列表，优先在练习题中覆盖。
    difficulty: 入门/初级/中级/高级。
    要求难度递进，同一知识点不出两道题。
    """
    weak_hint = (
        f"\n优先覆盖学生薄弱点: {', '.join(weak_points)}"
        if weak_points
        else ""
    )
    s += sgl.system(
        f"你是出题专家，为{difficulty}水平学生生成{topic}的练习题。{weak_hint}\n\n"
        "每道题格式:\n"
        "题目: [题目内容]\n"
        "类型: [选择题/填空题/问答题]\n"
        "考察知识点: [1-2 个知识点名称]\n"
        "参考答案: [完整答案]\n"
        "解题思路: [步骤化的解题过程]\n\n"
        f"生成 {num_questions} 道题，难度递进（第1题最简单），"
        "同一知识点不重复出现。"
    )
    s += sgl.user(f"请生成 {num_questions} 道{topic}练习题")
    s += sgl.assistant(sgl.gen("exercises", max_tokens=1500))


@sgl.function
def create_study_plan(
    s,
    subject: str,
    current_level: str,
    target_level: str,
    time_available: str,
    weak_points: Optional[List[str]] = None,
):
    """
    个性化学习计划生成。
    从当前水平到目标水平的阶段性路径，包含时间分配和验收标准。
    """
    weak_hint = (
        f"\n薄弱点（优先强化）: {', '.join(weak_points)}"
        if weak_points
        else ""
    )
    s += sgl.system(
        "你是学习规划专家，制定切实可行的个性化学习计划。\n\n"
        "计划要求:\n"
        "1. 基于学生现状，循序渐进\n"
        "2. 合理分配时间（包含练习和复习）\n"
        "3. 每个阶段有明确的验收标准\n"
        "4. 提供具体的学习方法建议\n\n"
        "计划结构:\n"
        "- 阶段划分（2-4 个阶段）\n"
        "- 每阶段: 学习内容 + 时间 + 验收标准\n"
        "- 日常学习建议（每天应做什么）\n"
        "- 注意事项（常见坑和如何避免）"
    )
    s += sgl.user(
        f"学科: {subject}\n"
        f"当前水平: {current_level}\n"
        f"目标水平: {target_level}\n"
        f"可用时间: {time_available}{weak_hint}\n\n"
        "请制定学习计划:"
    )
    s += sgl.assistant(sgl.gen("study_plan", max_tokens=1200))
