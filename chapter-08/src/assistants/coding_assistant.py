"""
src/assistants/coding_assistant.py
智能编程助手：代码生成、调试、优化、解释。

复用基础设施: SessionManager, ContextWindowManager, StreamingManager
定制内容: CodingSession 数据模型 + 4 个领域特定 @sgl.function
"""

from __future__ import annotations

from typing import List, Optional

import sglang as sgl


# ────────────────────────────────────────────────────────────────────
# 编程助手 SGLang 函数（模块级，符合 @sgl.function 约束）
# ────────────────────────────────────────────────────────────────────

@sgl.function
def generate_code(
    s,
    task_description: str,
    language: str,
    requirements: Optional[str] = None,
    context_code: Optional[str] = None,
):
    """
    代码生成函数。
    context_code 存在时为迭代修改场景，不存在时为从零生成。
    两种场景的系统提示有明显差异以引导模型输出正确类型的响应。
    """
    if context_code:
        system_content = (
            f"你是资深{language}工程师，对以下已有代码进行修改。\n\n"
            f"当前代码:\n```{language.lower()}\n{context_code}\n```\n\n"
            "修改要求:\n"
            "1. 只修改必要的部分，保持代码风格一致\n"
            "2. 在修改处添加注释说明修改原因\n"
            "3. 确保修改后代码完整可运行\n"
            + (f"额外要求: {requirements}" if requirements else "")
        )
    else:
        system_content = (
            f"你是资深{language}工程师，从零生成满足需求的完整代码。\n\n"
            "代码要求:\n"
            "1. 代码完整，不使用省略号或 TODO 占位符\n"
            "2. 为每个函数/类添加简洁注释\n"
            "3. 处理边界情况和常见异常\n"
            "4. 代码块后附一个完整的使用示例\n"
            + (f"额外要求: {requirements}" if requirements else "")
        )
    s += sgl.system(system_content)
    s += sgl.user(f"任务: {task_description}")
    s += sgl.assistant(sgl.gen("code_solution", max_tokens=1500))


@sgl.function
def debug_code(
    s,
    code: str,
    error_message: Optional[str] = None,
    description: Optional[str] = None,
):
    """
    代码调试函数，采用四段式输出格式。
    【问题定位】→【原因分析】→【修复代码】→【预防建议】
    修复代码要求完整，不能省略未修改的部分。
    """
    s += sgl.system(
        "你是代码调试专家。按以下四段格式输出（不省略任何一段）:\n\n"
        "【问题定位】: 指出具体的错误位置（行号/函数名）和错误类型\n"
        "【原因分析】: 从机制层面解释为什么产生这个错误\n"
        "【修复代码】: 给出完整的修复后代码（不是片段）\n"
        "【预防建议】: 如何在未来避免同类问题\n\n"
        "注意: 修复代码必须完整可运行，不能省略未修改的部分。"
    )
    prompt = f"待调试代码:\n```\n{code}\n```"
    if error_message:
        prompt += f"\n\n错误信息:\n{error_message}"
    if description:
        prompt += f"\n\n问题描述: {description}"
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("debug_result", max_tokens=1500))


@sgl.function
def optimize_code(
    s,
    code: str,
    optimization_goal: str = "性能",
    explain_complexity: bool = True,
):
    """
    代码优化函数。
    explain_complexity=True 时要求分析时间/空间复杂度变化（适合算法类优化）。
    """
    complexity_req = (
        "\n4. 对比优化前后的时间/空间复杂度（用大O表示法）"
        if explain_complexity
        else ""
    )
    s += sgl.system(
        f"你是代码优化专家，优化目标: {optimization_goal}。\n\n"
        "输出格式:\n"
        "1. 原有问题分析（列出 2-3 个主要优化点）\n"
        "2. 优化后的完整代码\n"
        f"3. 主要改动说明（每处改动一行解释）{complexity_req}"
    )
    s += sgl.user(f"需要优化的代码:\n```\n{code}\n```")
    s += sgl.assistant(sgl.gen("optimized_result", max_tokens=1500))


@sgl.function
def explain_code(s, code: str, detail_level: str = "详细"):
    """
    代码解释函数，三级详细程度（简要/详细/深入）。
    详细和深入级别会分析算法复杂度和设计模式。
    """
    depth_hint = {
        "简要": "用 3-5 句话概括核心功能，不展开细节",
        "详细": "逐段解释逻辑，说明关键算法和数据结构",
        "深入": "逐行分析，讨论设计模式、复杂度分析和潜在优化点",
    }.get(detail_level, "详细解释代码功能和逻辑")

    s += sgl.system(
        f"你是代码解释专家，解释程度: {detail_level}。\n"
        f"要求: {depth_hint}\n\n"
        "解释结构:\n"
        "1. 整体功能（1-2 句话）\n"
        "2. 核心逻辑逐步解释\n"
        "3. 关键算法/数据结构说明\n"
        "4. 注意事项或潜在问题（如有）"
    )
    s += sgl.user(f"请解释以下代码:\n```\n{code}\n```")
    s += sgl.assistant(sgl.gen("explanation", max_tokens=1000))
