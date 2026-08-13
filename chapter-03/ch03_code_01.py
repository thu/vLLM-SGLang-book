"""
第3章 代码示例 01
主题：传统命令式多步推理的痛点演示
场景：用传统OpenAI API实现问题分解-求解-综合的
      三步推理流程，展示字符串拼接、手动解析、
      上下文无法复用等系统性缺陷

运行前提：
    pip install openai
    export OPENAI_API_KEY="your-key"
"""

import json
import os

from openai import OpenAI

# 初始化OpenAI客户端
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "")
)


def solve_question_traditional(question: str) -> str:
    # 手动拼接提示词，格式维护困难，随需求变化极易引入错误
    prompt1 = (
        f"将复杂问题分解为3-5个子问题：\n"
        f"问题：{question}\n以JSON格式输出："
    )
    response1 = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt1}]
    )
    raw = response1.choices[0].message.content
    # 模型输出常包裹在markdown代码块中，需额外剥离
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    try:
        sub_questions = json.loads(
            raw.strip()
        )["sub_questions"]
    except (json.JSONDecodeError, KeyError):
        # 无降级策略，格式略有偏差即直接失败
        return "错误：无法解析子问题"

    sub_answers = []
    for sub_q in sub_questions:
        # 每个子问题独立发起调用，历史上下文无法复用
        prompt2 = f"请简洁回答（不超过100字）：{sub_q}"
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt2}]
        )
        sub_answers.append(resp.choices[0].message.content)

    # 综合提示词随子问题数量线性膨胀，每次调用重新发送全量内容
    prompt3 = f"原问题：{question}\n子问题与答案：\n"
    for i, (q, a) in enumerate(zip(sub_questions, sub_answers)):
        prompt3 += f"{i+1}. {q}\n答：{a}\n\n"
    prompt3 += "请给出最终答案："
    # 第三次API调用：重新发送全量内容，无任何缓存复用
    response3 = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt3}]
    )
    final_answer = response3.choices[0].message.content.strip()

    return final_answer


def _show_painpoints():
    """打印传统方式的系统性痛点分析。"""
    print("=" * 60)
    print("【传统方式的系统性痛点】")
    print("=" * 60)
    print()
    print("痛点1：字符串拼接地狱")
    print("  提示词模板散落在代码各处，修改一处需全局搜索，")
    print("  格式要求与业务逻辑紧密耦合，极难维护。")
    print()
    print("痛点2：JSON解析脆弱")
    print("  模型输出的markdown包裹、多余空格、字段缺失")
    print("  均会导致json.loads()抛出异常，无优雅降级。")
    print()
    print("痛点3：上下文无法复用")
    print("  三次API调用完全独立，后续调用无法利用前序")
    print("  已建立的推理上下文，每次重新发送全量内容。")
    print()
    print("痛点4：性能优化无从下手")
    print("  命令式调用序列对运行时完全不透明，")
    print("  缓存、批处理等优化只能手动实现。")
    print()
    print("对比方案：参见 ch03_code_02.py（SGLang声明式实现）")


def main():
    # 检查API密钥是否配置
    if not os.environ.get("OPENAI_API_KEY"):
        print("警告：未检测到OPENAI_API_KEY环境变量。")
        print("请执行：export OPENAI_API_KEY='your-key'")
        print()
        print("以下展示痛点分析，不实际调用API。")
        print()
        _show_painpoints()
        return

    question = (
        "为什么现代大语言模型在长链推理任务上仍然面临挑战，"
        "有哪些主要的技术路径可以改善这一问题？"
    )

    print("=" * 60)
    print("【传统命令式多步推理演示】")
    print("=" * 60)
    print(f"\n问题：{question}\n")
    print("正在执行三步推理（每步独立API调用）...\n")

    try:
        answer = solve_question_traditional(question)
        print(f"最终答案：\n{answer}")
    except Exception as e:
        print(f"执行出错：{e}")

    print()
    _show_painpoints()


if __name__ == "__main__":
    main()
