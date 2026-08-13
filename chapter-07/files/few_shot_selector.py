# src/prompts/few_shot_selector.py
# Few-shot 示例管理器
#
# 设计原则（对应正文 7.4.1）：
# 约束解码已保证格式，Few-shot 的价值完全集中在内容层面。
# 示例选取标准：覆盖提取规范中最容易产生歧义的边界情况。
# 三类高价值示例：日期规范化、在职状态、技能分类边界。
#
# 使用策略：从零示例开始，评估后针对高频错误类型逐个添加，
# 通常 2-3 个示例覆盖 80%+ 的歧义情况。

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from src.extractors.sglang_extractor import SYSTEM_PROMPT


# ──────────────────────────────────────────────
# 示例数据定义
# ──────────────────────────────────────────────

@dataclass
class FewShotExample:
    """单个 Few-shot 示例"""
    name:        str          # 示例标识名，用于日志和调试
    description: str          # 示例针对的歧义类型说明
    input:       str          # 示例简历文本
    output:      dict         # 期望的提取结果（dict，将被序列化为 JSON）
    tags:        list[str] = field(default_factory=list)  # 标签，用于按需筛选


# ──────────────────────────────────────────────
# 内置高价值示例库
# ──────────────────────────────────────────────

# 示例1：日期规范化
# 针对歧义：中文日期"2019年7月"应规范化为"2019-07"
DATE_NORMALIZATION_EXAMPLE = FewShotExample(
    name="date_normalization",
    description="中文日期格式规范化为 YYYY-MM",
    tags=["date", "chinese"],
    input="""\
李明
工作经历
2019年7月 - 2021年3月  腾讯科技  后端工程师
- 负责支付系统核心模块开发
- 接口响应时间优化至 50ms 以内""",
    output={
        "basic_info": {"name": "李明"},
        "education": [],
        "work_experience": [
            {
                "company":    "腾讯科技",
                "position":   "后端工程师",
                "start_date": "2019-07",
                "end_date":   "2021-03",
                "responsibilities": [
                    "负责支付系统核心模块开发",
                    "接口响应时间优化至 50ms 以内",
                ],
            }
        ],
        "skills": None,
    },
)

# 示例2：在职状态
# 针对歧义：当前在职的 end_date 应填 "present" 而非当前年份或 null
CURRENT_JOB_EXAMPLE = FewShotExample(
    name="current_job",
    description="当前在职工作经历的 end_date 填写 present",
    tags=["date", "current"],
    input="""\
王芳
邮箱：wangfang@example.com
工作经历
2022年5月至今  字节跳动  高级产品经理
- 负责短视频推荐产品迭代，DAU 提升 20%""",
    output={
        "basic_info": {
            "name":  "王芳",
            "email": "wangfang@example.com",
        },
        "education": [],
        "work_experience": [
            {
                "company":    "字节跳动",
                "position":   "高级产品经理",
                "start_date": "2022-05",
                "end_date":   "present",
                "responsibilities": ["负责短视频推荐产品迭代，DAU 提升 20%"],
            }
        ],
        "skills": None,
    },
)

# 示例3：技能分类边界
# 针对歧义：Kubernetes → tools，Gin → frameworks，TensorFlow → frameworks
SKILL_CLASSIFICATION_EXAMPLE = FewShotExample(
    name="skill_classification",
    description="技能分类边界：Kubernetes 归工具，Gin/TensorFlow 归框架",
    tags=["skill", "classification"],
    input="""\
陈磊
技能
编程：Go、Python
框架：Gin、TensorFlow
数据库：PostgreSQL
工具：Docker、Kubernetes、Git""",
    output={
        "basic_info": {"name": "陈磊"},
        "education": [],
        "work_experience": [],
        "skills": {
            "programming_languages": ["Go", "Python"],
            "frameworks":  ["Gin", "TensorFlow"],
            "databases":   ["PostgreSQL"],
            "tools":       ["Docker", "Kubernetes", "Git"],
        },
    },
)

# 所有内置示例，按推荐添加顺序排列
BUILTIN_EXAMPLES: list[FewShotExample] = [
    DATE_NORMALIZATION_EXAMPLE,
    CURRENT_JOB_EXAMPLE,
    SKILL_CLASSIFICATION_EXAMPLE,
]


# ──────────────────────────────────────────────
# Few-shot 提示构建
# ──────────────────────────────────────────────

def build_few_shot_prompt(
    examples: list[FewShotExample],
    base_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    将示例列表追加到系统提示末尾，构成完整的 Few-shot 提示。

    注意：返回的字符串将作为 system_prompt 传入 extract_resume，
    不同示例组合会产生不同的提示字符串，影响前缀缓存命中率。
    在一个批次内保持提示字符串不变，确保 RadixAttention 正常生效。

    Args:
        examples:    示例列表，建议不超过 3 个
        base_prompt: 基础系统提示

    Returns:
        注入了 Few-shot 示例的完整系统提示
    """
    if not examples:
        return base_prompt

    parts = [base_prompt, "\n以下是提取示例，请参照处理："]

    for i, ex in enumerate(examples, 1):
        output_json = json.dumps(ex.output, ensure_ascii=False, indent=2)
        parts.append(f"\n示例{i}输入：\n{ex.input}")
        parts.append(f"示例{i}输出：\n{output_json}")

    return "\n".join(parts)


def get_prompt_with_examples(
    example_names: Optional[list[str]] = None,
) -> str:
    """
    按名称获取内置示例并构建提示。
    example_names 为 None 时使用全部内置示例。

    Args:
        example_names: 示例名称列表，如 ["date_normalization", "current_job"]
                       传入 None 使用全部内置示例

    Returns:
        完整系统提示字符串
    """
    if example_names is None:
        examples = BUILTIN_EXAMPLES
    else:
        name_map = {ex.name: ex for ex in BUILTIN_EXAMPLES}
        examples = []
        for name in example_names:
            if name in name_map:
                examples.append(name_map[name])
            else:
                available = list(name_map.keys())
                raise ValueError(
                    f"未找到示例：'{name}'，可用示例：{available}"
                )

    return build_few_shot_prompt(examples)


def list_examples() -> list[dict]:
    """列出所有内置示例的摘要信息，用于选择"""
    return [
        {
            "name":        ex.name,
            "description": ex.description,
            "tags":        ex.tags,
        }
        for ex in BUILTIN_EXAMPLES
    ]
