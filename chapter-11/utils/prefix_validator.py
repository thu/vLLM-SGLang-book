# utils/prefix_validator.py
# 前缀一致性验证工具（ch11_code_01 核心片段）
#
# 缓存命中率低下的最常见原因之一是前缀的字节级不一致：
# 系统提示词或任务指令模板在不同请求之间存在微小差异，
# 导致缓存键不匹配，整个前缀层的缓存失效。
#
# 本模块在批次正式发送之前对所有请求的前缀进行验证，
# 及早发现并定位不一致问题。
#
# 常见的不一致来源：
#   1. 换行符差异（\r\n vs \n）
#   2. 尾随空格或空行
#   3. Unicode 标准化形式（NFC vs NFD）
#   4. f-string 中意外引入的时间戳或动态字段
#   5. 字符串模板被不同代码路径以不同方式拼接

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from config.prompt_config import (
    STATIC_TASK_TEMPLATES,
    UNIFIED_SYSTEM_PROMPT,
)


@dataclass
class InconsistencyDetail:
    """单处前缀不一致记录。"""
    request_idx: int
    # 第一个字节差异的位置
    diff_position: int
    # 差异位置的字符（repr 格式，便于识别不可见字符）
    diff_char: str
    # 差异所属的前缀层（L1 / L2 / L3）
    diff_layer: str


@dataclass
class PrefixConsistencyReport:
    """前缀一致性验证报告。"""
    total_requests: int
    # 不一致请求数（生产环境中此值应为 0）
    inconsistency_count: int
    # 前 10 处不一致的详细信息
    details: list[InconsistencyDetail] = field(
        default_factory=list
    )
    # L1（系统提示）的哈希值（所有请求应相同）
    l1_hash: str = ""
    # L2 各任务类型的哈希值（同类任务应相同）
    l2_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def is_consistent(self) -> bool:
        return self.inconsistency_count == 0

    def summary(self) -> str:
        lines = [
            f"前缀一致性验证结果：",
            f"  总请求数    ：{self.total_requests}",
            f"  不一致数    ：{self.inconsistency_count}",
            f"  L1 哈希     ：{self.l1_hash[:16]}...",
            f"  L2 任务哈希 ：",
        ]
        for task, h in self.l2_hashes.items():
            lines.append(f"    {task}: {h[:16]}...")
        if not self.is_consistent:
            lines.append(f"  前 10 处差异：")
            for d in self.details[:10]:
                lines.append(
                    f"    请求 #{d.request_idx}，"
                    f"位置 {d.diff_position}，"
                    f"字符 {d.diff_char}，"
                    f"层级 {d.diff_layer}"
                )
        return "\n".join(lines)


def find_first_diff(s1: str, s2: str) -> int:
    """
    返回两个字符串首个不同字符的位置索引。

    若两字符串完全相同，返回 min(len(s1), len(s2))。
    """
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len


def extract_prefix_layers(
    full_prompt: str,
) -> tuple[str, str, str]:
    """
    从完整请求内容中提取 L1、L2、L3 三层前缀。

    返回
    ----
    tuple[str, str, str]
        (l1_text, l2_text, l3_text)
        l1_text：系统提示词部分
        l2_text：任务指令部分（L1 之后至文书正文之前）
        l3_text：文书正文部分（L2 之后至查询结束）

    实现说明：
        通过定位已知的固定分隔字符串来切分各层。
        若提示词结构与 build_stable_prefix() 的输出格式
        不一致，返回元组中对应层为空字符串。
    """
    l1 = UNIFIED_SYSTEM_PROMPT
    l1_end = full_prompt.find(l1)
    if l1_end == -1:
        return ("", "", full_prompt)
    l1_end += len(l1)

    # L2：L1 结束后的下一个 "\n\n" 到文书正文开始前
    # 通过匹配已知 L2 模板来定位
    l2_text = ""
    l2_end = l1_end + 2  # 跳过 "\n\n" 分隔符
    for template in STATIC_TASK_TEMPLATES.values():
        template_start = full_prompt.find(template, l1_end)
        if template_start != -1:
            l2_text = template
            l2_end = template_start + len(template)
            break

    l3_text = full_prompt[l2_end:]
    return (l1, l2_text, l3_text)


def verify_prefix_consistency(
    prompts: list[str],
) -> PrefixConsistencyReport:
    """
    验证批次中所有请求的 L1 和 L2 前缀的字节级一致性。

    验证逻辑：
      L1（系统提示）：所有请求的 L1 内容的 MD5 哈希应相同。
      L2（任务指令）：同类任务的 L2 内容的 MD5 哈希应相同。
      L3（文书正文）：各文书不同，不在本函数验证范围内。

    参数
    ----
    prompts : list[str]
        完整的用户侧请求内容列表
        （build_stable_prefix() 的输出）。

    返回
    ----
    PrefixConsistencyReport
        验证报告。生产环境中 inconsistency_count 应为 0。
    """
    if not prompts:
        return PrefixConsistencyReport(
            total_requests=0,
            inconsistency_count=0,
        )

    # 提取第一个请求的 L1 作为参考基准
    ref_l1, ref_l2, _ = extract_prefix_layers(prompts[0])
    ref_l1_hash = hashlib.md5(
        ref_l1.encode("utf-8")
    ).hexdigest()

    # 记录各任务类型的 L2 哈希（以第一次出现为基准）
    l2_hashes: dict[str, str] = {}
    for task_type, template in STATIC_TASK_TEMPLATES.items():
        l2_hashes[task_type] = hashlib.md5(
            template.encode("utf-8")
        ).hexdigest()

    inconsistencies: list[InconsistencyDetail] = []

    for i, prompt in enumerate(prompts[1:], start=1):
        l1, l2, _ = extract_prefix_layers(prompt)
        current_l1_hash = hashlib.md5(
            l1.encode("utf-8")
        ).hexdigest()

        if current_l1_hash != ref_l1_hash:
            diff_pos = find_first_diff(ref_l1, l1)
            diff_char = (
                repr(l1[diff_pos])
                if diff_pos < len(l1)
                else "<EOF>"
            )
            inconsistencies.append(InconsistencyDetail(
                request_idx=i,
                diff_position=diff_pos,
                diff_char=diff_char,
                diff_layer="L1_system_prompt",
            ))

        # 验证 L2：若能识别任务类型，检查模板是否与预期一致
        if l2:
            current_l2_hash = hashlib.md5(
                l2.encode("utf-8")
            ).hexdigest()
            matched = any(
                current_l2_hash == h
                for h in l2_hashes.values()
            )
            if not matched:
                diff_pos = find_first_diff(ref_l2, l2)
                diff_char = (
                    repr(l2[diff_pos])
                    if diff_pos < len(l2)
                    else "<EOF>"
                )
                inconsistencies.append(InconsistencyDetail(
                    request_idx=i,
                    diff_position=diff_pos,
                    diff_char=diff_char,
                    diff_layer="L2_task_instruction",
                ))

    return PrefixConsistencyReport(
        total_requests=len(prompts),
        inconsistency_count=len(inconsistencies),
        details=inconsistencies[:10],
        l1_hash=ref_l1_hash,
        l2_hashes=l2_hashes,
    )
