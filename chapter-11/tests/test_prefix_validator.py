# tests/test_prefix_validator.py
# 前缀一致性验证单元测试

import pytest

from config.prompt_config import build_stable_prefix
from data.text_normalizer import normalize_document_text
from utils.prefix_validator import (
    PrefixConsistencyReport,
    find_first_diff,
    verify_prefix_consistency,
)


class TestFindFirstDiff:
    """find_first_diff() 的单元测试。"""

    def test_identical_strings(self):
        """测试：完全相同的字符串返回字符串长度。"""
        s = "完全相同的字符串"
        assert find_first_diff(s, s) == len(s)

    def test_first_char_diff(self):
        """测试：首字符不同时返回 0。"""
        assert find_first_diff("ABC", "XBC") == 0

    def test_middle_diff(self):
        """测试：中间位置不同时返回正确索引。"""
        assert find_first_diff("ABCDE", "ABXDE") == 2

    def test_different_lengths_same_prefix(self):
        """测试：一个是另一个的前缀时，返回较短字符串的长度。"""
        assert find_first_diff("ABC", "ABCDE") == 3

    def test_empty_strings(self):
        """测试：两个空字符串返回 0。"""
        assert find_first_diff("", "") == 0

    def test_invisible_char_diff(self):
        """测试：不可见字符差异（换行符）能被检测到。"""
        s1 = "合同条款\n附件"
        s2 = "合同条款\r附件"
        diff = find_first_diff(s1, s2)
        assert diff == 4  # "合同条款" 之后的换行符位置


class TestVerifyPrefixConsistency:
    """verify_prefix_consistency() 的单元测试。"""

    def _make_prompts(
        self,
        doc_texts: list[str],
        task_type: str = "risk",
    ) -> list[str]:
        return [
            build_stable_prefix(task_type, normalize_document_text(t))
            for t in doc_texts
        ]

    def test_consistent_prompts_pass(self):
        """测试：字节级一致的提示词通过验证。"""
        # 使用相同系统提示和任务指令，不同文书正文
        texts = [
            "合同甲方：北京公司。第一条：双方权利义务如下。",
            "合同乙方：上海公司。第二条：违约责任如下所述。",
        ]
        prompts = self._make_prompts(texts)
        report = verify_prefix_consistency(prompts)
        # L1 和 L2 层完全相同，应通过一致性验证
        assert report.is_consistent

    def test_empty_list(self):
        """测试：空列表返回零不一致。"""
        report = verify_prefix_consistency([])
        assert report.total_requests == 0
        assert report.inconsistency_count == 0
        assert report.is_consistent

    def test_single_prompt(self):
        """测试：单个请求无法进行比较，返回一致。"""
        prompts = self._make_prompts(["单份合同文书内容。"])
        report = verify_prefix_consistency(prompts)
        assert report.is_consistent

    def test_report_has_l1_hash(self):
        """测试：报告中包含 L1 哈希值。"""
        texts = ["文书甲", "文书乙"]
        prompts = self._make_prompts(texts)
        report = verify_prefix_consistency(prompts)
        assert len(report.l1_hash) == 32  # MD5 十六进制长度

    def test_report_has_l2_hashes(self):
        """测试：报告中包含 L2 任务类型哈希。"""
        texts = ["文书甲"]
        prompts = self._make_prompts(texts)
        report = verify_prefix_consistency(prompts)
        assert len(report.l2_hashes) > 0

    def test_summary_is_string(self):
        """测试：summary() 返回非空字符串。"""
        texts = ["文书甲", "文书乙"]
        prompts = self._make_prompts(texts)
        report = verify_prefix_consistency(prompts)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_inconsistent_detected(self):
        """
        测试：手动构造的不一致提示词被检测到。

        通过直接修改提示词字符串（而非通过 build_stable_prefix）
        模拟系统提示词被意外修改的场景。
        """
        texts = ["合同内容甲", "合同内容乙"]
        prompts = self._make_prompts(texts)
        # 手动篡改第二个提示词的系统提示层
        tampered = prompts[1].replace("专业的法律文书", "资深的法律文书")
        mixed = [prompts[0], tampered]
        report = verify_prefix_consistency(mixed)
        # 篡改后的提示词 L1 哈希与基准不同，应检测到不一致
        assert report.inconsistency_count > 0
        assert not report.is_consistent
