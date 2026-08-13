# tests/test_normalizer.py
# 文本标准化流水线单元测试

import pytest

from data.text_normalizer import (
    normalize_document_text,
    normalize_system_prompt,
)


class TestNormalizeDocumentText:
    """normalize_document_text() 的单元测试。"""

    def test_removes_bom(self):
        """测试：去除 UTF-8 BOM 头。"""
        text = "\ufeff合同正文内容"
        result = normalize_document_text(text)
        assert not result.startswith("\ufeff")
        assert result == "合同正文内容"

    def test_nfc_normalization(self):
        """测试：Unicode NFC 标准化（NFD → NFC）。"""
        import unicodedata
        # NFD 形式：é = e + 组合重音符
        nfd_text = unicodedata.normalize("NFD", "café")
        nfc_text = unicodedata.normalize("NFC", "café")
        # NFD 输入经过标准化后应与 NFC 输入结果相同
        assert (
            normalize_document_text(nfd_text)
            == normalize_document_text(nfc_text)
        )

    def test_unifies_line_endings(self):
        """测试：统一换行符（\\r\\n 和 \\r 均转为 \\n）。"""
        crlf = "第一行\r\n第二行\r\n第三行"
        cr = "第一行\r第二行\r第三行"
        lf = "第一行\n第二行\n第三行"
        assert normalize_document_text(crlf) == lf
        assert normalize_document_text(cr) == lf

    def test_strips_trailing_whitespace(self):
        """测试：去除行末尾随空白。"""
        text = "第一行   \n第二行\t\n第三行"
        result = normalize_document_text(text)
        lines = result.split("\n")
        for line in lines:
            assert line == line.rstrip(), (
                f"行末存在尾随空白：{repr(line)}"
            )

    def test_compresses_blank_lines(self):
        """测试：压缩连续空行（3 个及以上压缩为 2 个）。"""
        text = "段落一\n\n\n\n\n段落二"
        result = normalize_document_text(text)
        assert "\n\n\n" not in result
        assert "段落一\n\n段落二" == result

    def test_compresses_spaces(self):
        """测试：压缩行内连续空格。"""
        text = "第一条　甲方应在    规定时间内完成交付"
        result = normalize_document_text(text)
        assert "    " not in result

    def test_strips_leading_trailing_whitespace(self):
        """测试：去除首尾空白。"""
        text = "\n\n  合同正文  \n\n"
        result = normalize_document_text(text)
        assert result == result.strip()

    def test_idempotent(self):
        """测试：幂等性——对已标准化的文本再次处理结果不变。"""
        text = "甲方应履行以下义务：\n\n1. 按时交货\n2. 保证质量"
        once = normalize_document_text(text)
        twice = normalize_document_text(once)
        assert once == twice

    def test_same_source_same_output(self):
        """测试：相同来源的不同格式文本产生相同输出。"""
        variant_a = "合同条款\r\n\r\n第一条　定义\r\n"
        variant_b = "合同条款\n\n第一条　定义\n"
        assert (
            normalize_document_text(variant_a)
            == normalize_document_text(variant_b)
        )


class TestNormalizeSystemPrompt:
    """normalize_system_prompt() 的单元测试。"""

    def test_removes_bom(self):
        prompt = "\ufeff你是一名法律助手。"
        assert not normalize_system_prompt(prompt).startswith(
            "\ufeff"
        )

    def test_strips_extra_newlines(self):
        prompt = "\n\n你是一名法律助手。\n\n"
        result = normalize_system_prompt(prompt)
        assert result == "你是一名法律助手。"
