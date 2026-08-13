# src/parsers/base_parser.py
# 文档解析基础模块
# 设计原则：只做文本提取，不做信息理解。
# 无论输入格式如何，输出契约只有一条：返回干净的 Unicode 字符串。
# 不在解析层做任何结构判断，避免规则逻辑与 LLM 提取逻辑的混合。

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


def normalize_text(raw: str) -> str:
    """
    对提取的原始文本做规范化处理。

    处理内容：
    - 统一换行符为 \\n
    - 合并三个以上连续空行为两个（保留段落间距）
    - 全角数字和字母转半角（简历中常见的排版习惯）
    - 去除行首行尾多余空格
    - 过滤明显的页码行（纯数字或"- N -"格式的独立行）
    """
    # 统一换行符
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 合并过多连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 全角转半角映射（数字 + 大写字母 + 小写字母）
    full2half = str.maketrans(
        "０１２３４５６７８９"
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz",
    )
    text = text.translate(full2half)

    # 过滤页码行：独立一行且只含数字或"- N -"格式
    lines = text.split("\n")
    lines = [
        line for line in lines
        if not re.fullmatch(r"\s*[-–—]?\s*\d+\s*[-–—]?\s*", line)
    ]

    # 去除每行的首尾空白
    lines = [line.strip() for line in lines]

    return "\n".join(lines).strip()


class BaseParser(ABC):
    """
    文档解析器抽象基类。
    所有格式的解析器都继承此类，保证接口一致性。
    """

    @abstractmethod
    def parse(self, file_path: str | Path) -> str:
        """
        解析文档，返回规范化后的纯文本。

        Args:
            file_path: 文档文件路径

        Returns:
            规范化后的纯文本字符串

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不被该解析器支持
        """
        ...

    def safe_parse(self, file_path: str | Path) -> Optional[str]:
        """
        带异常捕获的安全解析。
        解析失败时返回 None 并记录日志，而非抛出异常。
        批量处理时使用此方法，避免单个文件错误中断整个批次。
        """
        try:
            return self.parse(file_path)
        except Exception as e:
            print(f"[解析失败] {file_path}: {e}")
            return None


class DocumentParser:
    """
    统一文档解析器。
    根据文件扩展名自动选择对应的解析器。
    支持 .pdf、.docx、.txt 三种格式。
    """

    def __init__(self):
        # 延迟导入，避免在不需要某种格式时也要求安装对应依赖
        self._parsers: dict = {}

    def _get_parser(self, suffix: str) -> BaseParser:
        """根据文件扩展名获取或创建对应解析器"""
        suffix = suffix.lower()
        if suffix not in self._parsers:
            if suffix == ".pdf":
                from .pdf_parser import PDFParser
                self._parsers[suffix] = PDFParser()
            elif suffix == ".docx":
                from .docx_parser import DOCXParser
                self._parsers[suffix] = DOCXParser()
            elif suffix == ".txt":
                from .txt_parser import TXTParser
                self._parsers[suffix] = TXTParser()
            else:
                raise ValueError(f"不支持的文件格式：{suffix}")
        return self._parsers[suffix]

    def parse(self, file_path: str | Path) -> str:
        """解析任意支持格式的文档"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        parser = self._get_parser(path.suffix)
        return parser.parse(path)

    def safe_parse(self, file_path: str | Path) -> Optional[str]:
        """带异常捕获的安全解析"""
        try:
            return self.parse(file_path)
        except Exception as e:
            print(f"[解析失败] {file_path}: {e}")
            return None

    def parse_directory(
        self,
        directory: str | Path,
        extensions: list[str] = None
    ) -> dict[str, Optional[str]]:
        """
        解析目录下所有支持格式的文档。

        Args:
            directory: 目录路径
            extensions: 要处理的扩展名列表，默认为 ['.pdf', '.docx', '.txt']

        Returns:
            {文件名: 文本内容} 字典，解析失败的文件值为 None
        """
        if extensions is None:
            extensions = [".pdf", ".docx", ".txt"]

        directory = Path(directory)
        results = {}

        for ext in extensions:
            for file_path in sorted(directory.glob(f"*{ext}")):
                results[file_path.name] = self.safe_parse(file_path)

        return results
