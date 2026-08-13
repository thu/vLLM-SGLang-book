# src/parsers/txt_parser.py
# 纯文本解析器
# 处理 .txt 格式简历，主要工作是编码检测和文本规范化。

from __future__ import annotations

from pathlib import Path

from .base_parser import BaseParser, normalize_text


class TXTParser(BaseParser):
    """
    纯文本解析器。
    自动检测常见编码（UTF-8、GBK、GB2312）。
    """

    SUPPORTED_SUFFIX = ".txt"
    ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]

    def parse(self, file_path: str | Path) -> str:
        """
        读取纯文本文件，返回规范化文本。
        依次尝试多种编码，第一个成功的为准。

        Args:
            file_path: 文本文件路径

        Returns:
            规范化纯文本

        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 所有编码均无法解码
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if path.suffix.lower() != self.SUPPORTED_SUFFIX:
            raise ValueError(f"不支持的格式：{path.suffix}，期望 .txt")

        raw = None
        for encoding in self.ENCODINGS:
            try:
                raw = path.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if raw is None:
            raise UnicodeDecodeError(
                "unknown", b"", 0, 1,
                f"无法用以下编码解码 {path.name}：{self.ENCODINGS}"
            )

        return normalize_text(raw)
