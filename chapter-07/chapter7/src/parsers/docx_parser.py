# src/parsers/docx_parser.py
# DOCX 文档解析器
# 使用 python-docx 按段落顺序提取文本。
# 表格内容按行展开为纯文本，保留行间分隔。

from __future__ import annotations

from pathlib import Path

from .base_parser import BaseParser, normalize_text


class DOCXParser(BaseParser):
    """
    DOCX 解析器。
    依赖：pip install "python-docx~=1.1"
    """

    SUPPORTED_SUFFIX = ".docx"

    def parse(self, file_path: str | Path) -> str:
        """
        解析 DOCX 文件，返回规范化纯文本。

        提取顺序：按文档中的实际顺序，段落和表格交替处理。
        表格：每行展开为一行文本，单元格之间用制表符分隔。

        Args:
            file_path: DOCX 文件路径

        Returns:
            规范化纯文本
        """
        try:
            from docx import Document
            from docx.oxml.ns import qn
        except ImportError:
            raise ImportError(
                "未安装 python-docx，请执行：pip install 'python-docx~=1.1'"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if path.suffix.lower() != self.SUPPORTED_SUFFIX:
            raise ValueError(f"不支持的格式：{path.suffix}，期望 .docx")

        doc = Document(str(path))
        text_parts = []

        # 按文档元素顺序遍历（段落和表格）
        # 按文档元素顺序遍历：python-docx 通过 iter_block_items 获取段落和表格的混合顺序
        # 由于 python-docx 没有直接提供有序迭代接口，使用 element.body 遍历 XML 元素
        para_map  = {p._element: p for p in doc.paragraphs}
        table_map = {t._element: t for t in doc.tables}

        for element in doc.element.body:
            if element in para_map:
                # 段落：使用 python-docx Paragraph 对象获取文本
                para_text = para_map[element].text
                if para_text.strip():
                    text_parts.append(para_text.strip())

            elif element in table_map:
                # 表格：展开为文本行
                table_lines = self._extract_table(table_map[element])
                text_parts.extend(table_lines)

        raw = "\n".join(text_parts)
        return normalize_text(raw)

    def _extract_table(self, table) -> list[str]:
        """
        将表格展开为文本行列表。
        每行格式：单元格1\t单元格2\t...
        """
        lines = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                lines.append("\t".join(cells))
        return lines
