# src/parsers/pdf_parser.py
# PDF 文档解析器
# 使用 PyMuPDF（fitz）提取文本。
# 处理两类常见问题：
#   1. 多栏布局：按文字块坐标排序，恢复阅读顺序
#   2. 扫描件 PDF：检测无文本层时给出明确提示

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base_parser import BaseParser, normalize_text


class PDFParser(BaseParser):
    """
    PDF 解析器。
    依赖：pip install "pymupdf~=1.24"
    """

    SUPPORTED_SUFFIX = ".pdf"

    def parse(self, file_path: str | Path) -> str:
        """
        解析 PDF 文件，返回规范化纯文本。

        多栏布局处理策略：
        - 使用 get_text("blocks") 获取文字块及其坐标
        - 按 Y 坐标（垂直位置）排序，再按 X 坐标（水平位置）排序
        - 将同一水平区域内的块视为同行，按从左到右顺序拼接

        Args:
            file_path: PDF 文件路径

        Returns:
            规范化纯文本

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件不是 PDF 格式
            RuntimeError: PDF 为纯扫描件，无文本层
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "未安装 PyMuPDF，请执行：pip install 'pymupdf~=1.24'"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if path.suffix.lower() != self.SUPPORTED_SUFFIX:
            raise ValueError(f"不支持的格式：{path.suffix}，期望 .pdf")

        doc = fitz.open(str(path))
        all_pages_text = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = self._extract_page_text(page)

            if page_text.strip():
                all_pages_text.append(page_text)

        doc.close()

        if not any(t.strip() for t in all_pages_text):
            raise RuntimeError(
                f"{path.name} 似乎是纯扫描件 PDF，无文本层，"
                "需先进行 OCR 处理后再使用本解析器。"
            )

        raw = "\n\n".join(all_pages_text)
        return normalize_text(raw)

    def _extract_page_text(self, page) -> str:
        """
        从单页提取文本，处理多栏布局。

        PyMuPDF 的 get_text("blocks") 返回格式：
        (x0, y0, x1, y1, text, block_no, block_type)
        其中 block_type=0 为文字块，1 为图片块。
        """
        # 获取所有文字块，过滤图片块
        blocks = [
            b for b in page.get_text("blocks")
            if b[6] == 0 and b[4].strip()  # block_type=0 且非空
        ]

        if not blocks:
            return ""

        # 按 Y 坐标主排序、X 坐标次排序（从上到下、从左到右）
        # Y 坐标差距小于行高的 50% 认为是同行
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 15), b[0]))

        lines = [b[4].strip() for b in blocks_sorted if b[4].strip()]
        return "\n".join(lines)

    def get_metadata(self, file_path: str | Path) -> dict:
        """
        提取 PDF 元数据（标题、作者、页数等）。
        用于调试和日志，不参与提取流程。
        """
        try:
            import fitz
        except ImportError:
            return {}

        path = Path(file_path)
        doc = fitz.open(str(path))
        meta = doc.metadata or {}
        result = {
            "title":    meta.get("title", ""),
            "author":   meta.get("author", ""),
            "num_pages": len(doc),
            "file_size_kb": path.stat().st_size // 1024,
        }
        doc.close()
        return result
