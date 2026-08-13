# src/parsers/__init__.py
from .base_parser import BaseParser, DocumentParser, normalize_text
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .txt_parser import TXTParser

__all__ = [
    "BaseParser",
    "DocumentParser",
    "normalize_text",
    "PDFParser",
    "DOCXParser",
    "TXTParser",
]
