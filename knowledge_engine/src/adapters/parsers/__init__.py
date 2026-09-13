"""Парсеры статей."""

from knowledge_engine.src.adapters.parsers.base import ExtractedImage
from knowledge_engine.src.adapters.parsers.html_parser import HtmlArticleParser
from knowledge_engine.src.adapters.parsers.md_parser import MarkdownArticleParser
from knowledge_engine.src.adapters.parsers.pdf_parser import PdfArticleParser

__all__ = [
    "ExtractedImage",
    "HtmlArticleParser",
    "MarkdownArticleParser",
    "PdfArticleParser",
]
