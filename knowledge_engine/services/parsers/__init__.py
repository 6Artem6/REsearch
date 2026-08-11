"""Парсеры статей."""

from knowledge_engine.services.parsers.base import ExtractedImage
from knowledge_engine.services.parsers.html_parser import HtmlArticleParser
from knowledge_engine.services.parsers.md_parser import MarkdownArticleParser
from knowledge_engine.services.parsers.pdf_parser import PdfArticleParser

__all__ = [
    "ExtractedImage",
    "HtmlArticleParser",
    "MarkdownArticleParser",
    "PdfArticleParser",
]
