"""PDF/text extraction and local normalization (PyMuPDF)."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from knowledge_engine.src.state import ScrapedDocument

from knowledge_engine.ui.run_log import trace

_REFERENCES_SPLIT_RE = re.compile(
    r"\n\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Bibliografía|"
    r"Works Cited|LITERATURE)\s*\n",
    re.I,
)
_PAGE_NUM_ONLY_RE = re.compile(r"^\s*\d{1,4}\s*$")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


class CleanedDocument(BaseModel):
    doc_id: str
    title: str = ""
    clean_text: str = ""
    source_url: str = ""
    is_pdf: bool = False

    def to_scraped_document(self) -> "ScrapedDocument":
        from knowledge_engine.src.state import ScrapedDocument

        return ScrapedDocument(
            doc_id=self.doc_id,
            source_url=self.source_url,
            source_type="academic_pdf" if self.is_pdf else "trafilatura",
            raw_markdown=self.clean_text,
            title=self.title,
            is_pdf=self.is_pdf,
            cosine_dedup_passed=False,
        )


def _doc_id(url: str, prefix: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_clean_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = _HYPHEN_BREAK_RE.sub(r"\1\2", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_references_section(text: str) -> str:
    parts = _REFERENCES_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) == 2 and len(parts[0].strip()) > 400:
        return parts[0].strip()
    return text


def strip_headers_footers_pages(lines: list[str]) -> list[str]:
    """Remove isolated page numbers and very short repeated lines."""
    if not lines:
        return lines
    freq: dict[str, int] = {}
    for line in lines:
        s = line.strip()
        if 0 < len(s) < 80:
            freq[s] = freq.get(s, 0) + 1
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if _PAGE_NUM_ONLY_RE.match(s):
            continue
        if len(s) < 80 and freq.get(s, 0) >= 3:
            continue
        out.append(line)
    return out


def clean_pdf_bytes(
    pdf_bytes: bytes,
    source_url: str = "",
    title: str = "",
) -> Optional[CleanedDocument]:
    if not pdf_bytes or len(pdf_bytes) < 100:
        return None
    try:
        import fitz  # PyMuPDF
    except ImportError:
        trace("CLEANER ✗ PyMuPDF (fitz) not installed")
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        trace(f"CLEANER ✗ PDF open | {exc}")
        return None

    page_lines: list[str] = []
    meta_title = title
    meta = doc.metadata or {}
    if meta.get("title"):
        meta_title = str(meta["title"]).strip() or meta_title

    for page in doc:
        page_lines.extend(page.get_text("text").splitlines())
        page_lines.append("")

    doc.close()
    page_lines = strip_headers_footers_pages(page_lines)
    raw = "\n".join(page_lines)
    raw = normalize_clean_text(raw)
    raw = strip_references_section(raw)

    if len(raw) < 80:
        trace("CLEANER ✗ PDF too short after clean")
        return None

    trace(f"CLEANER ✓ PDF fitz | {len(raw)} chars")
    return CleanedDocument(
        doc_id=_doc_id(source_url or meta_title, "pdf"),
        title=meta_title or "academic-paper",
        clean_text=raw[:500_000],
        source_url=source_url,
        is_pdf=True,
    )


def clean_text_document(
    html_or_text: str,
    source_url: str = "",
    title: str = "",
    is_pdf: bool = False,
) -> Optional[CleanedDocument]:
    text = html_or_text or ""
    if "<html" in text.lower() or "<body" in text.lower():
        try:
            import trafilatura

            extracted = trafilatura.extract(
                text,
                include_comments=False,
                include_tables=True,
                output_format="markdown",
            )
            if extracted:
                text = extracted
        except Exception:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)

    text = normalize_clean_text(text)
    text = strip_references_section(text)
    if len(text) < 80:
        return None

    trace(f"CLEANER ✓ text | {len(text)} chars")
    return CleanedDocument(
        doc_id=_doc_id(source_url or title, "txt"),
        title=title or "document",
        clean_text=text[:500_000],
        source_url=source_url,
        is_pdf=is_pdf,
    )
