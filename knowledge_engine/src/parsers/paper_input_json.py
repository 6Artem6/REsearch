"""Build structured input_paper_json from PyMuPDF PDF bytes or plain text."""

from __future__ import annotations

import re

import fitz

from knowledge_engine.src.parsers.paper_structure_schema import (
    InputPaperJson,
    InputPaperPage,
    InputPaperParagraph,
)

_MIN_PARA_CHARS = 2
_HEADING_MAX_LEN = 120
_REF_HEADING_RE = re.compile(
    r"^(references|bibliography|literature|acknowledgments?|appendix)\b",
    re.I,
)
_REF_SECTION_TITLE_RE = re.compile(
    r"^(references|bibliography|literature)\s*$",
    re.I,
)
_TITLE_SUPPRESSED_RE = re.compile(r"title suppressed", re.I)
_PAGE_NUM_ONLY_RE = re.compile(r"^\d{1,3}$")
_AUTHOR_ET_AL_LINE_RE = re.compile(
    r"^(?:\d{1,3}\s+)?[A-Z][\w\-.]*(?:\s+[A-Z][\w\-.]*)+\s+et\s+al\.?\s*\d{0,3}$",
    re.I,
)
_PAGE_NUM_AUTHOR_RE = re.compile(
    r"^\d{1,3}\s+[A-Z]\.\s+[\w\s]+\s+et\s+al\.?$",
    re.I,
)
_BIB_ENTRY_START_RE = re.compile(
    r"^1\.\s+[A-Z][\w\-']",
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_running_header_footer(text: str) -> bool:
    """LNCS/IEEE/ACM page headers, footers, and suppressed titles."""
    t = _normalize_space(text)
    if not t:
        return True
    if _TITLE_SUPPRESSED_RE.search(t):
        return True
    if _PAGE_NUM_ONLY_RE.match(t):
        return True
    if _AUTHOR_ET_AL_LINE_RE.match(t):
        return True
    if _PAGE_NUM_AUTHOR_RE.match(t):
        return True
    if len(t) <= 40 and re.match(r"^\d+\s+[A-Z]\.", t):
        return True
    return False


def sanitize_section_title(title: str, last_valid: str) -> str:
    t = _normalize_space(title)
    if not t or is_running_header_footer(t) or _TITLE_SUPPRESSED_RE.search(t):
        return last_valid
    return t[:200]


def is_references_section_start(section_title: str, text: str) -> bool:
    """True when paragraph begins the References/Bibliography block."""
    st = _normalize_space(section_title)
    tx = _normalize_space(text)
    if _REF_SECTION_TITLE_RE.match(st):
        return True
    if _REF_SECTION_TITLE_RE.match(tx) and len(tx) < 80:
        return True
    if st.lower() == "references" and _BIB_ENTRY_START_RE.match(tx):
        return True
    if _BIB_ENTRY_START_RE.match(tx) and len(tx) > 120:
        low = tx.lower()
        if any(
            x in low
            for x in (
                "arxiv",
                "proceedings",
                "journal",
                "conference",
                "neural information",
                "doi",
                "isbn",
                "http",
            )
        ):
            return True
    return False


def find_first_references_paragraph_id(paper: InputPaperJson) -> int | None:
    rng = find_references_range_from_paper(paper)
    return rng[0] if rng else None


_APPENDIX_HEADING_RE = re.compile(
    r"^(appendix|supplementary(?:\s+material)?|supplemental)\b",
    re.I,
)
_APPENDIX_LETTER_RE = re.compile(
    r"^appendix\s+[A-Z]\b",
    re.I,
)
_LETTER_SECTION_RE = re.compile(r"^[A-Z]\.\s+\S")
_NEW_ARTICLE_HEADING_RE = re.compile(
    r"^(?:\d+\.\s+)?(introduction|abstract)\b",
    re.I,
)
_SM_HEADING_RE = re.compile(r"^SM\b", re.I)


def is_post_references_section_heading(section_title: str, text: str) -> bool:
    """Heading that ends the references block (appendix, new article, etc.)."""
    st = _normalize_space(section_title)
    tx = _normalize_space(text)
    for candidate in (st, tx):
        if not candidate or len(candidate) > _HEADING_MAX_LEN:
            continue
        if _APPENDIX_HEADING_RE.match(candidate):
            return True
        if _APPENDIX_LETTER_RE.match(candidate):
            return True
        if _SM_HEADING_RE.match(candidate):
            return True
        if _NEW_ARTICLE_HEADING_RE.match(candidate) and len(candidate) < 80:
            return True
        if _LETTER_SECTION_RE.match(candidate) and len(candidate.split()) <= 12:
            if not _BIB_ENTRY_START_RE.match(candidate):
                return True
    return False


def ordered_paragraphs(paper: InputPaperJson) -> list[InputPaperParagraph]:
    out: list[InputPaperParagraph] = []
    for page in paper.pages:
        out.extend(page.paragraphs)
    return out


def references_range_from_index(
    paragraphs: list[InputPaperParagraph],
    start_index: int,
) -> tuple[int, int]:
    """Inclusive range from start_index through refs block end (or document end)."""
    ref_start_id = paragraphs[start_index].paragraph_id
    ref_end_id = paragraphs[-1].paragraph_id
    for j in range(start_index + 1, len(paragraphs)):
        para = paragraphs[j]
        if is_post_references_section_heading(para.section_title, para.text):
            ref_end_id = paragraphs[j - 1].paragraph_id
            break
    return (ref_start_id, ref_end_id)


def find_references_range(
    paragraphs: list[InputPaperParagraph],
) -> tuple[int, int] | None:
    """Inclusive paragraph id range covering only the references/bibliography block."""
    if not paragraphs:
        return None
    start_index: int | None = None
    for i, para in enumerate(paragraphs):
        if is_references_section_start(para.section_title, para.text):
            start_index = i
            break
    if start_index is None:
        return None
    return references_range_from_index(paragraphs, start_index)


def find_references_range_from_paper(paper: InputPaperJson) -> tuple[int, int] | None:
    return find_references_range(ordered_paragraphs(paper))


def _looks_like_heading(text: str) -> bool:
    t = _normalize_space(text)
    if not t or len(t) > _HEADING_MAX_LEN:
        return False
    if is_running_header_footer(t):
        return False
    if t.endswith(".") and len(t.split()) > 6:
        return False
    if _REF_HEADING_RE.match(t):
        return True
    words = t.split()
    if len(words) <= 8 and (t.isupper() or t[0].isupper()):
        return True
    return False


def _page_text_paragraphs(
    page: fitz.Page, page_no: int, section: str
) -> tuple[str, list[tuple[str, str]]]:
    """Return (updated_section, list of (section_title, text))."""
    out: list[tuple[str, str]] = []
    current_section = section
    last_valid_section = section if section and section != "Body" else ""
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        parts: list[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(str(s.get("text", "")) for s in spans).strip()
            if line_text:
                parts.append(line_text)
        text = _normalize_space(" ".join(parts))
        if len(text) < _MIN_PARA_CHARS:
            continue
        if is_running_header_footer(text):
            continue
        if _looks_like_heading(text):
            candidate = sanitize_section_title(text, last_valid_section)
            if candidate and not is_running_header_footer(candidate):
                current_section = candidate
                last_valid_section = candidate
            continue
        sec = sanitize_section_title(current_section, last_valid_section)
        if not sec:
            sec = last_valid_section or "Body"
        out.append((sec, text))
    return current_section, out


def build_input_paper_json_from_pdf_bytes(pdf_bytes: bytes) -> InputPaperJson:
    if not pdf_bytes or pdf_bytes[:5] != b"%PDF-":
        return InputPaperJson(total_pages=0, pages=[])
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages: list[InputPaperPage] = []
        p_idx = 0
        section = "Body"
        last_valid = ""
        for page_index in range(doc.page_count):
            page_no = page_index + 1
            page = doc[page_index]
            section, pairs = _page_text_paragraphs(page, page_no, section)
            paras: list[InputPaperParagraph] = []
            for sec_title, text in pairs:
                p_idx += 1
                clean_sec = sanitize_section_title(sec_title, last_valid)
                if clean_sec:
                    last_valid = clean_sec
                paras.append(
                    InputPaperParagraph(
                        paragraph_id=p_idx,
                        section_title=clean_sec or last_valid or "Body",
                        text=text[:8000],
                    )
                )
            pages.append(InputPaperPage(page_number=page_no, paragraphs=paras))
        return InputPaperJson(total_pages=doc.page_count, pages=pages)
    finally:
        doc.close()


def build_input_paper_json_from_plain_text(text: str) -> InputPaperJson:
    """Fallback when only markdown/plain body is available (single virtual page)."""
    raw = (text or "").strip()
    if not raw:
        return InputPaperJson(total_pages=1, pages=[])
    section = "Body"
    last_valid = ""
    paras: list[InputPaperParagraph] = []
    p_idx = 0
    blocks = re.split(r"\n\s*\n+", raw)
    for block in blocks:
        t = _normalize_space(block)
        if len(t) < _MIN_PARA_CHARS:
            continue
        if is_running_header_footer(t):
            continue
        if _looks_like_heading(t):
            candidate = sanitize_section_title(t, last_valid)
            if candidate:
                section = candidate
                last_valid = candidate
            continue
        p_idx += 1
        clean_sec = sanitize_section_title(section, last_valid)
        if clean_sec:
            last_valid = clean_sec
        paras.append(
            InputPaperParagraph(
                paragraph_id=p_idx,
                section_title=clean_sec or last_valid or "Body",
                text=t[:8000],
            )
        )
    page = InputPaperPage(page_number=1, paragraphs=paras)
    return InputPaperJson(total_pages=1, pages=[page])


def input_paper_json_for_llm(
    paper: InputPaperJson,
    *,
    max_paragraphs: int = 400,
    max_chars_per_para: int = 600,
) -> dict:
    """Shrink payload for Gemini while keeping ids stable for filtering."""
    pages_out: list[dict] = []
    count = 0
    for page in paper.pages:
        paras_out: list[dict] = []
        for para in page.paragraphs:
            if count >= max_paragraphs:
                break
            paras_out.append(
                {
                    "paragraph_id": para.paragraph_id,
                    "section_title": para.section_title,
                    "text": (para.text or "")[:max_chars_per_para],
                }
            )
            count += 1
        if paras_out:
            pages_out.append({"page_number": page.page_number, "paragraphs": paras_out})
        if count >= max_paragraphs:
            break
    return {
        "total_pages": paper.total_pages,
        "pages": pages_out,
    }
