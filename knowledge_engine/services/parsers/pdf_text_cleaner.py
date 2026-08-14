"""Post-processing OCR/PDF text: hyphen merge, wordninja, ACM/header noise."""

from __future__ import annotations

import re
from typing import Callable

import ftfy
import wordninja

from knowledge_engine.services.article_ingestion.annotated_article_ops import (
    sorted_p_ids,
)
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

_P_TAG_RE = re.compile(r"^\[(P_\d+)\]\s*(.*)$", re.I | re.S)
_FIG_TAG_RE = re.compile(r"^\[FIG_", re.I)

_MATH_INLINE_RE = re.compile(r"\$(?:\\.|[^$\\])+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```|`[^`\n]+`")

_ACM_PERMISSION_RE = re.compile(
    r"permission\s+to\s+make\s+digital\s+or\s+hard\s+copies",
    re.I,
)
_CONFERENCE_RUNNING_RE = re.compile(
    r"(?:ASPLOS|IEEE|ACM|Proceedings)\s*[’']?\s*\d{2,4}|"
    r"March\s+\d{1,2}[-–]\w+\s+\d{4}.*Rotterdam|"
    r"Derrick\s+Quinn\s+et\s+al\.|"
    r"Accelerating\s+Retrieval-Augmented\s+Generation\s+ASPLOS",
    re.I,
)
_PAGE_NUMBER_ONLY_RE = re.compile(r"^\d{1,3}$")
_SECTION_HEADING_RE = re.compile(
    r"^\d+(?:\.\d+)*\s+[A-Z\u0410-\u042f][\w\s\-–—]{2,80}$"
)

_AXIS_ISLAND_RE = re.compile(
    r"^(?:"
    r"\d+\s*"
    r"|(?:queries/sec|qps|latency|throughput|log\s+scale|linear\s+scale)"
    r"(?:\s*\([^)]{0,40}\))?"
    r"|\d+(?:\.\d+)?\s*%"
    r"|[0-9]+(?:\.[0-9]+)?"
    r")$",
    re.I,
)

_CONCAT_WORD_MIN_LEN = 20
_PLACEHOLDER = "\x00KEKEEP{}\x00"


def _protect_regions(text: str) -> tuple[str, list[str]]:
    kept: list[str] = []

    def stash(m: re.Match[str]) -> str:
        kept.append(m.group(0))
        return _PLACEHOLDER.format(len(kept) - 1)

    out = text
    for pat in (_MATH_INLINE_RE, _CODE_FENCE_RE):
        out = pat.sub(stash, out)
    return out, kept


def _restore_regions(text: str, kept: list[str]) -> str:
    for i, fragment in enumerate(kept):
        text = text.replace(_PLACEHOLDER.format(i), fragment)
    return text


def _merge_hyphenation(text: str) -> str:
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    def _join_hyphen_space(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        if len(left) < 4 or not right[:1].islower():
            return m.group(0)
        return left + right

    text = re.sub(r"(\w{2,})-\s+(\w{2,})", _join_hyphen_space, text)
    return text


def _split_concatenated_latin(token: str) -> str:
    if "$" in token or "\\" in token:
        return token
    if not re.fullmatch(r"[a-zA-Z]{%d,}" % _CONCAT_WORD_MIN_LEN, token):
        return token
    return " ".join(wordninja.split(token))


def _apply_wordninja(text: str) -> str:
    protected, kept = _protect_regions(text)

    def repl(m: re.Match[str]) -> str:
        return _split_concatenated_latin(m.group(0))

    protected = re.sub(r"\b[a-zA-Z]{%d,}\b" % _CONCAT_WORD_MIN_LEN, repl, protected)
    return _restore_regions(protected, kept)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_paragraph(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return True
    if _ACM_PERMISSION_RE.search(t):
        return True
    if _PAGE_NUMBER_ONLY_RE.match(t):
        return True
    if (
        len(t) <= 120
        and _CONFERENCE_RUNNING_RE.search(t)
        and not _SECTION_HEADING_RE.match(t)
    ):
        if t.count(".") <= 2:
            return True
    if len(t) <= 48 and _AXIS_ISLAND_RE.match(t):
        return True
    return False


def clean_paragraph_text(text: str) -> str | None:
    """Очистка тела одного абзаца (без тега [P_n]). None = выбросить абзац."""
    raw = (text or "").strip()
    if not raw:
        return None
    if is_noise_paragraph(raw):
        return None

    t = ftfy.fix_text(raw)
    if _SECTION_HEADING_RE.match(t.strip()):
        return _normalize_whitespace(t)

    t = _merge_hyphenation(t)
    t = _apply_wordninja(t)
    t = _normalize_whitespace(t)
    if len(t) < 2 or is_noise_paragraph(t):
        return None
    return t[:4000]


def clean_pdf_text(text: str) -> str:
    """Плоский текст / один blob (без разметки [P_n])."""
    t = ftfy.fix_text(text or "")
    if _ACM_PERMISSION_RE.search(t):
        t = re.sub(
            r"Permission to make digital or hard copies.*?(?=\n\n|\Z)",
            "",
            t,
            flags=re.I | re.S,
        )
    t = _merge_hyphenation(t)
    t = _apply_wordninja(t)
    return _normalize_whitespace(t)


def clean_paragraph_batch(
    paragraph_map: dict[str, str],
    *,
    cleaner: Callable[[str], str | None] | None = None,
) -> tuple[dict[str, str], set[str]]:
    """Пакетная очистка paragraph_map; возвращает (новый map, dropped P_ids)."""
    fn = cleaner or clean_paragraph_text
    out: dict[str, str] = {}
    dropped: set[str] = set()
    for pid in sorted_p_ids(paragraph_map):
        body = paragraph_map.get(pid, "")
        cleaned = fn(body)
        if cleaned is None:
            dropped.add(pid)
            continue
        out[pid] = cleaned
    return out, dropped


def rebuild_annotated_markdown(
    annotated_markdown: str,
    paragraph_map: dict[str, str],
    *,
    dropped_p_ids: set[str] | None = None,
) -> str:
    drop = dropped_p_ids or set()
    lines_out: list[str] = []
    for line in (annotated_markdown or "").split("\n"):
        stripped = line.strip()
        if not stripped:
            if lines_out and lines_out[-1] != "":
                lines_out.append("")
            continue
        m = _P_TAG_RE.match(stripped)
        if m:
            pid = m.group(1).upper()
            if pid.startswith("P_"):
                pid = f"P_{pid[2:]}"
            if pid in drop:
                continue
            body = paragraph_map.get(pid)
            if body is None:
                continue
            lines_out.append(f"[{pid}] {body}")
            continue
        if _FIG_TAG_RE.match(stripped):
            lines_out.append(stripped)
            continue
        lines_out.append(stripped)
    return "\n\n".join(
        block for block in "\n".join(lines_out).split("\n\n") if block.strip()
    ).strip()


def clean_annotated_article(article: AnnotatedArticle) -> AnnotatedArticle:
    """Production hook: чистит paragraph_map и пересобирает annotated_markdown."""
    pmap = article.paragraph_map or {}
    new_pmap, dropped = clean_paragraph_batch(pmap)
    new_pages = {
        k: v for k, v in (article.paragraph_page or {}).items() if k not in dropped
    }
    new_md = rebuild_annotated_markdown(
        article.annotated_markdown or "",
        new_pmap,
        dropped_p_ids=dropped,
    )
    return AnnotatedArticle(
        annotated_markdown=new_md,
        fig_map=dict(article.fig_map or {}),
        paragraph_map=new_pmap,
        page_url=article.page_url or "",
        fig_bytes=dict(article.fig_bytes or {}),
        fig_extract_source=dict(article.fig_extract_source or {}),
        fig_extract_topology=dict(article.fig_extract_topology or {}),
        paragraph_page=new_pages,
        source_pdf_bytes=article.source_pdf_bytes,
    )
