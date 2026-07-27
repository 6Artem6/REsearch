"""Linkify DOI, arXiv, and URLs in research text for web UI."""

from __future__ import annotations

import html
import re
from typing import Any

from knowledge_engine.src.processors.source_anchors import (
    expand_source_tags_to_markdown_links,
    linkify_source_anchors_html,
)

_DOI_BARE_RE = re.compile(
    r"(?<![/\w])(10\.\d{4,9}/[^\s<>,;)\]]+)",
    re.I,
)
_DOI_URL_RE = re.compile(r"https?://doi\.org/(10\.\S+)", re.I)
_ARXIV_RE = re.compile(
    r"\barxiv:(\d{4}\.\d{4,5}(?:v\d+)?)\b",
    re.I,
)
_URL_RE = re.compile(r"https?://[^\s<>,;)\]]+")
_DOI_EXTRACT_RE = re.compile(r"(10\.\d{4,9}/[^\s\]<\"')]+)", re.I)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\d.]+(?:v\d+)?)", re.I)

# Gemini / markdown иногда съедают \f \t → rac{, ext{ (не трогать уже валидные \frac / \text)
_LATEX_BROKEN_REPAIRS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\\f)rac\{"), "\\frac{"),
    (re.compile(r"(?<!\\t)ext\{"), "\\text{"),
    (re.compile(r"(?<!\\)mathrm\{"), "\\mathrm{"),
    (re.compile(r"(?<!\\)mathcal\{"), "\\mathcal{"),
    (re.compile(r"(?<!\\)mathbb\{"), "\\mathbb{"),
    (re.compile(r"(?<!\\)cdot(?=\s|$|[{}])"), "\\cdot"),
    (re.compile(r"(?<!\\)times(?=\s|$|[{}])"), "\\times"),
)

_MATH_DISPLAY_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
_MATH_INLINE_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
_LATEX_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _fix_corrupted_text_commands(s: str) -> str:
    """
    LLM/markdown ломает \\text: tab съедает часть команды или остаётся \\t внутри \\text.
    Примеры: _{\\text\\t total}, _{\\t ext{proc}}, ^{\\text proc,i}.
    """
    s = re.sub(r"\\t\s*ext\{", r"\\text{", s)
    # Реальный TAB после \text (съеден \tex)
    s = re.sub(
        r"\\text\t+([a-zA-Z][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z][a-zA-Z0-9_]*)",
        r"\\text{\1},\2",
        s,
    )
    s = re.sub(r"\\text\t+([a-zA-Z][a-zA-Z0-9_]*)", r"\\text{\1}", s)
    s = re.sub(
        r"\\text\\t\s+([a-zA-Z][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\}",
        r"\\text{\1},\2}",
        s,
    )
    s = re.sub(
        r"\\text\\t\s+([a-zA-Z][a-zA-Z0-9_]*)\s*\}",
        r"\\text{\1}}",
        s,
    )
    s = re.sub(r"_\{\t+\\text\{", r"_{\text{", s)
    s = re.sub(r"\^\{\t+\\text\{", r"^{\text{", s)
    s = re.sub(r"\\text\s+([a-zA-Z][a-zA-Z0-9_]*)\s*,", r"\\text{\1},", s)
    s = re.sub(
        r"\\text\s+([a-zA-Z][a-zA-Z0-9_]*)(?![a-zA-Z0-9_\{])",
        r"\\text{\1}",
        s,
    )
    s = re.sub(r"\\t\s+", " ", s)
    return s


def _sanitize_math_inner(inner: str) -> str:
    """Очистка TeX внутри $...$ / $$...$$ перед KaTeX."""
    s = inner
    s = _fix_corrupted_text_commands(s)
    s = _LATEX_CONTROL_RE.sub("", s)
    s = s.replace("\f", "")
    s = re.sub(r"\\f\\frac", r"\\frac", s)
    s = re.sub(r"(?<!\\)f\\frac", r"\\frac", s)
    s = re.sub(r"\\f(?=rac\{)", r"\\", s)
    s = re.sub(r"\\t+\\text", r"\\text", s)
    s = re.sub(r"\t+", " ", s)
    s = re.sub(r"(.{6,}?)\s+\1(?=[\)\}\]\s,;]|$)", r"\1", s)
    for pattern, repl in _LATEX_BROKEN_REPAIRS:
        s = pattern.sub(lambda _m, r=repl: r, s)
    return s.strip()


def repair_broken_latex(text: str) -> str:
    """Восстановить \\frac, \\text; убрать control chars и типичный мусор LLM."""
    if not text:
        return text
    out = text.replace("\f", "")
    out = re.sub(r"\\f\\frac", r"\\frac", out)

    def _disp(m: re.Match[str]) -> str:
        return f"$${_sanitize_math_inner(m.group(1))}$$"

    def _inline(m: re.Match[str]) -> str:
        return f"${_sanitize_math_inner(m.group(1))}$"

    out = _MATH_DISPLAY_RE.sub(_disp, out)
    out = _MATH_INLINE_RE.sub(_inline, out)
    return out


def extract_doi_from_text(text: str) -> str | None:
    m = _DOI_EXTRACT_RE.search(text or "")
    if not m:
        return None
    return m.group(1).rstrip(".,);]")


def extract_arxiv_id_from_text(text: str) -> str | None:
    m = _ARXIV_URL_RE.search(text or "")
    return m.group(1) if m else None


def doi_link_html(doi: str, label: str | None = None) -> str:
    doi = doi.strip()
    if not doi:
        return ""
    lab = label or f"DOI {doi[:48]}"
    return _link(f"https://doi.org/{doi}", lab, "doi-link")


def arxiv_link_html(arxiv_id: str) -> str:
    aid = arxiv_id.strip()
    return _link(f"https://arxiv.org/abs/{aid}", f"arXiv:{aid}", "arxiv-link")


# LaTeX в Reasoner: $...$ и $$...$$ — вынести до markdown, чтобы не съесть \ и _
def _protect_math_delimiters(text: str) -> tuple[str, list[str]]:
    """Временные маркеры → markdown → восстановить сырой TeX для KaTeX в браузере."""
    slots: list[str] = []

    def _slot(raw: str) -> str:
        slots.append(raw)
        return f"KEMATH{len(slots) - 1}END"

    out = _MATH_DISPLAY_RE.sub(lambda m: _slot(f"$${m.group(1)}$$"), text)
    out = _MATH_INLINE_RE.sub(lambda m: _slot(f"${m.group(1)}$"), out)
    return out, slots


def _restore_math_delimiters(html: str, slots: list[str]) -> str:
    for i, raw in enumerate(slots):
        html = html.replace(f"KEMATH{i}END", raw)
    return html


def _link(href: str, label: str, class_name: str = "ext-link") -> str:
    safe_href = html.escape(href, quote=True)
    safe_label = html.escape(label)
    return f'<a href="{safe_href}" class="{class_name}" target="_blank" rel="noopener noreferrer">{safe_label}</a>'


def linkify_references(
    text: str,
    source_registry: list[dict[str, Any]] | None = None,
) -> str:
    """Escape HTML then add clickable DOI / arXiv / URL / [Sx] source anchors."""
    if not text:
        return ""
    out = html.escape(text)

    def doi_url_sub(m: re.Match[str]) -> str:
        doi = m.group(1).rstrip(".,)")
        return _link(f"https://doi.org/{doi}", f"doi.org/{doi}", "doi-link")

    out = _DOI_URL_RE.sub(doi_url_sub, out)

    def doi_bare_sub(m: re.Match[str]) -> str:
        doi = m.group(1).rstrip(".,)")
        return _link(f"https://doi.org/{doi}", doi, "doi-link")

    out = _DOI_BARE_RE.sub(doi_bare_sub, out)

    def arxiv_sub(m: re.Match[str]) -> str:
        aid = m.group(1)
        return _link(f"https://arxiv.org/abs/{aid}", f"arXiv:{aid}", "arxiv-link")

    out = _ARXIV_RE.sub(arxiv_sub, out)

    def url_sub(m: re.Match[str]) -> str:
        url = m.group(0).rstrip(".,)")
        return _link(url, url if len(url) < 80 else url[:77] + "…", "ext-link")

    out = _URL_RE.sub(url_sub, out)
    out = linkify_source_anchors_html(out, source_registry)
    return out


def markdown_document_html(
    text: str,
    source_registry: list[dict[str, Any]] | None = None,
) -> str:
    """Markdown (Reasoner) → HTML: заголовки, таблицы, code blocks, LaTeX ($...$)."""
    raw = repair_broken_latex((text or "").strip())
    raw = expand_source_tags_to_markdown_links(raw, source_registry)
    if not raw:
        return ""
    try:
        import markdown

        protected, math_slots = _protect_math_delimiters(raw)
        body = markdown.markdown(
            protected,
            extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        )
        body = _restore_math_delimiters(body, math_slots)
        return f'<div class="md-body prose">{body}</div>'
    except ImportError:
        return f'<div class="md-body">{paragraphs_html(raw, source_registry)}</div>'


def paragraphs_html(
    text: str,
    source_registry: list[dict[str, Any]] | None = None,
) -> str:
    linked = linkify_references(repair_broken_latex(text), source_registry)
    blocks = [p.strip() for p in linked.split("\n\n") if p.strip()]
    if not blocks:
        return f"<p>{linked}</p>"
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in blocks)
