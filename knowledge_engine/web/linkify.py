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

# Потерянный «\» в \times → imes10 / kimes10 (после съеденного \t)
_TIMES_EXP_RE = r"(?:\^[\d{]+|\^\{[^{}]+\})?"


_EXP_CAP_RE = r"(\^[\d{]+|\^\{[^{}]+\})?"

_UNICODE_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _normalize_unicode_math_exponents(text: str) -> str:
    """k×10⁶ → k×10^6 (потерянный ^ в потоке LLM)."""
    if not text or "⁰" not in text and "¹" not in text and "⁶" not in text:
        if not any(ch in text for ch in "²³⁴⁵⁷⁸⁹"):
            return text
    return re.sub(
        r"(\d)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: f"{m.group(1)}^{m.group(2).translate(_UNICODE_SUPERSCRIPT)}",
        text,
    )


def heal_tab_corrupted_times(text: str) -> str:
    """
    JSON/LLM часто даёт «k» + TAB + «imes 10» вместо \\times (\\t съеден как escape).
    Также «\\t\\t…×10» и «\\t+imes» без обратного слэша.
    """
    if not text:
        return text
    text = _normalize_unicode_math_exponents(text)
    if (
        "\t" not in text
        and "imes" not in text
        and "×" not in text
        and "\\t" not in text
    ):
        return text
    s = text
    s = re.sub(r"\\t\s+imes", r"\\times", s, flags=re.IGNORECASE)
    s = re.sub(
        rf"k[\t ]+imes\s*(\d+){_EXP_CAP_RE}",
        lambda m: f"$k \\times {m.group(1)}{m.group(2) or ''}$",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"[\t]+imes\s*(\d+){_EXP_CAP_RE}",
        lambda m: f"$\\times {m.group(1)}{m.group(2) or ''}$",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"k[\t\s]*×\s*(\d+){_EXP_CAP_RE}",
        lambda m: f"$k \\times {m.group(1)}{m.group(2) or ''}$",
        s,
    )
    s = re.sub(
        rf"[\t]+×\s*(\d+){_EXP_CAP_RE}",
        lambda m: f"$\\times {m.group(1)}{m.group(2) or ''}$",
        s,
    )
    return s


def heal_broken_times_markup(text: str) -> str:
    """Восстановить $\\times 10$ из уже испорченного imes10 / k imes 10 / $k  imes 10$."""
    if not text:
        return text
    s = heal_tab_corrupted_times(text)
    if "imes" not in s and "×" not in s and "\\times" not in s:
        return s

    def _wrap_k_times(m: re.Match[str]) -> str:
        exp = m.group(2) or ""
        return f"$k \\times {m.group(1)}{exp}$"

    def _wrap_times(m: re.Match[str]) -> str:
        exp = m.group(2) or ""
        return f"$\\times {m.group(1)}{exp}$"

    # Уже обёрнуто в $…$ с мусором «k  imes 10»
    s = re.sub(
        r"\$\s*k\s+imes\s*(\d+)(\^[\d{]+|\^\{[^{}]+\})?\s*\$",
        lambda m: f"$k \\times {m.group(1)}{m.group(2) or ''}$",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\$\s*imes\s*(\d+)(\^[\d{]+|\^\{[^{}]+\})?\s*\$",
        lambda m: f"$\\times {m.group(1)}{m.group(2) or ''}$",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"(\d)\s*k\s+imes\s*(\d+){_TIMES_EXP_RE}",
        lambda m: f"{m.group(1)} $\\times {m.group(2)}{m.group(3) or ''}$",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"(\d)\s*imes(\d+){_TIMES_EXP_RE}",
        lambda m: f"{m.group(1)} $\\times {m.group(2)}{m.group(3) or ''}$",
        s,
    )
    s = re.sub(rf"k\s+imes\s*(\d+){_EXP_CAP_RE}", _wrap_k_times, s, flags=re.IGNORECASE)
    s = re.sub(rf"kimes(\d+){_EXP_CAP_RE}", _wrap_k_times, s, flags=re.IGNORECASE)
    s = re.sub(
        rf"(?<![a-zA-Z\\])imes\s*(\d+){_EXP_CAP_RE}",
        _wrap_times,
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"(\$\\times \d+[^$]*\$)(?:\1)+", r"\1", s)
    return s


def _heal_corrupted_times_in_math_inner(s: str) -> str:
    """Внутри $…$: «k  imes 10» → \\times 10 (после съеденного \\t)."""
    if not s or "imes" not in s:
        return s
    out = s
    out = re.sub(r"k[\t ]+imes\s*", r"\\times ", out, flags=re.IGNORECASE)
    out = re.sub(r"[\t]+imes\s*", r"\\times ", out, flags=re.IGNORECASE)
    out = re.sub(r"k[\t\s]*×\s*", r"\\times ", out, flags=re.IGNORECASE)
    out = re.sub(r"[\t]+×\s*", r"\\times ", out, flags=re.IGNORECASE)
    out = re.sub(r"k\s+imes\s*", r"\\times ", out, flags=re.IGNORECASE)
    out = re.sub(
        r"(?<![a-zA-Z\\])imes\s*(\d+)",
        r"\\times \1",
        out,
        flags=re.IGNORECASE,
    )
    return out


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
    s = re.sub(r"\\t\s+(?!imes)", " ", s)
    return s


def _heal_broken_frac_inner(s: str) -> str:
    """
    Восстановить \\frac{…}{…} из типичного мусора LLM:
    \\frac100 \\cdot 10^6 8 \\approx → \\frac{100 \\cdot 10^6}{8} \\approx
    """
    if not s or ("frac" not in s.lower() and "rac{" not in s):
        return s
    out = s.replace("·", "\\cdot ").replace("⋅", "\\cdot ")
    out = re.sub(r"(?<![a-zA-Z\\])frac", r"\\frac", out, flags=re.IGNORECASE)

    def _mk_frac(num: str, den: str, tail: str = "") -> str:
        return f"\\frac{{{num.strip()}}}{{{den.strip()}}}{tail}"

    # frac100⋅1068 — слипшиеся 10^6 и знаменатель 8 без ^ и пробелов
    def _glued_pow_den(m: re.Match[str]) -> str:
        return _mk_frac(f"{m.group(1)} \\cdot 10^{m.group(2)}", m.group(3))

    out = re.sub(
        r"(?<![a-zA-Z\\])frac(\d+)[·⋅]\s*10(\d)(\d)(?![0-9])",
        _glued_pow_den,
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\\frac\{?(\d+)\}?\\cdot\s*10(\d)(\d)(?![0-9])",
        _glued_pow_den,
        out,
        flags=re.IGNORECASE,
    )

    out = re.sub(
        r"\\frac\{?(\d+)\}?\s*"
        r"(?:\\cdot\s*)?"
        r"(\d+(?:\^[\d{]+|\^\{[^{}]+\})?)"
        r"\s+(\d{1,4})"
        r"(\s*(?:\\approx|≈|=(?!=))\s*)?",
        lambda m: _mk_frac(
            f"{m.group(1)} \\cdot {m.group(2)}",
            m.group(3),
            m.group(4) or " ",
        ),
        out,
    )
    out = re.sub(
        r"\\frac\{([^}]+)\}\s+(\d{1,4})\s*(\\approx|≈)?",
        lambda m: _mk_frac(
            m.group(1),
            m.group(2),
            f" {m.group(3)}" if m.group(3) else " ",
        ),
        out,
    )
    out = re.sub(
        r"\\frac(\d+)\s+(\d{1,4})\s*(\\approx|≈)",
        lambda m: _mk_frac(m.group(1), m.group(2), f" {m.group(3)}"),
        out,
    )
    return out


def _sanitize_math_inner(inner: str) -> str:
    """Очистка TeX внутри $...$ / $$...$$ перед KaTeX."""
    s = inner
    s = _heal_corrupted_times_in_math_inner(s)
    s = _heal_broken_frac_inner(s)
    s = _fix_corrupted_text_commands(s)
    s = _LATEX_CONTROL_RE.sub("", s)
    s = s.replace("\f", "")
    s = re.sub(r"\\f\\frac", r"\\frac", s)
    s = re.sub(r"(?<!\\)f\\frac", r"\\frac", s)
    s = re.sub(r"\f(?=rac\{)", "", s)
    s = re.sub(r"\\t+\\text", r"\\text", s)
    s = re.sub(r"\t+", " ", s)
    s = re.sub(r"(.{6,}?)\s+\1(?=[\)\}\]\s,;]|$)", r"\1", s)
    for pattern, repl in _LATEX_BROKEN_REPAIRS:
        s = pattern.sub(lambda _m, r=repl: r, s)
    return s.strip()


_MATH_CHUNK_RE = re.compile(
    r"\$\$[\s\S]+?\$\$|(?<!\$)\$[^$\n]+?\$(?!\$)",
)

_BARE_FRAC_GLUE_RE = re.compile(
    r"(?<![a-zA-Z\\])frac(\d+)[·⋅]\s*10(\d)(\d)(?![0-9\w])",
    re.IGNORECASE,
)

_BARE_FRAC_SPAN_RE = re.compile(
    r"(?:\\frac|(?<![a-zA-Z])frac)"
    r"(?:\{[^}]+\}|\d+)?"
    r"(?:\s*(?:\\cdot|·|⋅)\s*)?"
    r"[^\n$]{0,120}?"
    r"(?:\\approx|≈)\s*"
    r"[\d.,]+"
    r"(?:\s*(?:\\text\s*\{[^{}]+\}|\\text\{[^{}]+\}|"
    r"[А-Яа-яёA-Za-z][А-Яа-яёA-Za-z %]{0,30})?)?",
    re.IGNORECASE,
)

_BARE_K_TIMES_RE = re.compile(
    rf"(?<![$\w/])(k[\t ]*\\times\s*\d+{_EXP_CAP_RE})(?![$\w])",
    re.IGNORECASE,
)
_BARE_TIMES_RE = re.compile(
    rf"(?<![$\w/])(?<![kK]\s)(\\times\s*\d+{_EXP_CAP_RE})(?![$\w])",
    re.IGNORECASE,
)


def _apply_outside_math_chunks(text: str, transform) -> str:
    if not text:
        return text
    parts: list[str] = []
    last = 0
    for m in _MATH_CHUNK_RE.finditer(text):
        if m.start() > last:
            parts.append(transform(text[last : m.start()]))
        parts.append(m.group(0))
        last = m.end()
    if last < len(text):
        parts.append(transform(text[last:]))
    return "".join(parts) if parts else transform(text)


def _wrap_bare_latex_outside_math(text: str) -> str:
    """LLM часто пишет \\frac… и k \\times без $…$ — KaTeX их не видит."""
    if not text:
        return text

    def _plain(chunk: str) -> str:
        if not chunk:
            return chunk
        out = chunk
        if "frac" in out.lower():

            def _frac_sub(m: re.Match[str]) -> str:
                return f"${_sanitize_math_inner(m.group(0))}$"

            out = _BARE_FRAC_GLUE_RE.sub(_frac_sub, out)
            out = _BARE_FRAC_SPAN_RE.sub(_frac_sub, out)
        if "\\times" in out or "imes" in out.lower() or "×" in out:
            out = _BARE_K_TIMES_RE.sub(
                lambda m: f"${_sanitize_math_inner(m.group(1))}$",
                out,
            )
            out = _BARE_TIMES_RE.sub(
                lambda m: f"${_sanitize_math_inner(m.group(1))}$",
                out,
            )
        return out

    return _apply_outside_math_chunks(text, _plain)


def repair_broken_latex(text: str) -> str:
    """Восстановить \\frac, \\text; убрать control chars и типичный мусор LLM."""
    if not text:
        return text
    out = heal_tab_corrupted_times(text)
    out = heal_broken_times_markup(out)
    out = out.replace("\f", "")
    out = re.sub(r"\\f\\frac", r"\\frac", out)
    out = _wrap_bare_latex_outside_math(out)

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


_MATH_PAREN_INLINE_RE = re.compile(r"\\\((.+?)\\\)")


# LaTeX в Reasoner: $...$ и $$...$$ — вынести до markdown, чтобы не съесть \ и _
def _protect_math_delimiters(text: str) -> tuple[str, list[str]]:
    """Временные маркеры → markdown → восстановить сырой TeX для KaTeX в браузере."""
    slots: list[str] = []

    def _slot(raw: str) -> str:
        slots.append(raw)
        return f"KEMATH{len(slots) - 1}END"

    out = _MATH_DISPLAY_RE.sub(
        lambda m: _slot(f"$${_sanitize_math_inner(m.group(1))}$$"),
        text,
    )
    out = _MATH_INLINE_RE.sub(
        lambda m: _slot(f"\\({_sanitize_math_inner(m.group(1))}\\)"),
        out,
    )
    out = _MATH_PAREN_INLINE_RE.sub(
        lambda m: _slot(f"\\({_sanitize_math_inner(m.group(1))}\\)"),
        out,
    )
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


def _wrap_html_tables(html: str) -> str:
    """Горизонтальный скролл таблицы внутри блока, не всего чата."""
    if "<table" not in html.lower():
        return html
    out: list[str] = []
    i = 0
    low = html.lower()
    while True:
        start = low.find("<table", i)
        if start < 0:
            out.append(html[i:])
            break
        end_tag = low.find("</table>", start)
        if end_tag < 0:
            out.append(html[i:])
            break
        end = end_tag + len("</table>")
        chunk = html[start:end]
        out.append(html[i:start])
        if "md-table-scroll" in html[max(0, start - 80) : start]:
            out.append(chunk)
        else:
            out.append(f'<div class="md-table-scroll">{chunk}</div>')
        i = end
    return "".join(out)


_MATERIAL_TAG_RE = re.compile(
    r"\[(?:diagram|схема|code|код|card|карточка):((?:diagram|code|card)-\d+)\]",
    re.IGNORECASE,
)

_MATERIAL_LABELS = {
    "diagram": "схема",
    "code": "код",
    "card": "карточка",
}


def linkify_material_anchor_tags_md(text: str) -> str:
    """[diagram:diagram-1] → кликабельный якорь в markdown."""

    def repl(m: re.Match[str]) -> str:
        aid = m.group(1).lower()
        prefix = aid.split("-", 1)[0]
        label = _MATERIAL_LABELS.get(prefix, prefix)
        return f"[{label} {aid}](#ke-material-{aid})"

    return _MATERIAL_TAG_RE.sub(repl, text or "")


def linkify_material_anchors_html(html: str) -> str:
    """Добавить class/data для клика → панель материалов."""
    if "ke-material-" not in (html or "").lower():
        return html or ""

    def repl(m: re.Match[str]) -> str:
        aid = m.group(1).lower()
        return (
            f'href="#ke-material-{aid}" class="ke-material-anchor" '
            f'data-material-id="{aid}"'
        )

    out = re.sub(
        r'href="#ke-material-((?:diagram|code|card)-\d+)"',
        repl,
        html,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r'class="ke-material-anchor"\s+data-material-id="([^"]+)"\s+'
        r'class="ke-material-anchor"',
        r'class="ke-material-anchor" data-material-id="\1"',
        out,
    )


_HYPHEN_ONLY_P_RE = re.compile(r"<p>\s*-\s*</p>", re.IGNORECASE)
_P_TAG_CHUNK_RE = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)
# [)] вместо \) — вложенные «((?:…\))» ломают парсер re (Python 3.14).
_LETTER_SUB_LABEL = r"(?:\*\*[а-яёa-z][)]\*\*|[а-яёa-z][)])"
_LETTER_SUB_LABEL_WS = r"(?:\*\*[а-яёa-z][)]\*\*|[а-яёa-z][)]\s)"
_WRONG_NUMBERED_LETTER_LINE_RE = re.compile(
    rf"^\d{{1,2}}\.\s*({_LETTER_SUB_LABEL})\s*(.*)$",
    re.IGNORECASE,
)
_ORDERED_LIST_LINE_RE = re.compile(
    rf"^\d{{1,2}}\.\s+(?!{_LETTER_SUB_LABEL})",
    re.IGNORECASE,
)
_SUB_LINE_START_RE = re.compile(
    rf"^({_LETTER_SUB_LABEL_WS})",
    re.IGNORECASE,
)
_MATH_OPERATOR_TAIL_RE = re.compile(r"[\+\*\/\=,]\s*$")

_GLUE_ORDERED_IN_P_RE = re.compile(
    r"<p>(?P<body>(?:(?!</p>).)*\d{1,2}\.\s+(?:(?!</p>).)+)</p>",
    re.IGNORECASE | re.DOTALL,
)
# Letter sub-items only after a real line break — never mid-sentence «(a + b)».
_GLUE_SUB_IN_P_RE = re.compile(
    rf"<p>(?P<body>(?:(?!<\/p>).)*(?:\n|<br\s*/?>)\s*(?:{_LETTER_SUB_LABEL_WS})(?:(?!<\/p>).)+)</p>",
    re.IGNORECASE | re.DOTALL,
)


def letter_marker_is_paren_continuation(prefix: str) -> bool:
    """True when a following «b)» closes math/parens, not a list marker.

    CommonMark lists need a newline before the marker. A wrap inside
    ``(например, сложение a + b)`` is not a list.
    """
    prev = (prefix or "").rstrip()
    if not prev:
        return False
    if prev.count("(") > prev.count(")"):
        return True
    if prev.endswith(("(", "[", "{")):
        return True
    return bool(_MATH_OPERATOR_TAIL_RE.search(prev))


def _split_glued_ordered_paragraph(body: str) -> str | None:
    parts = re.split(r"(?<=[.!?…])\s+(?=\d{1,2}\.\s+)", body)
    if len(parts) < 2:
        parts = re.split(
            r"\s+(?=\d{1,2}\.\s+(?!(?:\*\*)?[а-яёa-z][)]))",
            body,
            flags=re.IGNORECASE,
        )
    if len(parts) < 2:
        return None
    items: list[str] = []
    sub_items: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        wrong = _WRONG_NUMBERED_LETTER_LINE_RE.match(chunk)
        if wrong:
            label = wrong.group(1).strip()
            tail = (wrong.group(2) or "").strip()
            sub_items.append(f"<li>{label} {tail}</li>".strip())
            continue
        chunk = re.sub(r"^\d{1,2}\.\s+", "", chunk)
        items.append(f"<li>{chunk}</li>")
    if len(items) < 2 and not sub_items:
        return None
    ol = f"<ol>{''.join(items)}</ol>" if len(items) >= 2 else ""
    ul = f"<ul>{''.join(sub_items)}</ul>" if sub_items else ""
    return f"{ol}{ul}" if (ol or ul) else None


def _split_glued_subitems_paragraph(body: str) -> str | None:
    text = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    parts = re.split(
        rf"\n\s*(?={_LETTER_SUB_LABEL_WS})",
        text,
        flags=re.IGNORECASE,
    )
    if len(parts) < 2:
        return None
    merged = [parts[0]]
    for part in parts[1:]:
        if letter_marker_is_paren_continuation(merged[-1]):
            merged[-1] = f"{merged[-1].rstrip()} {part.lstrip()}"
        else:
            merged.append(part)
    items = [p.strip() for p in merged if p.strip()]
    if len(items) < 2:
        return None
    return f"<ul>{''.join(f'<li>{chunk}</li>' for chunk in items)}</ul>"


_WRONG_OL_LETTER_ITEM_RE = re.compile(
    rf"<li>\s*\d{{1,2}}\.\s*({_LETTER_SUB_LABEL})\s*(.*?)</li>",
    re.IGNORECASE | re.DOTALL,
)


def _demote_wrong_numbered_letter_ol(html: str) -> str:
    """«3. а)» в <ol> → <ul> с «а)» (без ложной нумерации 3–5)."""

    def repl_ol(m: re.Match[str]) -> str:
        inner = m.group(1)
        if not _WRONG_OL_LETTER_ITEM_RE.search(inner):
            return m.group(0)
        fixed = _WRONG_OL_LETTER_ITEM_RE.sub(
            lambda mi: f"<li>{mi.group(1).strip()} {mi.group(2).strip()}</li>",
            inner,
        )
        if re.search(
            r"<li>\s*\d{1,2}\.\s+(?!(?:\*\*)?[а-яёa-z][)])",
            fixed,
            re.IGNORECASE,
        ):
            return f"<ol>{fixed}</ol>"
        return f"<ul>{fixed}</ul>"

    return re.sub(r"<ol>(.*?)</ol>", repl_ol, html, flags=re.IGNORECASE | re.DOTALL)


def _strip_paragraph_inner_html(inner: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", inner, flags=re.IGNORECASE)
    return re.sub(r"<[^>]+>", "", s).strip()


def _list_html_from_paragraph_texts(texts: list[str]) -> str | None:
    segments: list[tuple[str, list[str]]] = []
    cur_kind: str | None = None
    cur_items: list[str] = []

    def flush() -> None:
        nonlocal cur_kind, cur_items
        if cur_items:
            segments.append((cur_kind or "ol", cur_items))
        cur_kind = None
        cur_items = []

    prev_text = ""
    for raw in texts:
        text = raw.strip()
        if not text:
            return None
        if _SUB_LINE_START_RE.match(text) and letter_marker_is_paren_continuation(
            prev_text
        ):
            return None
        wrong = _WRONG_NUMBERED_LETTER_LINE_RE.match(text)
        if wrong:
            label = wrong.group(1).strip()
            tail = (wrong.group(2) or "").strip()
            li = f"<li>{label} {tail}</li>".strip()
            if cur_kind and cur_kind != "ul":
                flush()
            cur_kind = "ul"
            cur_items.append(li)
            prev_text = text
            continue
        if _ORDERED_LIST_LINE_RE.match(text):
            body = re.sub(r"^\d{1,2}\.\s+", "", text)
            if cur_kind and cur_kind != "ol":
                flush()
            cur_kind = "ol"
            cur_items.append(f"<li>{body}</li>")
            prev_text = text
            continue
        bullet = re.match(r"^-\s+(.*)$", text, re.DOTALL)
        if bullet:
            if cur_kind and cur_kind != "ul":
                flush()
            cur_kind = "ul"
            cur_items.append(f"<li>{bullet.group(1).strip()}</li>")
            prev_text = text
            continue
        if _SUB_LINE_START_RE.match(text):
            if cur_kind and cur_kind != "ul":
                flush()
            cur_kind = "ul"
            cur_items.append(f"<li>{text}</li>")
            prev_text = text
            continue
        return None
    flush()
    if not segments:
        return None
    parts: list[str] = []
    for kind, items in segments:
        if not items:
            continue
        tag = "ol" if kind == "ol" else "ul"
        parts.append(f"<{tag}>{''.join(items)}</{tag}>")
    return "".join(parts) if parts else None


def _merge_adjacent_paragraph_lists(html: str) -> str:
    """Соседние <p>1. …</p><p>2. …</p> → один <ol>; <p>- а)</p> → <ul>."""
    if not html or "<p>" not in html:
        return html or ""
    html = _HYPHEN_ONLY_P_RE.sub("", html)
    chunks: list[tuple[str, str]] = []
    pos = 0
    while pos < len(html):
        ws_m = re.match(r"\s*", html[pos:])
        if ws_m and ws_m.end():
            chunks.append(("raw", ws_m.group(0)))
            pos += ws_m.end()
            if pos >= len(html):
                break
        p_m = re.match(r"<p>(.*?)</p>", html[pos:], re.IGNORECASE | re.DOTALL)
        if p_m:
            chunks.append(("p", _strip_paragraph_inner_html(p_m.group(1))))
            pos += p_m.end()
            continue
        chunks.append(("raw", html[pos:]))
        break
    out: list[str] = []
    i = 0
    while i < len(chunks):
        kind, val = chunks[i]
        if kind != "p":
            out.append(val)
            i += 1
            continue
        run: list[str] = []
        j = i
        while j < len(chunks) and chunks[j][0] == "p":
            run.append(chunks[j][1])
            j += 1
        merged = _list_html_from_paragraph_texts(run)
        if merged:
            out.append(merged)
        else:
            for line in run:
                out.append(f"<p>{line}</p>")
        i = j
    return "".join(out)


def postprocess_html_glued_lists(html: str) -> str:
    """Если markdown оставил один <p> со «1. … 2. …» — разбить на ol/ul."""
    if not html or "<p>" not in html:
        return html or ""
    html = _merge_adjacent_paragraph_lists(html)

    def ordered_sub(m: re.Match[str]) -> str:
        repl = _split_glued_ordered_paragraph(m.group("body"))
        return repl if repl else m.group(0)

    out = _GLUE_ORDERED_IN_P_RE.sub(ordered_sub, html)

    def subitem_sub(m: re.Match[str]) -> str:
        repl = _split_glued_subitems_paragraph(m.group("body"))
        return repl if repl else m.group(0)

    out = _GLUE_SUB_IN_P_RE.sub(subitem_sub, out)
    return _demote_wrong_numbered_letter_ol(out)


def markdown_document_html(
    text: str,
    source_registry: list[dict[str, Any]] | None = None,
) -> str:
    """Markdown (Reasoner) → HTML: заголовки, таблицы, code blocks, LaTeX ($...$)."""
    raw = repair_broken_latex((text or "").strip())
    raw = linkify_material_anchor_tags_md(raw)
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
        body = _wrap_html_tables(body)
        body = linkify_material_anchors_html(body)
        body = postprocess_html_glued_lists(body)
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
