"""Markdown → AnnotatedArticle ([P_n], без FIG)."""

from __future__ import annotations

import re

from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_P_TAG_RE = re.compile(r"^\[(P_\d+)\]")


def build_annotated_markdown(raw: str) -> AnnotatedArticle:
    text = (raw or "").replace("\r\n", "\n")
    lines_out: list[str] = []
    paragraph_map: dict[str, str] = {}
    p_idx = 0
    pending: list[str] = []

    def flush_paragraph() -> None:
        nonlocal p_idx
        body = " ".join(pending).strip()
        pending.clear()
        if len(body) < 2:
            return
        p_idx += 1
        pid = f"P_{p_idx}"
        paragraph_map[pid] = body[:4000]
        lines_out.append(f"[{pid}] {body}")

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if _HEADING_RE.match(stripped) or _P_TAG_RE.match(stripped):
            flush_paragraph()
        if _P_TAG_RE.match(stripped):
            m = _P_TAG_RE.match(stripped)
            body = stripped[m.end() :].strip()
            if body:
                pending.append(body)
            continue
        hm = _HEADING_RE.match(stripped)
        if hm:
            pending.append(hm.group(2).strip())
            continue
        pending.append(stripped)
    flush_paragraph()

    return AnnotatedArticle(
        annotated_markdown="\n\n".join(lines_out).strip(),
        fig_map={},
        paragraph_map=paragraph_map,
        page_url="",
    )
