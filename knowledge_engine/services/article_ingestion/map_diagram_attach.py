"""[ATTACHED_DIAGRAMS] для MAP-промптов из FigureRegistry."""

from __future__ import annotations

import re

from knowledge_engine.services.article_ingestion.figure_registry_service import (
    FigureRegistry,
)

_FIG_REF_RE = re.compile(
    r"\[(FIG(?:_SEQ)?_\d+)(?::|\])"
    r"|(?:Figure|Fig\.?|Рис\.?)\s*(\d{1,3})(?:\s*\(([a-zA-Z])\))?",
    re.IGNORECASE,
)


def collect_figure_ids_in_text(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in _FIG_REF_RE.finditer(text or ""):
        if m.group(1):
            fid = m.group(1).upper()
            if not fid.startswith("FIG_") and fid.startswith("FIG"):
                fid = f"FIG_{fid[3:].lstrip('_')}"
        else:
            n = int(m.group(2))
            fid = f"FIG_{n}"
        if fid not in seen:
            seen.add(fid)
            found.append(fid)
    return found


def merge_figure_ids_for_chunk(
    chunk_text: str,
    extra_figure_ids: list[str] | None = None,
) -> list[str]:
    """Ссылки из текста + [FIG_*] маркеры из нарезки окна."""
    ids = collect_figure_ids_in_text(chunk_text)
    seen = set(ids)
    for raw in extra_figure_ids or []:
        fid = (raw or "").strip().upper()
        if not fid:
            continue
        if not fid.startswith("FIG"):
            fid = f"FIG_{fid}"
        if fid not in seen:
            seen.add(fid)
            ids.append(fid)
    return ids


def build_attached_diagrams_block(
    chunk_text: str,
    registry: FigureRegistry | None,
    *,
    extra_figure_ids: list[str] | None = None,
) -> str:
    if registry is None or not registry.entries:
        return ""
    ids = merge_figure_ids_for_chunk(chunk_text, extra_figure_ids)
    if not ids:
        return ""
    parts: list[str] = ["[ATTACHED_DIAGRAMS]"]
    for fid in ids:
        ent = registry.get(fid)
        if ent is None:
            continue
        labels = ", ".join(ent.labels[:6]) if ent.labels else fid
        header = f"### {fid} ({labels})"
        body = (ent.vlm_summary or "").strip()
        if not body and ent.caption:
            body = f"Caption: {ent.caption}"
        if ent.mermaid_code.strip():
            body = (
                body + "\n\nMermaid:\n```mermaid\n" + ent.mermaid_code.strip() + "\n```"
            )
        if not body:
            body = "(VLM description pending or figure skipped)"
        parts.append(f"{header}\n{body[:6000]}")
    if len(parts) <= 1:
        return ""
    return "\n\n".join(parts)
