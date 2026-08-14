"""Сопоставление FIG_n ↔ Figure/Fig./§ из текста и подписей."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from knowledge_engine.services.article_ingestion.annotated_article_ops import (
    sorted_p_ids,
)
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

_FIG_REF_RE = re.compile(
    r"(?:Figure|Fig\.?|Рис\.?)\s*(\d{1,3})(?:\s*\(([a-zA-Z])\))?"
    r"|§\s*(\d{1,3})"
    r"|\[(FIG(?:_SEQ)?_\d+)",
    re.IGNORECASE,
)
_CAPTION_LINE_RE = re.compile(
    r"^\s*(?:Fig\.?|Figure|Рис\.?)\s*(\d{1,3})",
    re.IGNORECASE,
)


@dataclass
class FigureAnchor:
    internal_id: str
    labels: list[str] = field(default_factory=list)
    caption: str = ""
    page_no: int = 0
    anchor_p_ids: list[str] = field(default_factory=list)
    document_number: int | None = None
    subpanel: str = ""


def _norm_fig_id(raw: str) -> str:
    t = (raw or "").strip().upper()
    if t.startswith("FIG_") or t.startswith("FIG_SEQ_"):
        return t
    if t.startswith("FIG"):
        return f"FIG_{t[3:].lstrip('_')}"
    return t


def _parse_document_number(fid: str) -> int | None:
    m = re.match(r"^FIG_(\d+)$", fid, re.I)
    if m:
        return int(m.group(1))
    return None


def _label_for_number(num: int, sub: str = "") -> list[str]:
    labels = [f"Figure {num}", f"Fig. {num}", f"Fig {num}"]
    if sub:
        labels.extend(
            [
                f"Figure {num}({sub})",
                f"Fig. {num}({sub})",
                f"Fig. {num}{sub}",
            ]
        )
    return labels


def build_figure_anchors(annotated: AnnotatedArticle) -> dict[str, FigureAnchor]:
    fig_ids = sorted(
        set((annotated.fig_map or {}).keys()) | set((annotated.fig_bytes or {}).keys()),
        key=lambda x: (
            0 if re.match(r"^FIG_\d+$", x, re.I) else 1,
            int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0,
            x,
        ),
    )
    anchors: dict[str, FigureAnchor] = {}
    for fid in fig_ids:
        fid = _norm_fig_id(fid)
        num = _parse_document_number(fid)
        anchors[fid] = FigureAnchor(
            internal_id=fid,
            labels=_label_for_number(num, "") if num is not None else [fid],
            document_number=num,
        )

    pmap = annotated.paragraph_map or {}
    ppage = annotated.paragraph_page or {}
    for pid in sorted_p_ids(pmap):
        text = (pmap.get(pid) or "").strip()
        if not text:
            continue
        page = int(ppage.get(pid, 0) or 0)
        for m in _FIG_REF_RE.finditer(text):
            fig_token = m.group(4)
            num_s = m.group(1) or m.group(3)
            sub = (m.group(2) or "").strip()
            target: str | None = None
            if fig_token:
                target = _norm_fig_id(fig_token.strip("[]"))
            elif num_s:
                n = int(num_s)
                target = f"FIG_{n}" if f"FIG_{n}" in anchors else None
                if target is None:
                    for k, a in anchors.items():
                        if a.document_number == n:
                            target = k
                            break
            if not target or target not in anchors:
                continue
            ent = anchors[target]
            if pid not in ent.anchor_p_ids:
                ent.anchor_p_ids.append(pid)
            if page and not ent.page_no:
                ent.page_no = page
            if _CAPTION_LINE_RE.match(text) and len(text) < 800:
                ent.caption = text[:2000]
            for lab in _label_for_number(
                int(num_s) if num_s else (ent.document_number or 0), sub
            ):
                if lab not in ent.labels:
                    ent.labels.append(lab)

    for line in (annotated.annotated_markdown or "").splitlines():
        s = line.strip()
        if not s.upper().startswith("[FIG"):
            continue
        m = re.match(r"^\[(FIG(?:_SEQ)?_\d+)", s, re.I)
        if not m:
            continue
        fid = _norm_fig_id(m.group(1))
        if fid in anchors and fid not in anchors[fid].labels:
            anchors[fid].labels.append(fid)

    return anchors


def anchors_to_json(anchors: dict[str, FigureAnchor]) -> str:
    payload = {
        k: {
            "internal_id": v.internal_id,
            "labels": v.labels,
            "caption": v.caption,
            "page_no": v.page_no,
            "anchor_p_ids": v.anchor_p_ids,
            "document_number": v.document_number,
        }
        for k, v in anchors.items()
    }
    return json.dumps(payload, ensure_ascii=False)
