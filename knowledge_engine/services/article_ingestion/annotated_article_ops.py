"""Операции над AnnotatedArticle: порядок [P_n]/[FIG_m], подрезка."""

from __future__ import annotations

import re

from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

_P_LINE_RE = re.compile(r"^\[(P_\d+)\]", re.I)
_FIG_LINE_RE = re.compile(r"^\[(FIG_\d+)", re.I)


def sorted_p_ids(paragraph_map: dict[str, str]) -> list[str]:
    ids = [k for k in paragraph_map if re.match(r"^P_\d+$", k, re.I)]
    return sorted(ids, key=lambda x: int(x.split("_", 1)[1]))


def p_index_map(paragraph_map: dict[str, str]) -> dict[str, int]:
    ordered = sorted_p_ids(paragraph_map)
    return {pid: i for i, pid in enumerate(ordered)}


def parse_annotated_blocks(annotated_markdown: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for line in (annotated_markdown or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _P_LINE_RE.match(line)
        if m:
            blocks.append(("P", _norm_p(m.group(1))))
            continue
        m = _FIG_LINE_RE.match(line)
        if m:
            blocks.append(("FIG", _norm_fig(m.group(1))))
    return blocks


def _norm_p(pid: str) -> str:
    t = (pid or "").strip().upper()
    if t.startswith("P_"):
        return t
    return f"P_{t.lstrip('P_')}"


def _norm_fig(fid: str) -> str:
    t = (fid or "").strip().upper()
    if t.startswith("FIG_"):
        return t
    return f"FIG_{t.lstrip('FIG_')}"


def fig_anchor_p_id(
    blocks: list[tuple[str, str]],
    fig_id: str,
) -> str | None:
    target = _norm_fig(fig_id)
    last_p: str | None = None
    for kind, ident in blocks:
        if kind == "P":
            last_p = _norm_p(ident)
        elif kind == "FIG" and _norm_fig(ident) == target:
            return last_p
    return last_p


def kept_p_id_set(
    paragraph_map: dict[str, str],
    keep_ranges: list[tuple[str, str]],
) -> set[str]:
    order = sorted_p_ids(paragraph_map)
    idx = p_index_map(paragraph_map)
    kept: set[str] = set()
    for start, end in keep_ranges:
        s = _norm_p(start)
        e = _norm_p(end)
        if s not in idx:
            continue
        si = idx[s]
        ei = idx.get(e, si)
        if e in idx and ei < si:
            ei = si
        for pid in order[si : ei + 1]:
            kept.add(pid)
    if not kept and order:
        kept = set(order)
    return kept


def _fig_line_for_id(annotated_markdown: str, fid: str) -> str | None:
    norm = _norm_fig(fid)
    suffix = norm.split("_", 1)[-1]
    for line in (annotated_markdown or "").split("\n"):
        s = line.strip()
        if s.upper().startswith(f"[{norm}") or s.upper().startswith(f"[FIG_{suffix}"):
            return s
    return None


def prune_annotated_article(
    annotated: AnnotatedArticle,
    kept_p_ids: set[str],
) -> AnnotatedArticle:
    kept_p = {_norm_p(p) for p in kept_p_ids}
    blocks = parse_annotated_blocks(annotated.annotated_markdown)
    new_lines: list[str] = []
    new_pmap: dict[str, str] = {}
    new_fig_map: dict[str, str] = {}
    new_fig_bytes: dict[str, tuple[bytes, str]] = {}
    new_fig_source: dict[str, str] = {}
    new_fig_topology: dict[str, dict] = {}
    new_pages: dict[str, int] = {}
    p_idx = 0

    for kind, ident in blocks:
        if kind == "P":
            pid = _norm_p(ident)
            if pid not in kept_p:
                continue
            text = annotated.paragraph_map.get(pid) or annotated.paragraph_map.get(
                ident, ""
            )
            p_idx += 1
            new_pid = f"P_{p_idx}"
            new_pmap[new_pid] = text
            old_page = (annotated.paragraph_page or {}).get(pid)
            if old_page is not None:
                new_pages[new_pid] = old_page
            new_lines.append(f"[{new_pid}] {text}")
        elif kind == "FIG":
            fid = _norm_fig(ident)
            anchor = fig_anchor_p_id(blocks, fid)
            if anchor and _norm_p(anchor) not in kept_p:
                continue
            if anchor is None:
                continue
            line = _fig_line_for_id(annotated.annotated_markdown, fid)
            if line:
                new_lines.append(line)
            if fid in annotated.fig_map:
                new_fig_map[fid] = annotated.fig_map[fid]
            if fid in annotated.fig_bytes:
                new_fig_bytes[fid] = annotated.fig_bytes[fid]
            if fid in annotated.fig_extract_source:
                new_fig_source[fid] = annotated.fig_extract_source[fid]
            if fid in (annotated.fig_extract_topology or {}):
                new_fig_topology[fid] = annotated.fig_extract_topology[fid]

    return AnnotatedArticle(
        annotated_markdown="\n\n".join(new_lines).strip(),
        fig_map=new_fig_map,
        paragraph_map=new_pmap,
        page_url=annotated.page_url,
        fig_bytes=new_fig_bytes,
        fig_extract_source=new_fig_source,
        fig_extract_topology=new_fig_topology,
        paragraph_page=new_pages,
    )


def _fig_sort_key(fid: str) -> int:
    t = _norm_fig(fid)
    try:
        return int(t.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def restore_figures_after_text_prune(
    before: AnnotatedArticle,
    after: AnnotatedArticle,
) -> AnnotatedArticle:
    """Triage shrinks [P_n] only; re-attach FIG assets for spatial/VLM MAP."""
    if not before.fig_map:
        return after
    missing = [
        fid
        for fid in before.fig_map
        if _norm_fig(fid) not in {_norm_fig(k) for k in after.fig_map}
    ]
    if not missing:
        return after

    new_fig_map = dict(after.fig_map)
    new_fig_bytes = dict(after.fig_bytes)
    new_fig_source = dict(after.fig_extract_source or {})
    new_fig_topology = dict(after.fig_extract_topology or {})
    extra_lines: list[str] = []
    for fid in sorted(before.fig_map.keys(), key=_fig_sort_key):
        norm = _norm_fig(fid)
        if norm in new_fig_map:
            continue
        url = before.fig_map.get(norm) or before.fig_map.get(fid, "")
        if url:
            new_fig_map[norm] = url
        if norm in before.fig_bytes:
            new_fig_bytes[norm] = before.fig_bytes[norm]
        elif fid in before.fig_bytes:
            new_fig_bytes[norm] = before.fig_bytes[fid]
        if norm in before.fig_extract_source:
            new_fig_source[norm] = before.fig_extract_source[norm]
        elif fid in before.fig_extract_source:
            new_fig_source[norm] = before.fig_extract_source[fid]
        if norm in (before.fig_extract_topology or {}):
            new_fig_topology[norm] = before.fig_extract_topology[norm]
        elif fid in (before.fig_extract_topology or {}):
            new_fig_topology[norm] = before.fig_extract_topology[fid]
        line = _fig_line_for_id(before.annotated_markdown, norm)
        if line:
            extra_lines.append(line)

    md = (after.annotated_markdown or "").strip()
    if extra_lines:
        md = (md + "\n\n" + "\n\n".join(extra_lines)).strip()

    return AnnotatedArticle(
        annotated_markdown=md,
        fig_map=new_fig_map,
        paragraph_map=after.paragraph_map,
        page_url=after.page_url,
        fig_bytes=new_fig_bytes,
        fig_extract_source=new_fig_source,
        fig_extract_topology=new_fig_topology,
        paragraph_page=after.paragraph_page,
    )
