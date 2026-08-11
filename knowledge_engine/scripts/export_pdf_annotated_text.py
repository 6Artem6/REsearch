"""QA: PDF → текст как в spatial Map-Reduce (triage + token windows) + сырой annotate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from knowledge_engine.config import (
    BLOG_SPATIAL_MAP_MAX_TOKENS,
    BLOG_SPATIAL_OVERLAP_TOKENS,
    BLOG_SPATIAL_TRIAGE_ENABLED,
)

_MARGIN_TOP_FRAC = 0.085
_MARGIN_BOTTOM_FRAC = 0.085
_HEADER_FOOTER_RE = re.compile(
    r"(?i)(aspLOS|proceedings|acm|isbn|doi\.org|copyright|©|"
    r"permission\s+to\s+make|^\s*\d{1,3}\s*$|"
    r"^\s*\d+\s+of\s+\d+\s*$)",
)
_FIG_LINE_RE = re.compile(r"^\[(FIG(?:_SEQ)?_\d+)(?::|\])")


def _fig_sort_key(key: str) -> tuple[int, str]:
    m = re.match(r"^FIG_(\d+)$", key, re.I)
    if m:
        return (0, f"{int(m.group(1)):06d}")
    m = re.match(r"^FIG_SEQ_(\d+)$", key, re.I)
    if m:
        return (1, f"{int(m.group(1)):06d}")
    return (2, key)


def _margin_zone(page_height: float, y0: float, y1: float) -> str | None:
    top = page_height * _MARGIN_TOP_FRAC
    bottom = page_height * (1.0 - _MARGIN_BOTTOM_FRAC)
    if y1 <= top + 2:
        return "top_margin"
    if y0 >= bottom - 2:
        return "bottom_margin"
    return None


def _audit_raw_blocks(pdf_bytes: bytes) -> list[dict]:
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rows: list[dict] = []
    try:
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            ph = page.rect.height
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                bbox = block.get("bbox") or (0, 0, 0, 0)
                y0, y1 = float(bbox[1]), float(bbox[3])
                parts: list[str] = []
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(str(s.get("text", "")) for s in spans).strip()
                    if line_text:
                        parts.append(line_text)
                text = " ".join(parts).strip()
                if len(text) < 2:
                    continue
                zone = _margin_zone(ph, y0, y1)
                boiler = bool(_HEADER_FOOTER_RE.search(text)) or (
                    zone is not None and len(text) < 140
                )
                rows.append(
                    {
                        "page": page_no,
                        "zone": zone,
                        "likely_boilerplate": boiler,
                        "y0": round(y0, 1),
                        "y1": round(y1, 1),
                        "preview": text[:160].replace("\n", " "),
                    }
                )
    finally:
        doc.close()
    return rows


def _paragraphs_in_margin(
    paragraph_map: dict[str, str],
    paragraph_page: dict[str, int],
    audit: list[dict],
) -> list[tuple[str, str, str]]:
    """P_id, page, preview — если текст похож на блок из margin audit."""
    by_page: dict[int, list[dict]] = {}
    for row in audit:
        if row.get("zone") and row.get("likely_boilerplate"):
            by_page.setdefault(int(row["page"]), []).append(row)
    hits: list[tuple[str, str, str]] = []
    for pid, text in paragraph_map.items():
        page = paragraph_page.get(pid, 0)
        preview = text[:120].replace("\n", " ")
        for row in by_page.get(page, []):
            if preview[:40] in row["preview"] or row["preview"][:40] in preview:
                hits.append((pid, str(page), preview))
                break
    return hits


def _guess_title(annotated) -> str:
    for pid in sorted(
        (annotated.paragraph_map or {}).keys(),
        key=lambda x: int(x.split("_")[1]),
    ):
        t = (annotated.paragraph_map.get(pid) or "").strip()
        if 12 <= len(t) <= 200 and not t.lower().startswith("abstract"):
            return t[:300]
    return ""


def _export_llm_map_windows(
    out_dir: Path,
    *,
    annotated_after_triage,
    page_url: str,
    title: str,
    figure_registry=None,
) -> list[dict]:
    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        _figure_ids,
    )
    from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
        _MAP_SYSTEM,
        MapReduceArticleJob,
        _prompt_for_window,
    )
    from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
        estimate_text_tokens,
        split_annotated_text_by_tokens,
    )

    body = (annotated_after_triage.annotated_markdown or "").strip()
    fig_ids = _figure_ids(annotated_after_triage)
    job_title = (title or page_url).strip()[:300]
    windows = split_annotated_text_by_tokens(
        body,
        title=job_title,
        all_figure_ids=fig_ids,
        figure_registry=figure_registry,
    )
    if not windows and body:
        from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
            TokenWindowChunk,
        )

        windows = [TokenWindowChunk(window_index=0, body=body)]

    map_dir = out_dir / "llm_map_windows"
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "00_system_prompt.txt").write_text(
        _MAP_SYSTEM.strip() + "\n", encoding="utf-8"
    )

    job = MapReduceArticleJob(
        job_id=page_url,
        title=job_title,
        url=page_url,
        windows=windows,
        all_figure_ids=fig_ids,
        figure_registry=figure_registry,
    )

    manifest: list[dict] = []
    combined_parts: list[str] = [
        "# MAP phase payloads (user message per window)\n",
        f"# title={job_title!r} url={page_url}\n",
        f"# triage_enabled={BLOG_SPATIAL_TRIAGE_ENABLED} "
        f"max_tokens={BLOG_SPATIAL_MAP_MAX_TOKENS} "
        f"overlap={BLOG_SPATIAL_OVERLAP_TOKENS}\n",
        f"# figure_ids ({len(fig_ids)}): {', '.join(fig_ids[:30])}"
        + (" …" if len(fig_ids) > 30 else "")
        + "\n",
    ]

    for w in windows:
        user_prompt = _prompt_for_window(job, w)
        stem = f"window_{w.window_index:03d}"
        (map_dir / f"{stem}_user.txt").write_text(user_prompt, encoding="utf-8")
        (map_dir / f"{stem}_body.txt").write_text(w.body, encoding="utf-8")
        est = estimate_text_tokens(f"{_MAP_SYSTEM}\n{user_prompt}")
        manifest.append(
            {
                "window_index": w.window_index,
                "est_tokens_system_plus_user": est,
                "est_attached_diagram_tokens": estimate_text_tokens(
                    w.attached_diagrams or ""
                ),
                "paragraph_ids": w.paragraph_ids,
                "figure_ids_in_window": w.figure_ids,
                "body_chars": len(w.body),
            }
        )
        combined_parts.append(
            f"\n\n{'=' * 72}\n"
            f"WINDOW {w.window_index} (~{est} tok) "
            f"P={len(w.paragraph_ids)} FIG={len(w.figure_ids)}\n"
            f"{'=' * 72}\n\n"
            f"{user_prompt}"
        )

    (out_dir / "llm_all_map_windows.txt").write_text(
        "".join(combined_parts).strip() + "\n",
        encoding="utf-8",
    )
    (map_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export spatial Map-Reduce LLM payloads + raw annotated PDF text",
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <stem>_annotated next to PDF)",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Canonical URL (as in ingest job; default: file://…)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Article title in MAP header (default: first title-like P_n or URL)",
    )
    parser.add_argument(
        "--with-registry",
        action="store_true",
        help="Build FigureRegistry (anchors only; no VLM) for [ATTACHED_DIAGRAMS] in MAP export",
    )
    args = parser.parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        print(f"not found: {pdf_path}", file=sys.stderr)
        return 2

    out_dir = args.out or (pdf_path.parent / f"{pdf_path.stem}_annotated")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        build_annotated_from_content,
    )
    from knowledge_engine.services.parsers.pdf_bytes import is_parseable_pdf
    from knowledge_engine.services.parsers.vector_pdf_cropper import export_status_label

    raw = pdf_path.read_bytes()
    if not is_parseable_pdf(raw):
        print(f"PDF not parseable: {len(raw)} bytes", file=sys.stderr)
        return 3

    page_url = (args.url or "").strip() or pdf_path.as_uri()
    annotated = build_annotated_from_content(raw, page_url)
    from knowledge_engine.services.article_ingestion.document_triage_engine import (
        triage_annotated_article,
    )

    pruned, triage_outcome = triage_annotated_article(
        annotated, raw=raw, source_format="pdf"
    )

    audit = _audit_raw_blocks(raw)

    (out_dir / "annotated_raw.txt").write_text(
        annotated.annotated_markdown or "", encoding="utf-8"
    )
    triaged_path = out_dir / "annotated_after_triage.txt"
    triaged_path.write_text(pruned.annotated_markdown or "", encoding="utf-8")

    title = (args.title or "").strip() or _guess_title(pruned) or page_url
    figure_registry = None
    if args.with_registry:
        from knowledge_engine.services.article_diagram_context import (
            canonical_article_id,
        )
        from knowledge_engine.services.article_ingestion.figure_registry_service import (
            persist_figure_registry,
        )

        aid = canonical_article_id("", page_url)
        figure_registry = persist_figure_registry(aid, pruned)
        (out_dir / "figure_registry.json").write_text(
            json.dumps(
                {
                    fid: {
                        "labels": ent.labels,
                        "caption": ent.caption,
                        "page_no": ent.page_no,
                        "vlm_summary": (ent.vlm_summary or "")[:200],
                    }
                    for fid, ent in figure_registry.entries.items()
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    map_manifest = _export_llm_map_windows(
        out_dir,
        annotated_after_triage=pruned,
        page_url=page_url,
        title=title,
        figure_registry=figure_registry,
    )

    # legacy alias
    annotated_path = out_dir / "annotated.txt"
    annotated_path.write_text(pruned.annotated_markdown or "", encoding="utf-8")

    fig_lines_in_md: set[str] = set()
    for line in (pruned.annotated_markdown or "").splitlines():
        m = _FIG_LINE_RE.match(line.strip())
        if m:
            fig_lines_in_md.add(m.group(1))

    fig_rows: list[dict] = []
    sources = pruned.fig_extract_source or {}
    topo_map = pruned.fig_extract_topology or {}
    all_fig_keys = sorted(
        set(pruned.fig_map.keys()) | set(pruned.fig_bytes.keys()) | set(sources.keys()),
        key=_fig_sort_key,
    )
    for key in all_fig_keys:
        source = sources.get(key, "unknown")
        status = export_status_label(source, topo_map.get(key))
        payload = pruned.fig_bytes.get(key)
        nbytes = len(payload[0]) if payload else 0
        fig_rows.append(
            {
                "id": key,
                "in_annotated_txt": key in fig_lines_in_md,
                "source": source,
                "status": status,
                "bytes": nbytes,
                "fig_map": pruned.fig_map.get(key),
            }
        )

    (out_dir / "figure_index.json").write_text(
        json.dumps(fig_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    paragraph_page = pruned.paragraph_page or {}
    para_lines = [
        f"{pid}\tpage={paragraph_page.get(pid, 0)}\t{text[:200].replace(chr(9), ' ')}"
        for pid, text in sorted(
            (pruned.paragraph_map or {}).items(),
            key=lambda x: int(x[0].split("_")[1]),
        )
    ]
    (out_dir / "paragraph_index.tsv").write_text(
        "\n".join(para_lines) + "\n",
        encoding="utf-8",
    )

    margin_hits = _paragraphs_in_margin(
        pruned.paragraph_map or {},
        pruned.paragraph_page or {},
        audit,
    )
    boiler_blocks = [r for r in audit if r.get("likely_boilerplate")]
    p_raw = len(annotated.paragraph_map or {})
    p_triaged = len(pruned.paragraph_map or {})
    report_lines = [
        f"PDF: {pdf_path.name}",
        f"pages: {len(audit) and max(r['page'] for r in audit)}",
        f"paragraphs P_n: {p_raw} raw → {p_triaged} after triage",
        f"MAP windows: {len(map_manifest)}",
        f"figures in fig_bytes: {len(pruned.fig_bytes or {})}",
        f"FIG markers in triaged txt: {len(fig_lines_in_md)}",
        f"raw text blocks (margin/boilerplate flagged): {len(boiler_blocks)} / {len(audit)}",
        "",
        "LLM payload: llm_all_map_windows.txt + llm_map_windows/window_*_user.txt",
        "System: llm_map_windows/00_system_prompt.txt",
        "",
        "NOTE: margin boilerplate is NOT auto-stripped in annotate; triage may prune sections via TOC.",
        "",
        "--- likely boilerplate blocks (raw PDF) ---",
    ]
    for row in boiler_blocks[:80]:
        report_lines.append(
            f"p{row['page']} {row.get('zone') or '-':12} y={row['y0']:.0f}-{row['y1']:.0f} | "
            f"{row['preview']}"
        )
    if len(boiler_blocks) > 80:
        report_lines.append(f"... +{len(boiler_blocks) - 80} more")
    report_lines.append("")
    report_lines.append("--- P_* paragraphs overlapping margin boilerplate ---")
    if not margin_hits:
        report_lines.append("(none matched by preview)")
    for pid, page, preview in margin_hits[:40]:
        report_lines.append(f"{pid} p{page} | {preview}")
    report_lines.append("")
    report_lines.append("--- FIG summary ---")
    for row in fig_rows:
        mark = "in_txt" if row["in_annotated_txt"] else "bytes_only"
        report_lines.append(
            f"{row['id']:12} {mark:10} {row['status']:24} {row['bytes']:8} B  {row['source']}"
        )

    (out_dir / "margin_audit.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"annotated_after_triage.txt → {triaged_path}")
    print(f"llm_all_map_windows.txt → {out_dir / 'llm_all_map_windows.txt'}")
    print(
        f"P {p_raw}→{p_triaged} | MAP windows={len(map_manifest)} | "
        f"FIG_bytes={len(pruned.fig_bytes or {})} | "
        f"FIG_in_txt={len(fig_lines_in_md)}"
    )
    print(f"→ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
