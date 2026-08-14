"""Smoke / regression check for PaperStructureAnalyzer (Gemini Flash Lite).

Usage (repo root, .env with GEMINI_API_KEY):
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_paper_structure_analyzer \\
    --url https://arxiv.org/pdf/2507.03226 \\
    --topic "GraphRAG knowledge graph construction hybrid retrieval dependency parsing"

  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_paper_structure_analyzer \\
    --pdf /path/to/paper.pdf --topic "..." --json-out report.json

  # Local regex fallback only (no Gemini):
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_paper_structure_analyzer \\
    --url https://arxiv.org/pdf/2507.03226 --fallback-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from knowledge_engine.config import (
    GEMINI_LITE_MODEL,
    refresh_vlm_gemini_env_from_dotenv,
)
from knowledge_engine.services.gemini_stateless import is_gemini_available
from knowledge_engine.services.vlm_gemini_pool import resolve_vlm_pool_model_ids
from knowledge_engine.src.parsers.paper_input_json import (
    build_input_paper_json_from_pdf_bytes,
    input_paper_json_for_llm,
)
from knowledge_engine.src.parsers.paper_structure_analyzer import (
    _PAPER_STRUCTURE_SYSTEM_STATIC,
    PaperStructureAnalyzer,
    _min_relevance_for_priority,
    _paragraph_in_references_range,
    _resolve_references_range,
    apply_structure_filter,
    local_fallback_analysis,
    try_fetch_pdf_bytes_for_url,
)
from knowledge_engine.src.parsers.paper_structure_schema import ParagraphPriority

_DEFAULT_URL = "https://arxiv.org/pdf/2507.03226"
_DEFAULT_TOPIC = (
    "Efficient knowledge graph construction and hybrid GraphRAG retrieval "
    "for enterprise RAG (dependency parsing, RRF, legacy code migration)"
)


def _load_pdf_bytes(url: str | None, pdf_path: str | None) -> bytes:
    if pdf_path:
        data = Path(pdf_path).read_bytes()
        if data[:5] != b"%PDF-":
            raise SystemExit(f"Not a PDF: {pdf_path}")
        return data
    if not url:
        raise SystemExit("Provide --url or --pdf")
    data = try_fetch_pdf_bytes_for_url(url)
    if not data:
        import httpx

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url.strip())
        data = resp.content
    if not data or data[:5] != b"%PDF-":
        raise SystemExit("Failed to download a valid PDF")
    return data


def _paper_stats(paper) -> dict:
    paras_per_page = [len(p.paragraphs) for p in paper.pages]
    total_paras = sum(paras_per_page)
    char_sum = sum(len(pa.text) for p in paper.pages for pa in p.paragraphs)
    return {
        "total_pages": paper.total_pages,
        "paragraph_count": total_paras,
        "chars_total": char_sum,
        "paragraphs_per_page_min": min(paras_per_page) if paras_per_page else 0,
        "paragraphs_per_page_max": max(paras_per_page) if paras_per_page else 0,
        "paragraphs_per_page_avg": (
            round(total_paras / len(paras_per_page), 1) if paras_per_page else 0
        ),
    }


def _analysis_stats(analysis) -> dict:
    by_pri = Counter(p.priority.value for p in analysis.paragraphs)
    rel_bins = Counter()
    for p in analysis.paragraphs:
        if p.topic_relevance >= 8:
            rel_bins["8-10"] += 1
        elif p.topic_relevance >= 6:
            rel_bins["6-7"] += 1
        elif p.topic_relevance >= 4:
            rel_bins["4-5"] += 1
        else:
            rel_bins["0-3"] += 1
    return {
        "references_start_page": analysis.references_start_page,
        "drop_pages": analysis.drop_pages,
        "drop_page_count": len(analysis.drop_pages or []),
        "paragraph_labels": dict(by_pri),
        "relevance_bins": dict(rel_bins),
        "labeled_paragraphs": len(analysis.paragraphs),
    }


def _filter_stats(paper, analysis, min_rel: int) -> dict:
    ref_range = _resolve_references_range(paper, analysis)
    by_id = {p.paragraph_id: p for p in analysis.paragraphs}
    kept_core = kept_ctx = dropped_pri = dropped_rel = dropped_refs = 0
    last_para_id = max(p.paragraph_id for pg in paper.pages for p in pg.paragraphs)
    for page in paper.pages:
        for para in page.paragraphs:
            pid = para.paragraph_id
            if _paragraph_in_references_range(pid, ref_range):
                dropped_refs += 1
                continue
            row = by_id.get(pid)
            if row is None:
                kept_ctx += 1
                continue
            if row.priority == ParagraphPriority.DROP:
                dropped_pri += 1
                continue
            need = _min_relevance_for_priority(row.priority, min_rel)
            if row.topic_relevance < need:
                dropped_rel += 1
                continue
            if row.priority == ParagraphPriority.CORE:
                kept_core += 1
            else:
                kept_ctx += 1
    raw_chars = sum(len(p.text) for pg in paper.pages for p in pg.paragraphs)
    filtered = apply_structure_filter(paper, analysis, min_topic_relevance=min_rel)
    return {
        "min_topic_relevance": min_rel,
        "references_range": list(ref_range) if ref_range else None,
        "references_range_ends_at_doc_end": (
            ref_range is not None and ref_range[1] == last_para_id
        ),
        "kept_core_paragraphs": kept_core,
        "kept_context_paragraphs": kept_ctx,
        "dropped_by_priority_drop": dropped_pri,
        "dropped_by_low_relevance": dropped_rel,
        "dropped_from_references_cut": dropped_refs,
        "raw_chars": raw_chars,
        "filtered_chars": len(filtered),
        "retention_pct": (
            round(100.0 * len(filtered) / raw_chars, 1) if raw_chars else 0
        ),
        "conclusion_102_in_filtered": "scalable method for constructing enterprise-grade"
        in filtered,
        "conclusion_103_in_filtered": "dependency parsing provides a lightweight"
        in filtered,
        "references_lewis_in_filtered": "Patrick Lewis" in filtered,
    }


def _sample_rows(analysis, n: int = 5) -> dict:
    paras = list(analysis.paragraphs)
    core = [p for p in paras if p.priority == ParagraphPriority.CORE]
    drop = [p for p in paras if p.priority == ParagraphPriority.DROP]
    high = sorted(paras, key=lambda x: -x.topic_relevance)[:n]
    return {
        "top_relevance": [
            {
                "id": p.paragraph_id,
                "page": p.page_number,
                "section": p.section_title[:80],
                "priority": p.priority.value,
                "relevance": p.topic_relevance,
                "reason": p.reason[:120],
            }
            for p in high
        ],
        "sample_core": [
            {
                "id": p.paragraph_id,
                "page": p.page_number,
                "section": p.section_title[:80],
                "relevance": p.topic_relevance,
                "reason": p.reason[:120],
            }
            for p in core[:n]
        ],
        "sample_drop": [
            {
                "id": p.paragraph_id,
                "page": p.page_number,
                "section": p.section_title[:80],
                "relevance": p.topic_relevance,
                "reason": p.reason[:120],
            }
            for p in drop[:n]
        ],
    }


def _print_report(report: dict) -> None:
    meta = report["meta"]
    print("=" * 72)
    print("PaperStructureAnalyzer check")
    print("=" * 72)
    print(f"URL/PDF:     {meta.get('source')}")
    print(f"Topic:       {meta.get('topic')[:100]}")
    print(f"Mode:        {meta.get('mode')}")
    print(
        f"Gemini:      available={meta.get('gemini_available')} models={meta.get('models')}"
    )
    print(f"Lite model:  {meta.get('primary_model')}")
    print()
    ps = report["paper_stats"]
    print(
        f"PDF extract: {ps['total_pages']} pages, {ps['paragraph_count']} paragraphs, "
        f"{ps['chars_total']:,} chars"
    )
    print(
        f"  paras/page: min={ps['paragraphs_per_page_min']} max={ps['paragraphs_per_page_max']} "
        f"avg={ps['paragraphs_per_page_avg']}"
    )
    llm = report.get("llm_payload_stats", {})
    if llm:
        print(
            f"LLM payload: {llm.get('pages_sent')} pages, "
            f"{llm.get('paragraphs_sent')} paragraphs (truncated for API)"
        )
    print()
    an = report["analysis"]
    print(f"References start page: {an['references_start_page']}")
    print(
        f"Drop pages ({an['drop_page_count']}): {an['drop_pages'][:20]}{'...' if an['drop_page_count'] > 20 else ''}"
    )
    print(f"Labels: {an['paragraph_labels']}")
    print(f"Relevance bins: {an['relevance_bins']}")
    print()
    fs = report["filter"]
    print(
        f"References range (inclusive): {fs.get('references_range')} "
        f"ends_at_doc_end={fs.get('references_range_ends_at_doc_end')}"
    )
    print(
        f"After filter (min_rel={fs['min_topic_relevance']}): "
        f"kept CORE={fs['kept_core_paragraphs']} CONTEXT={fs['kept_context_paragraphs']} | "
        f"refs_cut={fs['dropped_from_references_cut']} DROP={fs['dropped_by_priority_drop']} "
        f"low_rel={fs['dropped_by_low_relevance']}"
    )
    print(
        f"Conclusion checks: #102={fs.get('conclusion_102_in_filtered')} "
        f"#103={fs.get('conclusion_103_in_filtered')} "
        f"refs_lewis={fs.get('references_lewis_in_filtered')}"
    )
    print(
        f"Chars: {fs['raw_chars']:,} → {fs['filtered_chars']:,} "
        f"({fs['retention_pct']}% retained)"
    )
    print()
    samples = report["samples"]
    print("--- Top relevance ---")
    for row in samples["top_relevance"]:
        print(
            f"  p{row['page']} #{row['id']} {row['priority']} rel={row['relevance']}: {row['reason']}"
        )
    print("--- Sample CORE ---")
    for row in samples["sample_core"]:
        print(f"  p{row['page']} #{row['id']} rel={row['relevance']}: {row['reason']}")
    if samples["sample_drop"]:
        print("--- Sample DROP ---")
        for row in samples["sample_drop"]:
            print(
                f"  p{row['page']} #{row['id']} rel={row['relevance']}: {row['reason']}"
            )
    print()
    print(
        f"Static system prompt length: {report['prompt']['static_chars']} chars (no target_topic/json)"
    )
    print(f"Dynamic payload length: {report['prompt']['dynamic_chars']} chars")
    if report.get("error"):
        print(f"ERROR: {report['error']}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check PaperStructureAnalyzer on a PDF"
    )
    parser.add_argument("--url", default=_DEFAULT_URL, help="PDF URL (arxiv pdf link)")
    parser.add_argument(
        "--pdf", default="", help="Local PDF path (overrides --url fetch)"
    )
    parser.add_argument(
        "--topic", default=_DEFAULT_TOPIC, help="target_topic for relevance"
    )
    parser.add_argument("--min-relevance", type=int, default=4)
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip Gemini; use local regex fallback only",
    )
    parser.add_argument("--json-out", default="", help="Write full report JSON to path")
    parser.add_argument(
        "--save-input-json",
        default="",
        help="Save input_paper_json (full extract) to path",
    )
    args = parser.parse_args(argv)

    refresh_vlm_gemini_env_from_dotenv()
    pdf_bytes = _load_pdf_bytes(args.url if not args.pdf else None, args.pdf or None)
    paper = build_input_paper_json_from_pdf_bytes(pdf_bytes)
    if args.save_input_json:
        Path(args.save_input_json).write_text(
            paper.model_dump_json(indent=2), encoding="utf-8"
        )

    paper_llm = input_paper_json_for_llm(paper)
    dynamic_chars = len(
        json.dumps(
            {"target_topic": args.topic, "input_paper_json": paper_llm},
            ensure_ascii=False,
        )
    )

    models = resolve_vlm_pool_model_ids()
    primary = models[0] if models else GEMINI_LITE_MODEL
    source = args.pdf or args.url

    error: str | None = None
    used_fallback = bool(args.fallback_only)

    if args.fallback_only:
        analysis = local_fallback_analysis(paper, args.topic)
    else:
        try:
            analyzer = PaperStructureAnalyzer()
            analysis = analyzer.analyze(
                args.topic,
                paper,
                label="check_script",
                anchor="paper_structure:check",
            )
            # Heuristic: fallback reasons are fixed strings
            if analysis.paragraphs and all(
                p.reason.startswith("fallback:") for p in analysis.paragraphs[:3]
            ):
                used_fallback = True
        except Exception as exc:
            error = str(exc)
            analysis = local_fallback_analysis(paper, args.topic)
            used_fallback = True

    report = {
        "meta": {
            "source": source,
            "topic": args.topic,
            "mode": "fallback" if used_fallback else "gemini_lite",
            "gemini_available": is_gemini_available(),
            "models": models,
            "primary_model": primary,
        },
        "paper_stats": _paper_stats(paper),
        "llm_payload_stats": {
            "pages_sent": len(paper_llm.get("pages", [])),
            "paragraphs_sent": sum(
                len(p.get("paragraphs", [])) for p in paper_llm.get("pages", [])
            ),
        },
        "analysis": _analysis_stats(analysis),
        "filter": _filter_stats(paper, analysis, args.min_relevance),
        "samples": _sample_rows(analysis),
        "prompt": {
            "static_chars": len(_PAPER_STRUCTURE_SYSTEM_STATIC),
            "dynamic_chars": dynamic_chars,
        },
        "error": error,
        "analysis_full": analysis.model_dump(),
    }

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _print_report(report)
    return 0 if not error and analysis.paragraphs else 1


if __name__ == "__main__":
    sys.exit(main())
