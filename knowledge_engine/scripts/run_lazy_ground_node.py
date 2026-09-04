"""Принудительный targeted search + summarizer/spatial для одной DEEP-ноды (без UI init)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Direct execution places ``knowledge_engine/scripts`` on sys.path, rather
# than the repository root required by absolute ``knowledge_engine.*`` imports.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    get_curriculum_meta,
)
from knowledge_engine.src.curriculum.curriculum_lancedb_persist import (
    persist_approved_curriculum_hits_to_lancedb_async,
)
from knowledge_engine.src.curriculum.schemas import CurriculumGraph
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.curriculum.source_material_pipeline import (
    enrich_search_hits_with_extracts_async,
    summarize_whitelist_blog_hits_async,
)
from knowledge_engine.src.curriculum.source_policy import resolve_source_policy
from knowledge_engine.src.curriculum.targeted_node_search import (
    search_sources_for_deep_node_async,
)
from knowledge_engine.ui.run_log import init_run_log, trace


async def _run(
    curriculum_id: str,
    node_id: str,
    *,
    on_demand: bool,
    also_spatial_url: str | None,
    also_spatial_pdf: str | None = None,
) -> int:
    raw = get_curriculum_graph(curriculum_id)
    if not raw:
        print(f"no graph for curriculum_id={curriculum_id}", file=sys.stderr)
        return 2
    graph = CurriculumGraph.model_validate(raw)
    node = next((n for n in graph.nodes if n.node_id == node_id), None)
    if not node:
        print(f"node_id={node_id} not in graph", file=sys.stderr)
        return 2

    meta = get_curriculum_meta(curriculum_id) or {}
    target_goal = str(meta.get("target_goal") or graph.description or "").strip()
    source_policy = resolve_source_policy(
        meta.get("source_policy"),
        str(meta.get("generation_mode") or "fast"),
        default="hybrid",
    )
    anchor = f"cli_lazy:{curriculum_id}:{node_id}"

    exclude: set[str] = set()
    for e in graph.curriculum_sources_registry:
        key = _normalize_url_key(e.url)
        if key:
            exclude.add(key)

    trace(
        f"CLI lazy ground ▶ | {curriculum_id}/{node_id} "
        f"title={node.title!r} on_demand={on_demand} exclude_urls={len(exclude)}"
    )

    hits = await search_sources_for_deep_node_async(
        node,
        target_goal,
        source_policy=source_policy,
        anchor=anchor,
        exclude_url_keys=exclude,
        on_demand=on_demand,
        registry_entries=list(graph.curriculum_sources_registry),
    )
    trace(f"CLI lazy ground search ✓ | hits={len(hits)}")
    for h in hits[:12]:
        trace(
            f"  hit | tier={h.source_tier} skip_ollama={h.skip_ollama_summary} "
            f"extracts={len(h.key_extracts)} | {h.url[:85]}"
        )

    if hits:
        hits = await summarize_whitelist_blog_hits_async(hits, target_goal)
        hits = await enrich_search_hits_with_extracts_async(hits, target_goal)
        await persist_approved_curriculum_hits_to_lancedb_async(
            hits,
            label=f"lazy_ground:{node_id}",
        )
        trace(f"CLI lazy ground summarize ✓ | hits={len(hits)}")

    extra_ingest_urls: list[str] = []
    if hits:
        extra_ingest_urls.extend(
            [
                str(h.url or "").strip()
                for h in hits
                if str(h.url or "").startswith("http")
            ]
        )
    spatial_canonical = (also_spatial_url or "").strip()
    if spatial_canonical:
        extra_ingest_urls.append(spatial_canonical)

    if also_spatial_url or also_spatial_pdf:
        from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
            prepare_spatial_diagram_job,
            run_spatial_diagram_ingest_jobs_async,
        )

        url = (also_spatial_url or "").strip()
        if not url and also_spatial_pdf:
            url = Path(also_spatial_pdf).expanduser().resolve().as_uri()
        pdf_path = (also_spatial_pdf or "").strip()
        raw_bytes: bytes | None = None
        if pdf_path:
            raw_bytes = Path(pdf_path).expanduser().read_bytes()
            if not url.startswith("http"):
                print(
                    "also-spatial-pdf requires --also-spatial-url (canonical DOI)",
                    file=sys.stderr,
                )
                return 2
        trace(f"CLI lazy ground spatial extra ▶ | {url[:80]}")
        job = await asyncio.to_thread(
            prepare_spatial_diagram_job,
            "cli_spatial",
            url,
            raw_bytes=raw_bytes,
        )
        if job is None:
            trace("CLI lazy ground spatial extra ⊘ | prepare returned None")
        else:
            n_fig = len(job.annotated.fig_map)
            trace(
                f"CLI lazy ground spatial extra | FIG={n_fig} P={len(job.annotated.paragraph_map)}"
            )
            saved = await run_spatial_diagram_ingest_jobs_async([job])
            trace(f"CLI lazy ground spatial extra ✓ | vlm_saved={saved}")

    from knowledge_engine.src.node_deep_dive.diagram_session import (
        curriculum_node_to_data_input,
        refresh_node_session_diagrams_from_articles,
    )

    n_diag = refresh_node_session_diagrams_from_articles(
        curriculum_id,
        curriculum_node_to_data_input(node),
        extra_urls=extra_ingest_urls,
        rebuild=True,
    )
    trace(
        f"CLI lazy ground session diagrams ✓ | content.diagrams={n_diag} "
        f"ingest_urls={len(extra_ingest_urls)}"
    )

    trace("CLI lazy ground ✓ | done")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Force targeted search + spatial for one node"
    )
    parser.add_argument("--curriculum-id", default="vector_db_mechanics")
    parser.add_argument("--node-id", default="hybrid_search_fusion")
    parser.add_argument(
        "--on-demand",
        action="store_true",
        default=True,
        help="Как lazy init (default true)",
    )
    parser.add_argument(
        "--full-academic",
        action="store_true",
        help="on_demand=False (дольше, без fast-return academic)",
    )
    parser.add_argument(
        "--also-spatial-url",
        default="",
        help="Доп. spatial ingest по URL (пусто = не запускать)",
    )
    parser.add_argument(
        "--also-spatial-pdf",
        default="",
        help="Локальный PDF для spatial ingest (canonical URL = --also-spatial-url)",
    )
    args = parser.parse_args()
    on_demand = not args.full_academic
    spatial_url = (args.also_spatial_url or "").strip() or None
    init_run_log(f"lazy ground {args.node_id}")
    return asyncio.run(
        _run(
            args.curriculum_id,
            args.node_id,
            on_demand=on_demand,
            also_spatial_url=spatial_url,
            also_spatial_pdf=(args.also_spatial_pdf or "").strip() or None,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
