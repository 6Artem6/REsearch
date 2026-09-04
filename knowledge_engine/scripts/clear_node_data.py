"""Unified node-data cleanup — LOCAL + CLOUD + curriculum-library GC in one place.

Combines three existing scripts, in this order, driven by --curriculum-id/--node-id:

  1. CLOUD   (cleanup_cloud_resources.cleanup_qdrant / cleanup_redis) — deletes
     this node's resolved source URLs from Qdrant (rag_chunks/document_summaries/
     knowledge_atoms, by url/doc_id — Qdrant payloads carry no node_id/
     curriculum_id field, so resolution goes through the curriculum graph's
     source registry, same as step 2), plus the node's Redis grounding-lock key.
  2. LOCAL   (clear_node_sources.apply_clear) — LanceDB rag_chunks/summaries/
     knowledge_atoms by the SAME resolved URLs, graph node patch
     (mapped_source_ids/grounding_status/etc. reset), node session reset,
     blocklist domain removal, curriculum_sources_registry key_extracts scrub.
     knowledge_atoms writes moved to Qdrant-only a while back, but pre-migration
     rows were left behind locally with no cleanup path until this stage was
     wired up — see VectorStore.delete_knowledge_atoms_for_urls docstring.
  3. LIBRARY GC (sync_curriculum_library_sources.apply_sync) — runs AFTER step 2
     has emptied this node's mapped_source_ids, so any curriculum_sources_registry
     entry no longer referenced by ANY node in the curriculum is now orphaned;
     prunes those, re-syncs remaining node sessions to the cleaned registry, and
     removes their LanceDB rag_chunks/summaries/knowledge_atoms rows, the same
     URLs' Qdrant Cloud points, article_diagrams (Mermaid) rows, and
     figure_registry (VLM figure) rows too
     — only if no OTHER saved curriculum still references that URL. Without
     this, diagrams/figures for a re-ingested source were reused verbatim from
     the old cache instead of regenerating.

Ordering matters: step 1 and step 2 both resolve this node's URLs from
mapped_source_ids as the graph stands BEFORE step 2's patch — CLOUD must run
first, or it would read the already-emptied mapped_source_ids and find nothing
to delete. Step 3 must run LAST, after step 2's patch is saved, or it can't see
this node's sources as orphaned yet.

In --dry-run (default) mode, step 3's preview is computed against the CURRENT
(pre-clear) graph — it does not simulate step 2's patch, so it will under-report
entries that would only become orphaned once step 2 actually runs. This is
called out explicitly in the printed report.

Each stage is independently skippable (--skip-cloud / --skip-library-gc) — one
stage failing does not abort the others.

Usage:
  PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/clear_node_data.py \\
    --curriculum-id agentic_systems_architecture --node-id governed_agent_pipelines
  # review the dry-run plan, then:
  PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/clear_node_data.py \\
    --curriculum-id agentic_systems_architecture --node-id governed_agent_pipelines --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.scripts.cleanup_cloud_resources import (
    cleanup_qdrant,
    cleanup_redis,
)
from knowledge_engine.scripts.clear_node_sources import apply_clear as local_apply_clear
from knowledge_engine.scripts.clear_node_sources import build_plan as local_build_plan
from knowledge_engine.scripts.sync_curriculum_library_sources import (
    apply_sync as library_apply_sync,
)
from knowledge_engine.scripts.sync_curriculum_library_sources import (
    build_plan as library_build_plan,
)


def run(
    curriculum_id: str,
    node_id: str,
    *,
    apply: bool,
    clear_blocklist: bool = True,
    scrub_registry: bool = True,
    skip_cloud: bool = False,
    skip_library_gc: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "curriculum_id": curriculum_id,
        "node_id": node_id,
        "apply": apply,
    }

    # Stage 1: CLOUD — must run before Stage 2 patches the graph; both resolve
    # this node's URLs from the same (still-unpatched) mapped_source_ids.
    if skip_cloud:
        report["cloud"] = {"skipped": True}
    else:
        try:
            report["cloud"] = {
                "redis": cleanup_redis(node_id, curriculum_id, dry_run=not apply),
                "qdrant": cleanup_qdrant(node_id, curriculum_id, dry_run=not apply),
            }
        except Exception as exc:
            report["cloud"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Stage 2: LOCAL — LanceDB + graph patch (empties mapped_source_ids) +
    # session reset + blocklist.
    try:
        if apply:
            report["local"] = local_apply_clear(
                curriculum_id,
                node_id,
                clear_blocklist=clear_blocklist,
                scrub_registry=scrub_registry,
            )
        else:
            plan = local_build_plan(
                curriculum_id, node_id, clear_blocklist=clear_blocklist
            )
            plan["applied"] = False
            report["local"] = plan
    except Exception as exc:
        report["local"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Stage 3: LIBRARY GC — only fully meaningful once Stage 2 has actually
    # patched+saved the graph (see module docstring re: dry-run preview limits).
    if skip_library_gc:
        report["library_gc"] = {"skipped": True}
    else:
        try:
            if apply:
                report["library_gc"] = library_apply_sync(curriculum_id)
            else:
                plan = library_build_plan(curriculum_id)
                plan["applied"] = False
                plan.pop("_kept_registry", None)
                plan.pop("_session_sync", None)
                report["library_gc"] = plan
        except Exception as exc:
            report["library_gc"] = {"error": f"{type(exc).__name__}: {exc}"}

    return report


def _print_human(report: dict[str, Any]) -> None:
    cid, nid, applied = report["curriculum_id"], report["node_id"], report["apply"]
    print(
        f"=== CLEAR NODE DATA | curriculum_id={cid!r} node_id={nid!r} apply={applied} ==="
    )

    cloud = report.get("cloud") or {}
    if cloud.get("skipped"):
        print("\n[1/3] CLOUD: skipped (--skip-cloud)")
    elif cloud.get("error"):
        print(f"\n[1/3] CLOUD: ERROR — {cloud['error']}")
    else:
        redis_r, qdrant_r = cloud.get("redis") or {}, cloud.get("qdrant") or {}
        print(
            f"\n[1/3] CLOUD — Redis: found={len(redis_r.get('keys_found') or [])} "
            f"deleted={redis_r.get('keys_deleted', 0)}"
        )
        if qdrant_r.get("applicable", True):
            print(
                f"      Qdrant: urls_found={len(qdrant_r.get('urls_found') or [])} "
                f"urls_deleted={qdrant_r.get('urls_deleted', 0)}"
            )
        else:
            print("      Qdrant: not applicable")

    local = report.get("local") or {}
    if local.get("error"):
        print(f"\n[2/3] LOCAL: ERROR — {local['error']}")
    elif local.get("error_") or local.get("error"):
        print(f"\n[2/3] LOCAL: {local.get('error')}")
    else:
        print(f"\n[2/3] LOCAL — mapped_source_ids={local.get('mapped_source_ids')}")
        print(f"      urls ({len(local.get('urls') or [])}):")
        for u in local.get("urls") or []:
            print(f"        - {u}")
        if applied:
            print(
                f"      Lance rag_chunks removed={local.get('lance_rag_chunks_removed')} "
                f"summaries={local.get('lance_summaries_removed')} "
                f"knowledge_atoms={local.get('lance_knowledge_atoms_removed')}"
            )
            print(
                f"      registry scrubbed={local.get('registry_entries_scrubbed')} "
                f"node_patched={local.get('graph_node_patched')} "
                f"session_reset={local.get('session_reset')} "
                f"blocklist_removed={local.get('blocklist_domains_removed')}"
            )
        else:
            print("      dry-run — re-run with --apply to execute")

    lib = report.get("library_gc") or {}
    if lib.get("skipped"):
        print("\n[3/3] LIBRARY GC: skipped (--skip-library-gc)")
    elif lib.get("error"):
        print(f"\n[3/3] LIBRARY GC: ERROR — {lib['error']}")
    else:
        print(
            f"\n[3/3] LIBRARY GC — registry: {lib.get('registry_before')} → "
            f"{lib.get('registry_after')} | orphaned={len(lib.get('orphaned_registry_entries') or [])}"
        )
        if not applied:
            print(
                "      (preview against PRE-clear graph — Stage 2 hasn't run yet in "
                "dry-run, so this UNDER-reports what will be orphaned after --apply)"
            )
        else:
            print(
                f"      sessions_synced={lib.get('sessions_synced')} "
                f"lance_rag_chunks_removed={lib.get('lance_rag_chunks_removed')} "
                f"lance_summaries_removed={lib.get('lance_summaries_removed')} "
                f"lance_knowledge_atoms_removed={lib.get('lance_knowledge_atoms_removed')} "
                f"qdrant_urls_removed={lib.get('qdrant_urls_removed')} "
                f"diagrams_removed={lib.get('diagrams_removed')} "
                f"figure_registry_removed={lib.get('figure_registry_removed')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--curriculum-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--apply", action="store_true", help="Execute (default: dry-run plan only)"
    )
    parser.add_argument(
        "--skip-cloud", action="store_true", help="Skip Qdrant/Redis stage"
    )
    parser.add_argument(
        "--skip-library-gc",
        action="store_true",
        help="Skip curriculum-library GC stage",
    )
    parser.add_argument("--no-clear-blocklist", action="store_true")
    parser.add_argument("--no-scrub-registry", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(
        args.curriculum_id.strip(),
        args.node_id.strip(),
        apply=args.apply,
        clear_blocklist=not args.no_clear_blocklist,
        scrub_registry=not args.no_scrub_registry,
        skip_cloud=args.skip_cloud,
        skip_library_gc=args.skip_library_gc,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
