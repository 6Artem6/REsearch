"""Sanitize Mermaid in all node sessions for a curriculum (backfill).

Pipeline matches serve/persist: normalize_stored_mermaid (regex + format)
before validation; optional Gemma repair for still-invalid diagrams.

Gemma repairs run in parallel (shared httpx + RateLimitedLLMClient) with
``--concurrency`` (default: GEMMA_CONCURRENCY / MAX_CONCURRENT_MAP_REQUESTS).

Example:
  ./.venv/bin/python -m knowledge_engine.scripts.sanitize_curriculum_diagrams \\
      --curriculum-id agentic_systems_architecture --with-gemma
  ./.venv/bin/python -m knowledge_engine.scripts.sanitize_curriculum_diagrams \\
      --curriculum-id agentic_systems_architecture --with-gemma --concurrency 4
  ./.venv/bin/python -m knowledge_engine.scripts.sanitize_curriculum_diagrams \\
      --curriculum-id agentic_systems_architecture \\
      --node-id multi_agent_orchestration \\
      --diagram-ids diagram-4 diagram-6 --with-gemma --force-gemma
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from knowledge_engine.services.mermaid_validate import (
    process_mermaid_for_ingest_async,
    strip_mermaid_fences,
    validate_mermaid_syntax,
)
from knowledge_engine.src.node_deep_dive.content_assets import (
    normalize_node_content_diagrams,
)
from knowledge_engine.src.node_deep_dive.session_store import get_session, save_session


def _list_session_keys(curriculum_id: str) -> list[tuple[str, str]]:
    import json

    from knowledge_engine.config import PACKAGE_ROOT

    store = PACKAGE_ROOT / ".runs" / "node_deep_dive_sessions.json"
    cid = (curriculum_id or "").strip()
    prefix = f"{cid}::"
    if not store.is_file():
        return []
    try:
        raw = json.loads(store.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    out: list[tuple[str, str]] = []
    for key in sorted(raw.keys()):
        if not str(key).startswith(prefix):
            continue
        parts = str(key).split("::", 1)
        if len(parts) != 2 or not parts[1].strip():
            continue
        out.append((parts[0], parts[1]))
    return out


def _normalize_diagram_selector(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if re.fullmatch(r"\d+", s):
        return f"diagram-{s}"
    return s


def _parse_diagram_selectors(items: list[str] | None) -> set[str]:
    out: set[str] = set()
    for raw in items or []:
        s = _normalize_diagram_selector(raw)
        if s:
            out.add(s)
    return out


def _diagram_selected(
    *,
    diagram_id: str,
    title: str,
    selectors: set[str],
) -> bool:
    """Match by id (`diagram-4` / `4`) or case-insensitive title substring/equality."""
    if not selectors:
        return True
    did = (diagram_id or "").strip()
    title_l = (title or "").strip().lower()
    for sel in selectors:
        sel_l = sel.lower()
        if did.lower() == sel_l:
            return True
        if re.fullmatch(r"diagram-\d+", did, re.I) and sel_l == did.lower().removeprefix(
            "diagram-"
        ):
            return True
        if title_l and (title_l == sel_l or sel_l in title_l):
            return True
    return False


def _repair_deterministic(raw: str) -> tuple[str, str, bool]:
    """
    Normalize without Gemma.
    Returns (mermaid, reason, needs_gemma).
    """
    text = (raw or "").strip()
    if not text:
        return "", "empty", False
    from knowledge_engine.services.mermaid_validate import normalize_stored_mermaid

    normed = normalize_stored_mermaid(text)
    if not normed:
        return text, "normalize_empty", True
    inner = strip_mermaid_fences(normed)
    if validate_mermaid_syntax(inner):
        if normed.strip() != text:
            return normed.strip(), "normalized", False
        return text, "already_valid", False
    return normed.strip() or text, "needs_gemma", True


@dataclass
class _Job:
    cid: str
    nid: str
    diagram_id: str  # "diagram-N" or "__legacy__"
    disk_before: str
    title: str
    after: str = ""
    reason: str = ""
    needs_gemma: bool = False


async def _run_gemma_jobs(
    jobs: list[_Job],
    *,
    concurrency: int,
) -> None:
    if not jobs:
        return
    import httpx

    from knowledge_engine.services.llm.gemma_client import RateLimitedLLMClient

    sem = asyncio.Semaphore(max(1, concurrency))
    timeout = httpx.Timeout(120.0)
    rl = RateLimitedLLMClient()
    print(
        f"gemma parallel | jobs={len(jobs)} concurrency={max(1, concurrency)}",
        flush=True,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def _one(job: _Job) -> None:
            async with sem:
                print(
                    f"  ▶ gemma | {job.cid}::{job.nid} | {job.diagram_id}",
                    flush=True,
                )
                fixed = await process_mermaid_for_ingest_async(
                    job.after or job.disk_before,
                    allow_gemma_repair=True,
                    client=client,
                    rl=rl,
                )
                if not fixed:
                    job.reason = "gemma_failed"
                    print(
                        f"  ✗ gemma | {job.cid}::{job.nid} | {job.diagram_id} | failed",
                        flush=True,
                    )
                    return
                if validate_mermaid_syntax(strip_mermaid_fences(fixed)):
                    job.after = fixed.strip()
                    job.reason = "gemma_repaired"
                else:
                    job.after = fixed.strip()
                    job.reason = "gemma_still_invalid"
                print(
                    f"  ✓ gemma | {job.cid}::{job.nid} | {job.diagram_id} | {job.reason}",
                    flush=True,
                )

        await asyncio.gather(*[_one(j) for j in jobs])


def sanitize_curriculum_sessions(
    curriculum_id: str,
    *,
    dry_run: bool = False,
    allow_gemma: bool = False,
    concurrency: int = 4,
    node_ids: list[str] | None = None,
    diagram_ids: list[str] | None = None,
    force_gemma: bool = False,
) -> dict[str, int]:
    import json

    from knowledge_engine.config import PACKAGE_ROOT
    from knowledge_engine.src.node_deep_dive.schemas import (
        DiagramAsset,
        NodeContentBlock,
    )

    store = PACKAGE_ROOT / ".runs" / "node_deep_dive_sessions.json"
    stats = {
        "sessions": 0,
        "diagrams": 0,
        "changed": 0,
        "invalid_after": 0,
        "gemma_jobs": 0,
    }
    try:
        all_data = json.loads(store.read_text(encoding="utf-8"))
    except Exception:
        all_data = {}
    if not isinstance(all_data, dict):
        all_data = {}

    node_filter = {n.strip() for n in (node_ids or []) if (n or "").strip()}
    diagram_filter = _parse_diagram_selectors(diagram_ids)

    session_jobs: dict[str, list[_Job]] = {}
    gemma_queue: list[_Job] = []

    for cid, nid in _list_session_keys(curriculum_id):
        if node_filter and nid not in node_filter:
            continue
        key = f"{cid}::{nid}"
        blob = all_data.get(key) or {}
        content_raw = blob.get("content") or {}
        try:
            content = NodeContentBlock.model_validate(content_raw)
        except Exception as exc:
            print(f"{key} | skip invalid content | {exc}")
            continue

        raw_by_id = {
            str(d.id or "").strip(): (d.mermaid or "").strip()
            for d in (content.diagrams or [])
        }
        legacy_raw = (content.diagram or "").strip()
        content = normalize_node_content_diagrams(content)
        diagrams = list(content.diagrams or [])
        if not diagrams and not legacy_raw:
            continue

        stats["sessions"] += 1
        jobs: list[_Job] = []

        for d in diagrams:
            did = str(d.id or "").strip() or "diagram"
            title = (d.title or "").strip()
            if not _diagram_selected(
                diagram_id=did, title=title, selectors=diagram_filter
            ):
                continue
            stats["diagrams"] += 1
            disk_before = raw_by_id.get(did, (d.mermaid or "").strip())
            after, reason, needs = _repair_deterministic(disk_before)
            force = bool(force_gemma and allow_gemma)
            job = _Job(
                cid=cid,
                nid=nid,
                diagram_id=did,
                disk_before=disk_before,
                title=title[:60],
                after=after,
                reason=reason,
                needs_gemma=bool((needs or force) and allow_gemma),
            )
            if job.needs_gemma:
                if force and not needs:
                    job.reason = "force_gemma"
                    job.after = disk_before
                gemma_queue.append(job)
            else:
                if not allow_gemma and needs:
                    job.reason = "normalized_still_invalid"
            jobs.append(job)

        if not diagrams and legacy_raw:
            if _diagram_selected(
                diagram_id="__legacy__",
                title="legacy.diagram",
                selectors=diagram_filter,
            ):
                stats["diagrams"] += 1
                after, reason, needs = _repair_deterministic(legacy_raw)
                force = bool(force_gemma and allow_gemma)
                job = _Job(
                    cid=cid,
                    nid=nid,
                    diagram_id="__legacy__",
                    disk_before=legacy_raw,
                    title="legacy.diagram",
                    after=after,
                    reason=reason,
                    needs_gemma=bool((needs or force) and allow_gemma),
                )
                if job.needs_gemma:
                    if force and not needs:
                        job.reason = "force_gemma"
                        job.after = legacy_raw
                    gemma_queue.append(job)
                elif not allow_gemma and needs:
                    job.reason = "normalized_still_invalid"
                jobs.append(job)

        if jobs:
            session_jobs[key] = jobs

    stats["gemma_jobs"] = len(gemma_queue)
    print(
        f"collected | sessions={stats['sessions']} diagrams={stats['diagrams']} "
        f"gemma_jobs={stats['gemma_jobs']}"
        + (f" nodes={sorted(node_filter)}" if node_filter else "")
        + (f" diagrams={sorted(diagram_filter)}" if diagram_filter else ""),
        flush=True,
    )
    if gemma_queue:
        asyncio.run(_run_gemma_jobs(gemma_queue, concurrency=concurrency))

    for key, jobs in session_jobs.items():
        if not jobs:
            continue
        cid, nid = key.split("::", 1)
        blob = all_data.get(key) or {}
        content_raw = blob.get("content") or {}
        try:
            content = NodeContentBlock.model_validate(content_raw)
        except Exception:
            continue
        content = normalize_node_content_diagrams(content)
        diagrams = list(content.diagrams or [])
        by_id = {j.diagram_id: j for j in jobs}
        changed = False
        new_diagrams: list[DiagramAsset] = []

        for d in diagrams:
            did = str(d.id or "").strip()
            job = by_id.get(did)
            if job is None:
                new_diagrams.append(d)
                continue
            after = job.after or job.disk_before
            ok = validate_mermaid_syntax(strip_mermaid_fences(after)) if after else False
            print(
                f"{key} | {job.diagram_id} | {job.reason} | valid={ok} | {job.title!r}",
                flush=True,
            )
            if after and after.strip() != job.disk_before.strip():
                changed = True
                stats["changed"] += 1
                new_diagrams.append(d.model_copy(update={"mermaid": after}))
            else:
                new_diagrams.append(
                    d.model_copy(update={"mermaid": after or d.mermaid})
                )
            if after and not ok:
                stats["invalid_after"] += 1

        legacy_job = by_id.get("__legacy__")
        if legacy_job is not None:
            after = legacy_job.after or legacy_job.disk_before
            print(f"{key} | legacy.diagram | {legacy_job.reason}", flush=True)
            if after and after.strip() != legacy_job.disk_before.strip():
                changed = True
                stats["changed"] += 1
                content = content.model_copy(update={"diagram": after})

        if changed:
            content = content.model_copy(update={"diagrams": new_diagrams})
            if not dry_run:
                session = get_session(cid, nid)
                save_session(
                    cid,
                    nid,
                    session.node_status,
                    content,
                    session.history,
                    memory=session.memory,
                )
                print(f"  saved {key}", flush=True)
            else:
                print(f"  dry-run skip save {key}", flush=True)

    return stats


def main() -> int:
    from knowledge_engine.config import GEMMA_CONCURRENCY

    parser = argparse.ArgumentParser(
        description="Sanitize Mermaid diagrams for all node sessions in a curriculum"
    )
    parser.add_argument(
        "--curriculum-id",
        required=True,
        help="e.g. agentic_systems_architecture",
    )
    parser.add_argument(
        "--node-id",
        action="append",
        default=[],
        help="Limit to node id(s); repeatable. e.g. --node-id multi_agent_orchestration",
    )
    parser.add_argument(
        "--diagram-ids",
        nargs="+",
        default=[],
        help=(
            "Limit to diagram id(s) and/or titles: diagram-4 6 "
            "'Base topology' (space-separated list)"
        ),
    )
    parser.add_argument(
        "--with-gemma",
        action="store_true",
        help="Run Gemma repair when deterministic sanitize still fails validation",
    )
    parser.add_argument(
        "--force-gemma",
        action="store_true",
        help="With --with-gemma: send selected diagrams to Gemma even if currently valid",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=max(1, int(GEMMA_CONCURRENCY)),
        help=f"Parallel Gemma repairs (default {GEMMA_CONCURRENCY})",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = sanitize_curriculum_sessions(
        args.curriculum_id.strip(),
        dry_run=args.dry_run,
        allow_gemma=bool(args.with_gemma),
        concurrency=max(1, int(args.concurrency)),
        node_ids=list(args.node_id or []),
        diagram_ids=list(args.diagram_ids or []),
        force_gemma=bool(args.force_gemma),
    )
    print(
        f"done | sessions={stats['sessions']} diagrams={stats['diagrams']} "
        f"gemma_jobs={stats['gemma_jobs']} changed={stats['changed']} "
        f"invalid_after={stats['invalid_after']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
