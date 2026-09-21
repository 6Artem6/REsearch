"""Dry-run audit: re-classify all real knowledge_atoms.scope values under the
NEW 5-category taxonomy (extraction_5_scopes.py: CONCEPT/MECHANIC/PRACTICE/
EDGE_CASE/ANTI_PATTERN) — WITHOUT mutating Postgres. Reads all 703 real atoms
(statement, current 3-category scope, article url/title), batches them 5-10
per LLM call, re-tags each under the NEW 5-category rules, and reports the
new distribution + 3-scope -> 5-scope transition matrix + contrastive
EDGE_CASE/ANTI_PATTERN examples to ./reclassification_5_scopes_result.log
(repo root).

Read-only against Postgres: only SELECT queries are issued, no
upsert/update/delete anywhere in this script.

This is a sibling of ``audit_and_reclassify_atoms.py`` (the 3-category dry
run, kept as-is for history) — reuses the same RPM-aware pacing,
retry-with-backoff on transient quota blocks, and resumable JSON cache that
script was fixed to use, adapted for the 5-category prompt and a separate
cache/output path so the two dry runs never collide.

Usage: python knowledge_engine/scripts/audit_and_reclassify_atoms_5_scopes.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from knowledge_engine.src.adapters.llm_providers.gemini_stateless import (
    GeminiQuotaExhaustedError,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.config import settings
from knowledge_engine.src.config.settings import (
    GEMINI_FLASH_LITE_MAX_RPM,
    GEMINI_LITE_MODEL,
)
from knowledge_engine.src.shared.extraction import (
    SCOPE_TAGGING_PROMPT_RULES as SCOPE_TAGGING_PROMPT_RULES_5,
)

_OUTPUT_PATH = Path(REPO_ROOT) / "reclassification_5_scopes_result.log"
_CACHE_PATH = (
    Path(REPO_ROOT)
    / "knowledge_engine"
    / ".runs"
    / "reclassification_5_scopes_cache.json"
)
_BATCH_SIZE = 8  # within the requested 5-10 range
_CONCURRENCY = 4  # within the requested 3-5 range
_RPM_BUDGET = max(1, GEMINI_FLASH_LITE_MAX_RPM - 3)
_QUOTA_RETRY_DELAYS_SEC = [10, 20, 40, 60, 60]

_NEW_SCOPES = ("CONCEPT", "MECHANIC", "PRACTICE", "EDGE_CASE", "ANTI_PATTERN")


class _RpmLimiter:
    """Sliding 60s-window limiter — paces dispatch, does not just cap
    concurrency (a concurrency-only Semaphore lets several requests fire in
    the same second, which trips the real per-minute quota)."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                wait = 60 - (now - self._timestamps[0]) + 0.1
                await asyncio.sleep(max(0.1, wait))


def _load_cache() -> dict[str, dict]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _CACHE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(_CACHE_PATH)


_RECLASSIFY_SYSTEM_5 = (
    "You RE-CLASSIFY already-extracted facts under a fixed 5-category scope "
    "taxonomy. You do not rewrite, extract, or invent new facts — you only "
    "decide, for each given statement AS WRITTEN, which single scope tag "
    "best fits it.\n\n"
    f"{SCOPE_TAGGING_PROMPT_RULES_5}\n\n"
    "For every atom in the input list, return exactly one result with the "
    "SAME id (copied exactly), the chosen scope, and a short reasoning (max "
    "~20 words) naming which positive/negative invariant decided it. Return "
    "one result per input atom, in any order — every id must appear exactly "
    "once."
)


class AtomReclassification5(BaseModel):
    id: str = Field(..., description="Copied exactly from the input atom id")
    new_scope: Literal["CONCEPT", "MECHANIC", "PRACTICE", "EDGE_CASE", "ANTI_PATTERN"]
    reasoning: str = Field(..., max_length=220)


class ReclassificationBatchResponse5(BaseModel):
    results: list[AtomReclassification5] = Field(default_factory=list)


@dataclass
class AtomRecord:
    id: str
    statement: str
    old_scope: str
    url: str
    article_title: str


def load_atoms_readonly() -> list[AtomRecord]:
    """SELECT-only: knowledge_atoms joined with document_summaries for title."""
    engine = create_engine(
        settings.POSTGRES_SQLALCHEMY_SYNC_DSN, connect_args={"connect_timeout": 5}
    )
    with engine.connect() as conn:
        atom_rows = conn.execute(
            text(
                """
                SELECT payload->>'id' AS id, payload->>'statement' AS statement,
                       payload->>'scope' AS scope, payload->>'url' AS url
                FROM knowledge_atoms
                """
            )
        ).fetchall()
        title_rows = conn.execute(
            text(
                "SELECT payload->>'url' AS url, payload->>'title' AS title FROM document_summaries"
            )
        ).fetchall()
    titles = {r.url: (r.title or "").strip() for r in title_rows if r.url}
    return [
        AtomRecord(
            id=r.id,
            statement=r.statement or "",
            old_scope=(r.scope or "").strip().upper(),
            url=r.url or "",
            article_title=titles.get(r.url) or (r.url or "")[:70],
        )
        for r in atom_rows
        if r.id and r.statement
    ]


def _batches(items: list[AtomRecord], size: int) -> list[list[AtomRecord]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def reclassify_batch(
    batch: list[AtomRecord],
    semaphore: asyncio.Semaphore,
    rpm_limiter: "_RpmLimiter",
    batch_idx: int,
) -> dict[str, AtomReclassification5]:
    payload_lines = "\n".join(f'- id="{a.id}": "{a.statement}"' for a in batch)
    user_payload = f"Atoms to re-classify ({len(batch)}):\n{payload_lines}"
    async with semaphore:
        for attempt, delay in enumerate([0, *_QUOTA_RETRY_DELAYS_SEC]):
            if delay:
                print(
                    f"  batch {batch_idx}: RPM-blocked, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{len(_QUOTA_RETRY_DELAYS_SEC) + 1})..."
                )
                await asyncio.sleep(delay)
            await rpm_limiter.acquire()
            try:
                response = await asyncio.to_thread(
                    run_gemini_structured_with_chain,
                    GEMINI_LITE_MODEL,
                    _RECLASSIFY_SYSTEM_5,
                    user_payload,
                    "",
                    ReclassificationBatchResponse5,
                    "audit_reclassify_atoms_5_scopes",
                )
                return {r.id: r for r in response.results}
            except GeminiQuotaExhaustedError as exc:
                print(f"  batch {batch_idx} of {len(batch)}: quota block ({exc})")
                continue
            except (
                Exception
            ) as exc:  # noqa: BLE001 — one failed batch must not abort the run
                print(
                    f"  batch {batch_idx} of {len(batch)} FAILED: {type(exc).__name__}: {exc}"
                )
                return {}
        print(
            f"  batch {batch_idx} of {len(batch)}: giving up after repeated RPM blocks"
        )
        return {}


async def _amain() -> None:
    atoms = load_atoms_readonly()
    if not atoms:
        print("No knowledge_atoms found — is Postgres reachable?", file=sys.stderr)
        raise SystemExit(1)

    baseline_dist = Counter(a.old_scope for a in atoms)
    print(
        f"Loaded {len(atoms)} real atoms (read-only). Baseline (3-scope): {dict(baseline_dist)}"
    )

    cache = _load_cache()
    by_id: dict[str, AtomReclassification5] = {
        aid: AtomReclassification5.model_validate(rec) for aid, rec in cache.items()
    }
    pending = [a for a in atoms if a.id not in by_id]
    print(
        f"Resumable cache: {len(by_id)} atoms already classified, "
        f"{len(pending)} pending this run."
    )

    if pending:
        batches = _batches(pending, _BATCH_SIZE)
        semaphore = asyncio.Semaphore(_CONCURRENCY)
        rpm_limiter = _RpmLimiter(_RPM_BUDGET)
        print(
            f"Dispatching {len(batches)} batches (size={_BATCH_SIZE}, "
            f"concurrency={_CONCURRENCY}, rpm_budget={_RPM_BUDGET})..."
        )

        cache_lock = asyncio.Lock()

        async def _run_and_cache(batch: list[AtomRecord], idx: int) -> None:
            result = await reclassify_batch(batch, semaphore, rpm_limiter, idx)
            if not result:
                return
            async with cache_lock:
                by_id.update(result)
                for aid, rec in result.items():
                    cache[aid] = rec.model_dump()
                _save_cache(cache)

        await asyncio.gather(
            *[_run_and_cache(b, i) for i, b in enumerate(batches, start=1)]
        )

    new_dist: Counter = Counter()
    transitions: Counter = Counter()
    examples_by_transition: dict[tuple[str, str], list[tuple[AtomRecord, str]]] = (
        defaultdict(list)
    )
    edge_case_examples: list[tuple[AtomRecord, str]] = []
    anti_pattern_examples: list[tuple[AtomRecord, str]] = []
    missing = 0
    for atom in atoms:
        result = by_id.get(atom.id)
        if result is None:
            missing += 1
            new_dist[
                atom.old_scope
            ] += 1  # unclassified -> keep old for the distribution
            continue
        new_dist[result.new_scope] += 1
        if result.new_scope != atom.old_scope:
            transitions[(atom.old_scope, result.new_scope)] += 1
            examples_by_transition[(atom.old_scope, result.new_scope)].append(
                (atom, result.reasoning)
            )
        if result.new_scope == "EDGE_CASE":
            edge_case_examples.append((atom, result.reasoning))
        elif result.new_scope == "ANTI_PATTERN":
            anti_pattern_examples.append((atom, result.reasoning))

    total = len(atoms)

    def _pct(n: int) -> str:
        return f"{100.0 * n / total:.1f}%"

    lines: list[str] = []
    lines.append(
        "Knowledge Atoms 5-Scope Reclassification Audit (DRY-RUN, no DB writes)"
    )
    lines.append(f"Total atoms: {total} | unclassified (LLM call failed): {missing}")
    lines.append("")
    lines.append("=== New distribution (5 categories) ===")
    for scope in _NEW_SCOPES:
        n = new_dist.get(scope, 0)
        lines.append(f"{scope}: {n} ({_pct(n)})")
    lines.append("")
    lines.append("=== Transition Matrix: 3 Scopes -> 5 Scopes ===")
    if not transitions:
        lines.append("(no transitions recorded)")
    for (old, new), count in sorted(transitions.items(), key=lambda kv: -kv[1]):
        lines.append(f"{old} -> {new}: {count} atoms")
    lines.append("")
    lines.append(f"=== EDGE_CASE examples (up to 7 of {len(edge_case_examples)}) ===")
    for atom, reasoning in edge_case_examples[:7]:
        lines.append(
            f'"{atom.statement[:160]}"\n'
            f"  Article: {atom.article_title[:70]}\n"
            f"  {atom.old_scope} -> EDGE_CASE | Reason: {reasoning}\n"
        )
    lines.append(
        f"=== ANTI_PATTERN examples (up to 7 of {len(anti_pattern_examples)}) ==="
    )
    for atom, reasoning in anti_pattern_examples[:7]:
        lines.append(
            f'"{atom.statement[:160]}"\n'
            f"  Article: {atom.article_title[:70]}\n"
            f"  {atom.old_scope} -> ANTI_PATTERN | Reason: {reasoning}\n"
        )

    report = "\n".join(lines)
    _OUTPUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote {_OUTPUT_PATH}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
