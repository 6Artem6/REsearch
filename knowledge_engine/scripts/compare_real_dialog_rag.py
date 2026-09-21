"""Local side-by-side (A/B) RAG comparison on REAL user replicas.

Extracts real user messages from real tutor dialog sessions and reruns the
knowledge_atoms RAG retrieval through two modes, writing a comparison report
to ./rag_real_data_comparison.log (repo root).

--- Data source note (read before trusting the numbers) ---
The task asked to pull real user messages from Postgres. Verified directly:
LangGraph's checkpoints table (AsyncPostgresSaver, real schema — see
tutor_graph_service.py) does NOT persist the raw ``request.user_msg`` input
as a channel — only internal graph state (route, intent, tutor_message,
anchor, ...). The ``.runs/node_deep_dive_sessions.json`` file also does not
hold it (it caches generated dense-lecture content per node, not per-turn
chat text). The only place the real, raw text of what a user actually typed
survives is ``logs/session_traces/**/turn_*_sub_concept_gap_exchange.md``
(per-LLM-call prompt/response traces, written by the real gap-evaluator
call for a real tutor turn) — confirmed against real thread_ids that also
appear in Postgres (node_deep_dive:<curriculum>:<node>), filtering OUT
anything whose GLOBAL ANCHOR starts with "test_" (pytest fixtures share the
same trace format and directory).

Retrieval itself (knowledge_atoms search + rerank) DOES hit the real,
backfilled Postgres knowledge_atoms table (703/703 scored) — that half is
exactly "из базы данных Postgres" as asked.

--- Mode A vs Mode B implementation note ---
Rather than mutating the process-wide settings.DIALOG_ATOMS_*_ENABLED flags
(which retrieve_dialog_knowledge_atoms_detailed reads at call time, but
whose per-atom scores/weights it does not expose — it only returns a
formatted text block) this script calls the underlying reusable primitives
directly, so every atom's individual score/W_ped/W_nov can be logged:
- Mode A: VectorStore.search_knowledge_atoms (plain ANN, current default).
- Mode B: same candidate pool -> _hybrid_rerank_pairs_async's Cross-Encoder
  query score -> pedagogical_reranker.apply_pedagogical_weights (pedagogical
  supersedes plain hybrid when both flags are on and a context is supplied,
  matching dialog_atoms_rag.py's actual branch order).
Both modes exercise the exact same production functions Mode A/B would call
in prod — nothing here is reimplemented, only orchestrated for visibility.

Turns run sequentially (not concurrently) since this is a one-shot local
report, not a load test — but the Cross-Encoder call is still wrapped in an
asyncio.Lock per the task's explicit instruction, since
dialog_atoms_rag.py's own hybrid/pedagogical rerank functions do not
serialize CE calls themselves (a real, separately-flagged production risk —
see this session's findings on rag_gateway/cross_encoder.py's non-thread-
safe singleton).

Usage: python knowledge_engine/scripts/compare_real_dialog_rag.py
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from knowledge_engine.src.domains.grounding.pedagogical_reranker import (
    apply_pedagogical_weights,
    calculate_novelty_weight,
    calculate_pedagogy_weight,
)
from knowledge_engine.src.domains.grounding.schemas import RetrievalPedagogicalContext
from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs
from knowledge_engine.src.shared.extraction import KnowledgeAtom, coerce_scope_type
from knowledge_engine.src.shared.vector_store import VectorStore

_TRACES_DIR = Path(REPO_ROOT) / "logs" / "session_traces"
_OUTPUT_PATH = Path(REPO_ROOT) / "rag_real_data_comparison.log"
_DEPTH_LADDER = ["intro", "deep_dive", "practice", "advanced", "expert"]
_TOP_N = 5
_FETCH_POOL = 30
_ALPHA = 0.7

# CRITICAL: dialog_atoms_rag.py's own _hybrid_rerank_pairs_async /
# _pedagogical_rerank_pairs_async do NOT serialize concurrent
# score_relevance_pairs calls on the shared Cross-Encoder singleton — a
# real deadlock was observed under concurrency earlier in this project's
# history (see backfill_core_relevance_score.py). Turns here run
# sequentially, but the lock is kept per the task's explicit instruction.
_CE_LOCK = asyncio.Lock()


@dataclass
class RealTurn:
    anchor: str
    node_title: str
    user_message: str
    trace_path: Path
    mtime: float
    depth_level: str = "intro"


_ANCHOR_RE = re.compile(r"GLOBAL ANCHOR[^\n]*\n(?P<anchor>node_deep_dive:[^\n]+)", re.M)
_NODE_TITLE_RE = re.compile(r"^### node_title\n(?P<val>.+)$", re.M)
_USER_MESSAGE_RE = re.compile(
    r"^### user_message\n(?P<val>.+?)(?=\n```|\n### |\n---|\Z)", re.M | re.S
)


def _extract_turn(path: Path) -> RealTurn | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m_anchor = _ANCHOR_RE.search(text)
    if not m_anchor:
        return None
    anchor = m_anchor.group("anchor").strip()
    if not anchor.startswith("node_deep_dive:") or ":test" in anchor.lower():
        return None
    # Fixture/pytest traces reuse this exact directory+format — the
    # anchor's curriculum segment for those is a "test_..." token.
    curriculum_part = anchor.split(":")[1] if anchor.count(":") >= 1 else ""
    if curriculum_part.startswith("test") or curriculum_part.startswith("demo"):
        return None
    m_title = _NODE_TITLE_RE.search(text)
    m_user = _USER_MESSAGE_RE.search(text)
    if not m_user:
        return None
    user_message = m_user.group("val").strip()
    if len(user_message) < 20:
        return None
    return RealTurn(
        anchor=anchor,
        node_title=(m_title.group("val").strip() if m_title else anchor),
        user_message=user_message,
        trace_path=path,
        mtime=path.stat().st_mtime,
    )


def collect_real_turns(max_turns: int = 10) -> list[RealTurn]:
    candidates: list[RealTurn] = []
    seen_messages: set[str] = set()
    for path in _TRACES_DIR.rglob("turn_*_sub_concept_gap_exchange.md"):
        turn = _extract_turn(path)
        if turn is None:
            continue
        key = turn.user_message[:200]
        if key in seen_messages:
            continue
        seen_messages.add(key)
        candidates.append(turn)

    candidates.sort(key=lambda t: t.mtime, reverse=True)
    selected = candidates[:max_turns]

    # Simulate depth progression within each real session (anchor), in the
    # session's own chronological order — per the task's explicit fallback
    # instruction (no real depth_level is persisted anywhere we could find).
    by_anchor: dict[str, list[RealTurn]] = {}
    for t in selected:
        by_anchor.setdefault(t.anchor, []).append(t)
    for turns in by_anchor.values():
        turns.sort(key=lambda t: t.mtime)
        for i, t in enumerate(turns):
            t.depth_level = _DEPTH_LADDER[min(i, len(_DEPTH_LADDER) - 1)]

    selected.sort(key=lambda t: t.mtime, reverse=True)
    return selected


@dataclass
class ScoredAtom:
    atom: KnowledgeAtom
    row: dict[str, Any]
    final_score: float
    w_ped: float | None = None
    w_nov: float | None = None


@dataclass
class ModeResult:
    label: str
    latency_ms: float
    top5: list[ScoredAtom] = field(default_factory=list)


def _row_to_atom(row: dict[str, Any]) -> KnowledgeAtom:
    return KnowledgeAtom(
        scope=coerce_scope_type(row.get("scope")),
        statement=str(row.get("statement") or "").strip()[:2000],
        source_chunk_ids=[],
        core_relevance_score=row.get("core_relevance_score"),
    )


async def run_mode_a_baseline(store: VectorStore, query: str) -> ModeResult:
    """Current default: plain ANN vector search, no rerank."""
    t0 = time.perf_counter()
    rows = await store.search_knowledge_atoms(query, limit=_FETCH_POOL)
    top5 = [
        ScoredAtom(
            atom=_row_to_atom(row), row=row, final_score=float(row.get("_score") or 0.0)
        )
        for row in rows[:_TOP_N]
    ]
    latency_ms = (time.perf_counter() - t0) * 1000
    return ModeResult(
        label="MODE A: BASELINE (Current Prod)", latency_ms=latency_ms, top5=top5
    )


async def run_mode_b_pedagogical(
    store: VectorStore, query: str, context: RetrievalPedagogicalContext
) -> ModeResult:
    """Hybrid + Pedagogical rerank — same candidate pool as Mode A, reranked
    via the real production formula (anchor_relevance.py's hybrid term,
    then pedagogical_reranker.py's depth/novelty weights on top)."""
    t0 = time.perf_counter()
    rows = await store.search_knowledge_atoms(query, limit=_FETCH_POOL)
    atoms = [_row_to_atom(row) for row in rows]
    statements = [a.statement for a in atoms]

    async with _CE_LOCK:
        query_scores = await asyncio.to_thread(score_relevance_pairs, query, statements)

    scored_atoms = list(zip(atoms, query_scores))
    ranked = apply_pedagogical_weights(scored_atoms, context, alpha=_ALPHA)

    atom_to_row = {id(a): r for a, r in zip(atoms, rows)}
    top5: list[ScoredAtom] = []
    for atom, final_score in ranked[:_TOP_N]:
        top5.append(
            ScoredAtom(
                atom=atom,
                row=atom_to_row[id(atom)],
                final_score=final_score,
                w_ped=calculate_pedagogy_weight(atom.scope, context.depth_level),
                w_nov=calculate_novelty_weight(atom, context),
            )
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    return ModeResult(
        label="MODE B: PEDAGOGICAL + HYBRID", latency_ms=latency_ms, top5=top5
    )


def _article_label(row: dict[str, Any]) -> str:
    url = str(row.get("url") or "").strip()
    doc_id = str(row.get("doc_id") or "").strip()
    if url:
        return url[:70]
    return doc_id or "unknown"


def _scope_counts(atoms: list[ScoredAtom]) -> dict[str, int]:
    counts = {
        "CONCEPT": 0,
        "MECHANIC": 0,
        "PRACTICE": 0,
        "EDGE_CASE": 0,
        "ANTI_PATTERN": 0,
    }
    for sa in atoms:
        counts[sa.atom.scope.value] = counts.get(sa.atom.scope.value, 0) + 1
    return counts


def _jaccard(a: list[ScoredAtom], b: list[ScoredAtom]) -> float:
    ids_a = {sa.row.get("_id") or sa.row.get("id") for sa in a}
    ids_b = {sa.row.get("_id") or sa.row.get("id") for sa in b}
    if not ids_a and not ids_b:
        return 100.0
    union = ids_a | ids_b
    if not union:
        return 0.0
    return 100.0 * len(ids_a & ids_b) / len(union)


def _format_mode_block(result: ModeResult) -> str:
    lines = [
        f"[{result.label}]",
        f"Latency: {result.latency_ms:.1f} ms",
        "Top-5 Atoms:",
    ]
    if not result.top5:
        lines.append("  (no atoms retrieved)")
    for i, sa in enumerate(result.top5, start=1):
        weight_tag = ""
        if sa.w_ped is not None and sa.w_nov is not None:
            weight_tag = f" [W_ped: {sa.w_ped:.2f}, W_nov: {sa.w_nov:.2f}]"
        lines.append(
            f"  {i}. [Score: {sa.final_score:.3f}]{weight_tag} "
            f"[Scope: {sa.atom.scope.value}] [Article: {_article_label(sa.row)}]"
        )
        lines.append(f'     Text: "{sa.atom.statement[:140]}"')
    return "\n".join(lines)


async def compare_one_turn(
    store: VectorStore,
    turn: RealTurn,
    index: int,
    context: RetrievalPedagogicalContext,
) -> tuple[str, ModeResult, ModeResult]:
    mode_a = await run_mode_a_baseline(store, turn.user_message)
    mode_b = await run_mode_b_pedagogical(store, turn.user_message, context)

    counts_a = _scope_counts(mode_a.top5)
    counts_b = _scope_counts(mode_b.top5)
    articles_a = {_article_label(sa.row) for sa in mode_a.top5}
    articles_b = {_article_label(sa.row) for sa in mode_b.top5}

    block = "\n".join(
        [
            "=" * 80,
            f'TURN [{index}] | QUERY: "{turn.user_message[:200]}"',
            f"CONTEXT: depth_level={context.depth_level} | subtopic={turn.node_title}",
            "=" * 80,
            "",
            _format_mode_block(mode_a),
            "",
            _format_mode_block(mode_b),
            "",
            "[DIVERGENCE METRICS]",
            f"- Jaccard Similarity (Top-5 overlap): {_jaccard(mode_a.top5, mode_b.top5):.0f}%",
            "- Scope Shift: Baseline (C:{c_a}/M:{m_a}/P:{p_a}/E:{e_a}/A:{a_a}) vs "
            "Pedagogical (C:{c_b}/M:{m_b}/P:{p_b}/E:{e_b}/A:{a_b})".format(
                c_a=counts_a["CONCEPT"],
                m_a=counts_a["MECHANIC"],
                p_a=counts_a["PRACTICE"],
                e_a=counts_a["EDGE_CASE"],
                a_a=counts_a["ANTI_PATTERN"],
                c_b=counts_b["CONCEPT"],
                m_b=counts_b["MECHANIC"],
                p_b=counts_b["PRACTICE"],
                e_b=counts_b["EDGE_CASE"],
                a_b=counts_b["ANTI_PATTERN"],
            ),
            f"- Article Diversity: Baseline ({len(articles_a)} articles) vs "
            f"Pedagogical ({len(articles_b)} articles)",
            "-" * 80,
            "",
        ]
    )
    return block, mode_a, mode_b


async def _amain() -> None:
    turns = collect_real_turns(max_turns=10)
    if not turns:
        print(
            "No real turns found under logs/session_traces/ — aborting.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    store = VectorStore()
    # recently_used_* accumulates per real anchor (session), matching the
    # task's ask to build a real per-session RetrievalPedagogicalContext.
    context_by_anchor: dict[str, RetrievalPedagogicalContext] = {}

    blocks: list[str] = []
    latencies_a: list[float] = []
    latencies_b: list[float] = []
    all_jaccards: list[float] = []

    for i, turn in enumerate(turns, start=1):
        ctx = context_by_anchor.get(turn.anchor)
        if ctx is None:
            ctx = RetrievalPedagogicalContext(
                current_subtopic_id=turn.node_title, depth_level=turn.depth_level
            )
        else:
            ctx = ctx.model_copy(update={"depth_level": turn.depth_level})

        block, mode_a, mode_b = await compare_one_turn(store, turn, i, ctx)
        blocks.append(block)
        latencies_a.append(mode_a.latency_ms)
        latencies_b.append(mode_b.latency_ms)
        all_jaccards.append(_jaccard(mode_a.top5, mode_b.top5))

        updated_atom_ids = {
            str(sa.row.get("_id") or sa.row.get("id")) for sa in mode_b.top5
        }
        updated_article_ids = {_article_label(sa.row) for sa in mode_b.top5}
        context_by_anchor[turn.anchor] = ctx.model_copy(
            update={
                "recently_used_atom_ids": ctx.recently_used_atom_ids | updated_atom_ids,
                "recently_used_article_ids": ctx.recently_used_article_ids
                | updated_article_ids,
            }
        )
        print(f"[{i}/{len(turns)}] {turn.anchor} | depth={turn.depth_level} | done")

    avg_a = sum(latencies_a) / len(latencies_a)
    avg_b = sum(latencies_b) / len(latencies_b)
    avg_jaccard = sum(all_jaccards) / len(all_jaccards)
    header = (
        "RAG Real-Data Side-by-Side Comparison\n"
        f"Turns: {len(turns)} (real user replicas, logs/session_traces)\n"
        f"Avg Latency: Mode A={avg_a:.1f}ms | Mode B={avg_b:.1f}ms\n"
        f"Avg Top-5 Jaccard overlap: {avg_jaccard:.0f}%\n"
    )
    _OUTPUT_PATH.write_text(header + "\n" + "\n".join(blocks), encoding="utf-8")
    print(f"\nWrote {_OUTPUT_PATH} ({len(turns)} turns)")
    print(header)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
