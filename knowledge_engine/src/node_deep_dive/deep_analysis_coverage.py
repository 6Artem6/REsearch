"""Asterisk-question Deep Analysis coverage tracking — novelty without surface-repeat."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

_COVERAGE_CAP = 32
_CHUNK_CAP = 64
_DIGEST_CAP = 8
_DIGEST_LEN = 420

# Rotating angle tokens for Dynamic Query Mutation on repeat Asterisk-question calls.
_DEEP_ANALYSIS_QUERY_ANGLES: tuple[str, ...] = (
    "memory layout event-loop concurrency",
    "serialization state management edge cases",
    "failure modes race conditions backpressure",
    "contracts invariants ownership lifecycle",
    "trade-offs latency consistency availability",
    "resource limits queues buffering saturation",
)

_S_CITE_RE = re.compile(r"\[S(\d+)\]", re.I)
_R_CITE_RE = re.compile(r"\[R(\d+)\]", re.I)
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.M)


def atom_key(statement: str) -> str:
    """Stable short key for a knowledge-atom statement (exclude / coverage)."""
    norm = " ".join((statement or "").strip().lower().split())
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def deep_analysis_repeat_index(memory: SessionMemory) -> int:
    """How many prior Asterisk-question coverage signals exist (0 = first call)."""
    return max(
        len(memory.deep_analysis_used_chunk_ids or []),
        len(memory.deep_analysis_used_atom_keys or []),
        len(memory.deep_analysis_prior_digests or []),
    )


def mutate_deep_analysis_query(base_query: str, memory: SessionMemory) -> str:
    """
    Dynamic Query Mutation: on repeat Asterisk-question calls, bias the search vector toward
    uncovered angles instead of the sticky node-title + chip query.
    """
    base = (base_query or "").strip()
    n = deep_analysis_repeat_index(memory)
    if n <= 0:
        return base
    angle = _DEEP_ANALYSIS_QUERY_ANGLES[(n - 1) % len(_DEEP_ANALYSIS_QUERY_ANGLES)]
    if not base:
        return angle
    if angle.lower() in base.lower():
        return base
    return f"{base} {angle}".strip()


def deep_analysis_retrieval_knobs(memory: SessionMemory) -> dict[str, Any]:
    """MMR / noise knobs for repeat deep_analysis vector search."""
    n = deep_analysis_repeat_index(memory)
    if n <= 0:
        return {
            "lambda_mult": 1.0,
            "query_noise": 0.0,
            "stochastic_sample": False,
            "pool_mult": 3,
        }
    # Shift toward diversity; mild noise; sample from a wider MMR pool.
    return {
        "lambda_mult": max(0.35, 0.75 - 0.08 * min(n, 4)),
        "query_noise": min(0.08, 0.03 + 0.01 * n),
        "stochastic_sample": True,
        "pool_mult": 5,
    }


_S_NODE_TOKEN = "[s_node]"


def _mentions_s_node(text: str) -> bool:
    """Structured citation token check (not a word-stem dictionary)."""
    return _S_NODE_TOKEN in (text or "").lower()


def _is_edge_related_text(text: str) -> bool:
    """Semantic edge/bottleneck/trade-off match via startup-synced vector lexicon."""
    from knowledge_engine.src.node_deep_dive.edge_case_lexicon import (
        is_edge_related_thesis,
    )

    return is_edge_related_thesis(text)


def attraction_summary_from_rows(
    attraction: list[dict[str, Any]] | None,
    *,
    max_items: int = 4,
) -> str:
    """Compact human summary of FACT_ATTRACTION claims for exhausted-RAG directive."""
    claims: list[str] = []
    seen: set[str] = set()
    for row in attraction or []:
        claim = (row.get("claim") or "").strip()
        if not claim:
            # Fall back to fact_line tail after Focus:
            fl = (row.get("fact_line") or "").strip()
            if "Focus:" in fl:
                claim = fl.split("Focus:", 1)[-1].strip()
            else:
                claim = fl
        claim = claim[:100].strip()
        key = claim.lower()
        if not claim or key in seen:
            continue
        seen.add(key)
        claims.append(claim)
        if len(claims) >= max_items:
            break
    return "; ".join(claims) if claims else "open bottlenecks from FACT_ATTRACTION"


def format_rag_exhausted_directive(*, attraction_summary: str = "") -> str:
    """
    Dynamic instruction when no fresh [R*] atoms are available.

    Forbids isomorphic node/code-1 re-teach; forces Attraction stress-test.
    """
    focus = (attraction_summary or "").strip() or "open bottlenecks from FACT_ATTRACTION"
    return (
        "[RAG_STATUS: EXHAUSTED]\n"
        "Внимание: новые RAG-атомы отсутствуют. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО делать "
        "общий обзор или заново пересказывать базовые материалы ноды "
        "(включая code-1 / SubAgentOrchestrator как «как устроен оркестратор»).\n"
        f"Задание: Проведи глубокий стресс-тест и декомпозицию узких мест из "
        f"ПОЛЮСА ПРИТЯЖЕНИЯ ({focus}). Разбери конкретные крайние случаи "
        "(edge cases) и сбои.\n"
        "code-1 / node materials — ТОЛЬКО как объект поиска уязвимостей и пробелов, "
        "не как основное тело ответа. Секции строй динамически: "
        "Проблема/Узкое место → Edge cases → Trade-off.\n"
        "Цитаты: [S_node] / [code-N] допустимы; не изобретай [R*] / [S1] если их нет "
        "в payload. JSON references: [].\n"
    )


def format_citation_policy_block(
    *,
    registry_empty: bool,
    atoms_empty: bool,
) -> str:
    """Host coverage/citation waiver when whitelist + RAG are both empty."""
    if not registry_empty and not atoms_empty:
        return ""
    lines = ["[CITATION_POLICY]"]
    if registry_empty:
        lines.append(
            "SOURCE REGISTRY empty (no mapped_source_ids): [S1]/[S2] and non-empty "
            "JSON `references` requirements are INACTIVE this turn. "
            "Use [S_node] / [code-N] / [diagram-N]. references MUST be []."
        )
    if atoms_empty:
        lines.append(
            "No fresh [R*] atoms this turn: do not invent [R1]…; do not treat "
            "missing [R*] cites as a coverage error."
        )
    return "\n".join(lines) + "\n"


def _first_thesis_line(body: str) -> str:
    """First non-empty prose line (skip fences / headings) with inline cites kept."""
    for raw in (body or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if line[:2] in ("- ", "* ", "• "):
            line = line[2:].strip()
        if len(line) < 12:
            continue
        for sep in (". ", "! ", "? ", "… "):
            if sep in line:
                line = line.split(sep, 1)[0] + sep.strip()
                break
        s_ids = parse_cited_s_indices(line)
        r_ids = parse_cited_r_indices(line)
        if not s_ids and not r_ids:
            # Keep line as-is; cites may appear later in section.
            return line[:180]
        cite_txt = "".join(f"[S{n}]" for n in s_ids[:4]) + "".join(
            f"[R{n}]" for n in r_ids[:4]
        )
        if cite_txt and cite_txt not in line:
            line = f"{line} {cite_txt}".strip()
        return line[:180]
    return ""


def make_technical_digest(
    technical_explanation: str,
    *,
    max_len: int = _DIGEST_LEN,
    rag_exhausted: bool = False,
) -> str:
    """
    Compact thesis digest of an Asterisk-question technical_explanation.

    Prefer heading + first thesis sentence + cited ids (not heading-only).
    When ``rag_exhausted``, prioritize concrete edge-case / bottleneck theses
    so the next asterisk question turn sees which failure mode was already covered.
    """
    text = (technical_explanation or "").strip()
    if not text:
        return ""
    parts: list[str] = []
    edge_parts: list[str] = []
    # Split by markdown headings while keeping heading titles.
    chunks = re.split(r"(?=^#{1,3}\s+)", text, flags=re.M)
    for chunk in chunks:
        chunk = (chunk or "").strip()
        if not chunk:
            continue
        hm = _HEADING_RE.match(chunk)
        if hm:
            title = hm.group(1).strip()[:72]
            body = chunk[hm.end() :].strip()
            thesis = _first_thesis_line(body)
            entry = f"{title}: {thesis}" if thesis else title
            parts.append(entry)
            probe = f"{entry}\n{body[:400]}"
            if _is_edge_related_text(probe):
                edge_parts.append(entry)
        else:
            thesis = _first_thesis_line(chunk)
            if thesis:
                parts.append(thesis)
                if _is_edge_related_text(thesis):
                    edge_parts.append(thesis)
        if len(parts) >= 6:
            break
    if rag_exhausted and edge_parts:
        # Lead with concrete edge cases already analyzed.
        ordered = list(dict.fromkeys(edge_parts + parts))
        digest = "EDGE_CASES_COVERED: " + " || ".join(ordered[:5])
    elif not parts:
        s_ids = parse_cited_s_indices(text)
        r_ids = parse_cited_r_indices(text)
        cite = "".join(f"[S{n}]" for n in s_ids[:4]) + "".join(
            f"[R{n}]" for n in r_ids[:4]
        )
        if _mentions_s_node(text):
            cite = (cite + "[S_node]").strip()
        prose = " ".join(text.split())[:200]
        digest = f"{prose} {cite}".strip()
    else:
        digest = " || ".join(parts[:5])
    # Always surface a cite summary if present in the full text.
    s_all = parse_cited_s_indices(text)
    r_all = parse_cited_r_indices(text)
    cite_tail_bits: list[str] = []
    if s_all:
        cite_tail_bits.append("".join(f"[S{n}]" for n in s_all[:6]))
    if r_all:
        cite_tail_bits.append("".join(f"[R{n}]" for n in r_all[:6]))
    if _mentions_s_node(text):
        cite_tail_bits.append("[S_node]")
    if cite_tail_bits:
        cite_tail = "cites:" + "".join(cite_tail_bits)
        if cite_tail not in digest:
            digest = f"{digest} | {cite_tail}"
    if rag_exhausted and not digest.startswith("EDGE_CASES_COVERED:"):
        digest = f"EDGE_CASES_COVERED: {digest}"
    return digest[:max_len]


def format_thesis_digests_block(memory: SessionMemory) -> str:
    """Dynamic thesis digest list for Asterisk-question user payload (anti-paraphrase)."""
    digests = [d for d in (memory.deep_analysis_prior_digests or []) if d][-4:]
    if not digests:
        return (
            "[PRIOR_ASTERISK_QUESTION_THESIS_DIGESTS]\n"
            "(none yet — first deep_analysis turn)\n"
        )
    lines = "\n".join(f"- {d}" for d in digests)
    return (
        "[PRIOR_ASTERISK_QUESTION_THESIS_DIGESTS]\n"
        f"{lines}\n"
        "Do NOT paraphrase or restate these theses. Deepen uncovered "
        "FACT_ATTRACTION angles (new edge cases / trade-offs). "
        "If [R*] are present, prefer fresh atoms; if RAG is exhausted, "
        "stress-test Attraction without re-teaching node basics.\n"
    )


def compact_assistant_turn_for_api_history(
    technical_explanation: str,
    *,
    follow_up_question: str = "",
    references: list[Any] | None = None,
) -> str:
    """
    Compact model turn for Gemini api_turns after Asterisk-question generation.

    Preserves thesis digest + cited ids + short follow-up stub (not longreads).
    """
    digest = make_technical_digest(technical_explanation)
    cite_ids: list[str] = []
    for n in parse_cited_s_indices(technical_explanation or ""):
        cite_ids.append(f"S{n}")
    for n in parse_cited_r_indices(technical_explanation or ""):
        cite_ids.append(f"R{n}")
    for ref in references or []:
        if isinstance(ref, dict):
            asset = str(ref.get("asset_id") or ref.get("id") or "").strip()
        else:
            asset = str(
                getattr(ref, "asset_id", "") or getattr(ref, "id", "") or ""
            ).strip()
        if asset:
            cite_ids.append(asset)
    cite_txt = ", ".join(dict.fromkeys(cite_ids)) if cite_ids else "(none)"
    fu = " ".join((follow_up_question or "").split())[:220]
    parts = [
        "[DEEP_ANALYSIS_TURN_DIGEST]",
        f"theses: {digest or '(empty)'}",
        f"cited: {cite_txt}",
    ]
    if fu:
        parts.append(f"follow_up: {fu}")
    return "\n".join(parts)[:2000]


def parse_cited_s_indices(text: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for m in _S_CITE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen and n >= 1:
            seen.add(n)
            out.append(n)
    return out


def parse_cited_r_indices(text: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for m in _R_CITE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen and n >= 1:
            seen.add(n)
            out.append(n)
    return out


def _append_unique(dst: list[str], items: list[str], *, cap: int) -> list[str]:
    seen = {x for x in dst if x}
    for raw in items:
        key = (raw or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dst.append(key)
        if len(dst) >= cap:
            break
    return dst[-cap:]


def format_rag_coverage_state_block(
    memory: SessionMemory,
    *,
    rag_exhausted: bool,
) -> str:
    """
    Dynamic [RAG_COVERAGE_STATE] for isolated deep_analysis turns.

    Affirmative novelty rules: no surface repeat; Why/How/Mechanics deepen OK.
    """
    sources = [s for s in (memory.deep_analysis_used_source_ids or []) if s][-16:]
    digests = [d for d in (memory.deep_analysis_prior_digests or []) if d][-4:]
    src_txt = ", ".join(sources) if sources else "(none yet)"
    dig_lines = "\n".join(f"- {d}" for d in digests) if digests else "- (none yet)"
    exhausted = "true" if rag_exhausted else "false"
    exhausted_rule = (
        "4. IF RAG Exhausted is True: Do NOT invent fake theory. Do NOT restate "
        "the node overview or re-implement code-1. Focus 100% on FACT_ATTRACTION "
        "stress-tests / edge cases not listed in Prior thesis digests "
        "(see also [RAG_STATUS: EXHAUSTED] if present).\n"
    )
    return (
        "[RAG_COVERAGE_STATE]\n"
        f"Prior Asterisk-question analyzed source concepts: {src_txt}\n"
        f"Prior Asterisk-question thesis digests (do not paraphrase):\n{dig_lines}\n"
        f"RAG Exhausted: {exhausted}\n\n"
        "INSTRUCTIONS:\n"
        "1. NO SURFACE REPEAT: Do NOT repeat surface-level summaries or high-level "
        "overviews listed in Prior thesis digests.\n"
        "2. WHY / HOW / MECHANICS DEEP DIVE: You MAY reference previously used "
        "chunks/sources ONLY IF you go significantly deeper into the underlying "
        "Why / How / Mechanics (architectural motivation & constraints; protocols "
        "& contracts; event-loop / memory / edge failure modes) that were NOT "
        "covered in prior iterations.\n"
        "3. IF RAG Exhausted is False: Prioritize unseen aspects and sources "
        "[S*]/[R*] for the explanation.\n"
        f"{exhausted_rule}"
    )


def record_deep_analysis_coverage(
    memory: SessionMemory,
    *,
    technical_explanation: str,
    feedback_on_answer: str = "",
    follow_up_question: str = "",
    references: list[Any] | None = None,
    rag_exhausted: bool | None = None,
) -> None:
    """
    After an Asterisk-question turn: merge cited [Sx]/[Rx] and a digest into SessionMemory.

    [Rx] maps via ``memory.last_deep_analysis_atom_keys`` (R1 = index 0).
    [Sx] recorded as ``S{n}`` plus any ``asset_id`` from references.
    ``[S_node]`` is recorded as coverage signal when registry was empty.
    """
    blob = "\n".join(
        [
            technical_explanation or "",
            feedback_on_answer or "",
            follow_up_question or "",
        ]
    )
    source_ids: list[str] = [f"S{n}" for n in parse_cited_s_indices(blob)]
    if _mentions_s_node(blob):
        source_ids.append("S_node")
    for ref in references or []:
        asset = ""
        if isinstance(ref, dict):
            asset = str(ref.get("asset_id") or ref.get("id") or "").strip()
        else:
            asset = str(
                getattr(ref, "asset_id", "") or getattr(ref, "id", "") or ""
            ).strip()
        if asset:
            source_ids.append(asset)

    turn_keys = list(memory.last_deep_analysis_atom_keys or [])
    atom_keys: list[str] = []
    for r_idx in parse_cited_r_indices(blob):
        pos = r_idx - 1
        if 0 <= pos < len(turn_keys) and turn_keys[pos]:
            atom_keys.append(turn_keys[pos])
    # Hard exclude on next Asterisk question: record all atoms/chunks shown this turn.
    atom_keys.extend(turn_keys)
    chunk_ids = list(memory.last_deep_analysis_chunk_ids or [])
    atom_ids = list(memory.last_deep_analysis_atom_ids or [])
    # Atom row ids also act as exclude keys when source_chunk_ids were empty.
    chunk_ids.extend(atom_ids)

    memory.deep_analysis_used_source_ids = _append_unique(
        list(memory.deep_analysis_used_source_ids or []),
        source_ids,
        cap=_COVERAGE_CAP,
    )
    memory.deep_analysis_used_atom_keys = _append_unique(
        list(memory.deep_analysis_used_atom_keys or []),
        atom_keys,
        cap=_COVERAGE_CAP,
    )
    memory.deep_analysis_used_chunk_ids = _append_unique(
        list(memory.deep_analysis_used_chunk_ids or []),
        chunk_ids,
        cap=_CHUNK_CAP,
    )
    exhausted = (
        bool(rag_exhausted)
        if rag_exhausted is not None
        else not bool(turn_keys)
    )
    digest = make_technical_digest(
        technical_explanation, rag_exhausted=exhausted
    )
    if digest:
        digests = list(memory.deep_analysis_prior_digests or [])
        # Avoid adjacent duplicate digests from identical Asterisk-question outputs.
        if not digests or digests[-1] != digest:
            digests.append(digest)
        memory.deep_analysis_prior_digests = digests[-_DIGEST_CAP:]


def citations_required_for_turn(
    *,
    registry_empty: bool,
    atoms_empty: bool,
) -> dict[str, bool]:
    """
    Host coverage checker flags for deep_analysis.

    When whitelist + RAG are empty, missing [S*]/references is NOT an error.
    """
    return {
        "require_s_registry_cites": not registry_empty,
        "require_references": not registry_empty,
        "require_r_cites": not atoms_empty,
        "allow_s_node": registry_empty,
        "citations_inactive": registry_empty and atoms_empty,
    }