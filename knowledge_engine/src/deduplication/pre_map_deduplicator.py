"""Pre-MAP Dedup: BGE clustering + Flash Lite bulk gate, before the expensive
Gemma MAP+REDUCE phase (blog_spatial_summarizer.py / map_reduce_jobs_pooled_async).

Protects the shared TPM budget by skipping duplicate MAP+REDUCE work for
near-identical sources, while never dropping a URL outright — every source
that turns out to be a duplicate is recorded as an ALIAS of a CANONICAL one,
keeping it citable for grounding without paying for a second MAP+REDUCE pass.

Pipeline (per deduplicate_before_map_reduce call, one curriculum-node batch):

1. Context Extraction (extract -> Triage (Group Batch) -> MMR/Top-K) —
   split TEXT vs CODE per candidate (detect_code_content, reused from
   pre_flight_triage.py, same 3-layer detector as Pre-Flight Triage).
   Triage itself is NOT a local heuristic: both TEXT paragraphs and CODE
   AST units are classified CORE/CONTEXT/DROP by the EXISTING Flash Lite
   structure pass (paper_structure_analyzer.py — the same Pass 1 used by
   the inbound ingest gate). ALL candidates in the batch are Triaged in
   ONE Flash Lite call (_flash_lite_triage_core_units_batch — Group
   Batching, TPM-guarded by PRE_MAP_DEDUP_TRIAGE_MAX_TPM, falls back to
   one call per candidate if the combined payload is oversized or the
   call fails). Only CORE units proceed.
   TEXT: CORE paragraphs then go through the unmodified, existing MMR
   (_mmr_top_by_centroid -> greedy_mmr_select) for a diverse Top-K
   centroid fingerprint, not just the first paragraphs.
   CODE: AST-extraction (_ast_semantic_extracts — signatures/docstrings/
   comments/shallow calls, no BGE call at all) produces top-level units;
   those go through the same batched Triage as text, then are capped to
   Top-K.
2. BGE Clustering (TEXT only, 0 LLM calls) — pool each candidate's Top-K
   fingerprint into one document vector, Union-Find over pairwise cosine >=
   PRE_MAP_DEDUP_COSINE_THRESHOLD into "suspect groups". A text candidate
   with no close neighbor is autonomous and never reaches Flash Lite.
3. Flash Lite comparison — TEXT and CODE are routed to two different
   comparison paths from here on:
   3a. Bulk Gate (Group Batching + TPM guard) — every suspect TEXT group
       (code no longer rides along) is packed into as FEW structured calls
       as possible: if the estimated token size exceeds
       PRE_MAP_DEDUP_BULK_GATE_MAX_TPM, the batch is greedily split into
       sequential sub-batches (a suspect group is never split across calls
       — its members must be compared together).
   3b. Isolated Code Deduplication (code_deduplicator.py) — ALL code
       candidates (code is never BGE-clustered — a cosine gate on AST
       signatures is not a reliable duplicate signal for "same algorithm,
       different language") go through a SEPARATE, project-context-enriched
       comparison: README (module + root, BGE-anchor-filtered) + a balanced
       directory-tree snippet + a self-contained Head-3/Tail-3 full-body AST
       extract, framed as "Behavioral Intent & Data Structure State", not
       flat AST signatures — see that module's docstring for why.
   Both paths return the same canonical_map: {canonical_id: [alias_id,
   ...]} shape; results are merged before sanitization.
4. Canonical Pooling — CANONICAL candidates are exactly the ones NOT listed
   as an alias anywhere in the (sanitized) canonical_map; everyone else is
   ALIAS(canonical_id). Fail-open throughout: any BGE/Lite error at any step
   just leaves the affected candidates CANONICAL — this module can never
   cause a source to be silently dropped from ingest.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from knowledge_engine.config import (
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
    PRE_MAP_DEDUP_BULK_GATE_MAX_TPM,
    PRE_MAP_DEDUP_COSINE_THRESHOLD,
    PRE_MAP_DEDUP_TOP_K,
    PRE_MAP_DEDUP_TRIAGE_MAX_TPM,
)
from knowledge_engine.schemas.llm_contracts.pre_map_dedup import CanonicalMapContract
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    estimate_text_tokens,
)
from knowledge_engine.src.curriculum.pre_flight_triage import (
    _ast_semantic_extracts,
    _bulk_gate_code_context,
    _code_paragraphs_from_raw_text,
    _cosine,
    _extract_paragraphs,
    detect_code_content,
    greedy_mmr_select,
)
from knowledge_engine.ui.run_log import trace

_BULK_GATE_SYSTEM = """
You are a Duplicate Content Auditor. You are given source candidates — text
"suspect groups" (already pre-clustered by BGE cosine similarity) and/or
source code files, each with an id and a short semantic extract (the
document's top paragraphs, or function/class signatures and docstrings).

Task: determine which candidates are DUPLICATES of each other — they
describe or implement EXACTLY THE SAME THING (the same text reworded; the
same algorithm in different programming languages; a mirror of the same
article/repository at different URLs). Candidates within one group are NOT
required to be duplicates of each other — compare their extracts on the
merits.

Do NOT treat candidates as duplicates just because they share a topic —
different content, different depth of coverage, or a different focus means
they are NOT duplicates.

For each group of genuine duplicates you find, pick ONE canonical_id — the
most complete/highest-quality one by its own extract — and list the rest as
its aliases. A candidate with no duplicates is simply left out of mappings.

Return ONLY JSON: {"mappings": [{"canonical_id": "...", "aliases": ["...", ...]}]}.
""".strip()


@dataclass
class PreMapCandidate:
    """One source before MAP+REDUCE. ``id`` is the stable key used
    throughout the result (the caller's own URL is the natural choice)."""

    id: str
    url: str
    text: str
    method: str = ""
    is_code: bool | None = None


@dataclass
class PreMapDedupDecision:
    id: str
    is_canonical: bool
    canonical_id: str
    is_code: bool
    context_units: list[str] = field(default_factory=list)


@dataclass
class PreMapDedupResult:
    decisions: dict[str, PreMapDedupDecision] = field(default_factory=dict)
    alias_map: dict[str, list[str]] = field(default_factory=dict)

    def canonical_ids(self) -> list[str]:
        return [d.id for d in self.decisions.values() if d.is_canonical]

    def is_alias(self, candidate_id: str) -> bool:
        d = self.decisions.get(candidate_id)
        return d is not None and not d.is_canonical

    def canonical_of(self, candidate_id: str) -> str:
        d = self.decisions.get(candidate_id)
        return d.canonical_id if d is not None else candidate_id


async def _flash_lite_triage_core_units(
    units: list[str], *, target_topic: str = "", label: str = ""
) -> list[str]:
    """Triage (Step 1a) — NOT a local heuristic: reuses the EXISTING Flash
    Lite structure pass (paper_structure_analyzer.py's PaperStructureAnalyzer,
    Pass 1 of the inbound ingest gate — same CORE/CONTEXT/DROP classification,
    same GEMINI_LITE_MODEL) to classify each unit and keep only CORE ones.
    Works identically for TEXT paragraphs and CODE AST units — both are just
    "paragraphs" to this classifier. Fail-open: any error, or an empty/all-
    non-CORE verdict, falls back to the full input list rather than handing
    the next step nothing to work with."""
    if not units:
        return []
    from knowledge_engine.src.parsers.paper_input_json import (
        build_input_paper_json_from_plain_text,
    )
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        PaperStructureAnalyzer,
    )
    from knowledge_engine.src.parsers.paper_structure_schema import ParagraphPriority

    input_paper = build_input_paper_json_from_plain_text("\n\n".join(units))
    id_to_unit = {
        p.paragraph_id: p.text for page in input_paper.pages for p in page.paragraphs
    }
    if not id_to_unit:
        return list(units)
    try:
        analysis = await asyncio.to_thread(
            PaperStructureAnalyzer().analyze,
            target_topic,
            input_paper,
            label=label or "pre_map_dedup_triage",
        )
    except Exception as exc:
        trace(f"PRE_MAP_DEDUP triage(flash_lite) ✗ | {type(exc).__name__}: {exc}")
        return list(units)
    core_texts = [
        id_to_unit[row.paragraph_id]
        for row in analysis.paragraphs
        if row.priority == ParagraphPriority.CORE and row.paragraph_id in id_to_unit
    ]
    trace(
        f"PRE_MAP_DEDUP triage(flash_lite) ✓ | CORE={len(core_texts)}/{len(id_to_unit)} "
        f"label={label or 'pre_map_dedup_triage'}"
    )
    return core_texts or list(units)


async def _triage_per_candidate(
    units_by_id: dict[str, list[str]], *, target_topic: str = "", label: str = ""
) -> dict[str, list[str]]:
    """Fallback path for _flash_lite_triage_core_units_batch: one Flash Lite
    call per candidate via the original single-candidate Triage function —
    exactly the pre-Group-Batching behavior, used only when the batched call
    itself is unavailable (oversized payload or a raised exception)."""
    out: dict[str, list[str]] = {}
    for cid, units in units_by_id.items():
        out[cid] = await _flash_lite_triage_core_units(
            units,
            target_topic=target_topic,
            label=f"{label}:{cid[:60]}" if label else cid[:60],
        )
    return out


async def _flash_lite_triage_core_units_batch(
    units_by_id: dict[str, list[str]],
    *,
    target_topic: str = "",
    label: str = "",
    max_tpm: int | None = None,
) -> dict[str, list[str]]:
    """Group Batching (Step 1a) — classifies ALL candidates' units in ONE
    Flash Lite call instead of one call per candidate: every candidate's raw
    units become one "page" of a single combined InputPaperJson (globally
    unique paragraph_id, per the schema's own "unique... across the full
    document" contract), submitted to the SAME PaperStructureAnalyzer pass
    _flash_lite_triage_core_units() uses for a single candidate, then routed
    back per candidate by paragraph_id. TPM-guarded: if the combined payload
    would exceed max_tpm, or the batched call raises, falls back to
    _triage_per_candidate() (one call per candidate, previous behavior) so a
    batching failure never loses Triage coverage. Fail-open per candidate,
    same guarantee as the single-candidate function: a candidate with zero
    surviving CORE units falls back to its own full unit list."""
    active_ids = [cid for cid, units in units_by_id.items() if units]
    if not active_ids:
        return {cid: [] for cid in units_by_id}

    from knowledge_engine.src.parsers.paper_input_json import (
        input_paper_json_for_llm,
    )
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        PaperStructureAnalyzer,
    )
    from knowledge_engine.src.parsers.paper_structure_schema import (
        InputPaperJson,
        InputPaperPage,
        InputPaperParagraph,
        ParagraphPriority,
    )

    pages: list[InputPaperPage] = []
    id_to_owner: dict[int, tuple[str, str]] = {}
    pid = 0
    for page_no, cid in enumerate(active_ids, start=1):
        paras: list[InputPaperParagraph] = []
        for unit in units_by_id[cid]:
            pid += 1
            paras.append(
                InputPaperParagraph(
                    paragraph_id=pid, section_title=cid[:120], text=unit[:8000]
                )
            )
            id_to_owner[pid] = (cid, unit)
        pages.append(InputPaperPage(page_number=page_no, paragraphs=paras))
    combined_paper = InputPaperJson(total_pages=len(pages), pages=pages)

    limit = max_tpm if max_tpm is not None else PRE_MAP_DEDUP_TRIAGE_MAX_TPM
    payload_tokens = estimate_text_tokens(
        json.dumps(input_paper_json_for_llm(combined_paper), ensure_ascii=False)
    )
    if payload_tokens > limit:
        trace(
            f"PRE_MAP_DEDUP triage(batch) skip | tokens≈{payload_tokens} > {limit} "
            f"— per-candidate fallback"
        )
        return await _triage_per_candidate(
            units_by_id, target_topic=target_topic, label=label
        )

    try:
        analysis = await asyncio.to_thread(
            PaperStructureAnalyzer().analyze,
            target_topic,
            combined_paper,
            label=label or "pre_map_dedup_triage_batch",
        )
    except Exception as exc:
        trace(
            f"PRE_MAP_DEDUP triage(batch) ✗ | {type(exc).__name__}: {exc} "
            f"— per-candidate fallback"
        )
        return await _triage_per_candidate(
            units_by_id, target_topic=target_topic, label=label
        )

    core_by_id: dict[str, list[str]] = {cid: [] for cid in active_ids}
    for row in analysis.paragraphs:
        owner = id_to_owner.get(row.paragraph_id)
        if owner is None or row.priority != ParagraphPriority.CORE:
            continue
        cid, unit_text = owner
        core_by_id[cid].append(unit_text)

    for cid in active_ids:
        if not core_by_id[cid]:
            core_by_id[cid] = list(units_by_id[cid])
    for cid, units in units_by_id.items():
        if not units:
            core_by_id[cid] = []

    trace(
        f"PRE_MAP_DEDUP triage(batch) ✓ | candidates={len(active_ids)} "
        f"units={len(id_to_owner)} tokens≈{payload_tokens} "
        f"label={label or 'pre_map_dedup_triage_batch'}"
    )
    return core_by_id


def _mmr_top_by_centroid(
    paragraphs: list[str], *, top_k: int, lambda_param: float = 0.65
) -> list[str]:
    """MMR (Step 1b) — unmodified: greedy-MMR-select a diverse Top-K from
    whatever paragraphs it is given (the caller is expected to have already
    run these through Triage — _flash_lite_triage_core_units). A fingerprint,
    not just the first K paragraphs. Fail-open: a BGE error falls back to
    the first-K paragraphs rather than raising (this candidate just gets a
    cruder fingerprint for clustering — it never crashes the whole batch)."""
    if not paragraphs:
        return []
    if len(paragraphs) <= top_k:
        return list(paragraphs)
    from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

    try:
        vecs = embed_texts_bge_m3(paragraphs)
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        scores = [_cosine(v, centroid) for v in vecs]
        idx = greedy_mmr_select(vecs, scores, top_k=top_k, lambda_param=lambda_param)
        return [paragraphs[i] for i in idx]
    except Exception as exc:
        trace(f"PRE_MAP_DEDUP mmr_fingerprint ✗ | {type(exc).__name__}: {exc}")
        return paragraphs[:top_k]


def _pool_vector(fingerprint: list[str]) -> list[float] | None:
    """Fail-open: a BGE error returns None (candidate skips clustering and
    stays autonomous/CANONICAL by construction) instead of raising."""
    if not fingerprint:
        return None
    from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

    try:
        vecs = embed_texts_bge_m3(fingerprint)
        dim = len(vecs[0])
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    except Exception as exc:
        trace(f"PRE_MAP_DEDUP pool_vector ✗ | {type(exc).__name__}: {exc}")
        return None


def _cluster_text_candidates(
    doc_vectors: dict[str, list[float]], *, threshold: float
) -> list[list[str]]:
    """Union-Find over pairwise cosine(doc_i, doc_j) >= threshold. A
    singleton group is an autonomous candidate — never sent to Flash Lite."""
    ids = list(doc_vectors.keys())
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if _cosine(doc_vectors[a], doc_vectors[b]) >= threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for cid in ids:
        groups.setdefault(find(cid), []).append(cid)
    return list(groups.values())


def _build_bulk_gate_payload(
    suspect_groups: list[list[str]],
    code_ids: list[str],
    context_by_id: dict[str, list[str]],
) -> str:
    text_section = [
        {
            "group_id": f"g{i}",
            "candidates": [
                {
                    "id": cid,
                    "extract": "\n---\n".join(context_by_id.get(cid, []))[:4000],
                }
                for cid in group
            ],
        }
        for i, group in enumerate(suspect_groups)
    ]
    code_section = [
        {"id": cid, "extract": "\n---\n".join(context_by_id.get(cid, []))[:4000]}
        for cid in code_ids
    ]
    payload = {"suspect_text_groups": text_section, "code_files": code_section}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _bulk_gate_unit_tokens(fragment: dict) -> int:
    return estimate_text_tokens(json.dumps(fragment, ensure_ascii=False))


def _pack_bulk_gate_sub_batches(
    suspect_groups: list[list[str]],
    code_ids: list[str],
    context_by_id: dict[str, list[str]],
    *,
    max_tokens: int,
) -> list[tuple[list[list[str]], list[str]]]:
    """Group Batching + TPM guard: greedily pack units (a WHOLE suspect
    group, or one code id — a group's members are never split across
    sub-batches, since they must be compared together) until the running
    token estimate would exceed max_tokens, then start a new sub-batch. A
    single oversized unit still gets its own sub-batch rather than being
    silently dropped or truncated."""
    units: list[tuple[str, object, int]] = []
    for i, group in enumerate(suspect_groups):
        fragment = {
            "group_id": f"g{i}",
            "candidates": [
                {
                    "id": cid,
                    "extract": "\n---\n".join(context_by_id.get(cid, []))[:4000],
                }
                for cid in group
            ],
        }
        units.append(("text", group, _bulk_gate_unit_tokens(fragment)))
    for cid in code_ids:
        fragment = {
            "id": cid,
            "extract": "\n---\n".join(context_by_id.get(cid, []))[:4000],
        }
        units.append(("code", cid, _bulk_gate_unit_tokens(fragment)))

    batches: list[tuple[list[list[str]], list[str]]] = []
    cur_groups: list[list[str]] = []
    cur_codes: list[str] = []
    cur_tokens = 0
    for kind, payload, tokens in units:
        if cur_tokens + tokens > max_tokens and (cur_groups or cur_codes):
            batches.append((cur_groups, cur_codes))
            cur_groups, cur_codes, cur_tokens = [], [], 0
        if kind == "text":
            cur_groups.append(payload)  # type: ignore[arg-type]
        else:
            cur_codes.append(payload)  # type: ignore[arg-type]
        cur_tokens += tokens
    if cur_groups or cur_codes:
        batches.append((cur_groups, cur_codes))
    return batches


async def _call_bulk_gate_once(
    suspect_groups: list[list[str]],
    code_ids: list[str],
    context_by_id: dict[str, list[str]],
    *,
    anchor: str,
    batch_label: str = "1/1",
) -> dict[str, list[str]]:
    from knowledge_engine.services.gemini_stateless import (
        run_gemini_structured_with_chain,
    )

    payload = _build_bulk_gate_payload(suspect_groups, code_ids, context_by_id)
    tokens = estimate_text_tokens(_BULK_GATE_SYSTEM) + estimate_text_tokens(payload)
    t0 = time.monotonic()
    try:
        result = await asyncio.to_thread(
            run_gemini_structured_with_chain,
            GEMINI_LITE_MODEL,
            _BULK_GATE_SYSTEM,
            payload,
            anchor or "pre_map_dedup",
            CanonicalMapContract,
            f"pre_map_dedup / bulk_gate[{batch_label}]",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        )
        elapsed = time.monotonic() - t0
        trace(
            f"PRE_MAP_DEDUP bulk_gate ✓ | batch={batch_label} groups={len(suspect_groups)} "
            f"code={len(code_ids)} tokens≈{tokens} latency={elapsed:.2f}s"
        )
        return result.to_dict()
    except Exception as exc:
        elapsed = time.monotonic() - t0
        trace(
            f"PRE_MAP_DEDUP bulk_gate ✗ | batch={batch_label} tokens≈{tokens} "
            f"latency={elapsed:.2f}s | {type(exc).__name__}: {exc}"
        )
        return {}


async def _run_bulk_gate(
    suspect_groups: list[list[str]],
    code_ids: list[str],
    context_by_id: dict[str, list[str]],
    *,
    anchor: str,
    max_tpm: int | None = None,
) -> dict[str, list[str]]:
    if not suspect_groups and not code_ids:
        return {}
    from knowledge_engine.services.gemini_stateless import is_gemini_available

    if not is_gemini_available():
        return {}

    limit = max_tpm if max_tpm is not None else PRE_MAP_DEDUP_BULK_GATE_MAX_TPM
    sub_batches = _pack_bulk_gate_sub_batches(
        suspect_groups, code_ids, context_by_id, max_tokens=limit
    )
    trace(
        f"PRE_MAP_DEDUP bulk_gate pack ✓ | units={len(suspect_groups) + len(code_ids)} "
        f"sub_batches={len(sub_batches)} max_tpm={limit}"
    )

    merged: dict[str, list[str]] = {}
    for i, (groups, codes) in enumerate(sub_batches):
        partial = await _call_bulk_gate_once(
            groups,
            codes,
            context_by_id,
            anchor=anchor,
            batch_label=f"{i + 1}/{len(sub_batches)}",
        )
        for canonical_id, aliases in partial.items():
            merged.setdefault(canonical_id, [])
            merged[canonical_id].extend(aliases)
    return merged


def _sanitize_canonical_map(
    raw_map: dict[str, list[str]], valid_ids: set[str]
) -> dict[str, list[str]]:
    """Drop hallucinated ids, self-aliases, and conflicts (an id claimed as
    an alias twice, or an id used as both a canonical and someone's alias) —
    keeps the assignment a clean forest, never a cycle or a double-claim."""
    assigned_as_alias: set[str] = set()
    canonical_ids_seen: set[str] = set()
    clean: dict[str, list[str]] = {}
    for canonical_id, aliases in raw_map.items():
        if canonical_id not in valid_ids or canonical_id in assigned_as_alias:
            continue
        kept: list[str] = []
        for alias_id in aliases:
            if alias_id not in valid_ids or alias_id == canonical_id:
                continue
            if alias_id in assigned_as_alias or alias_id in canonical_ids_seen:
                continue
            kept.append(alias_id)
            assigned_as_alias.add(alias_id)
        if kept:
            clean[canonical_id] = kept
            canonical_ids_seen.add(canonical_id)
    return clean


async def _run_step3a_bulk_gate(
    suspect_groups: list[list[str]],
    context_by_id: dict[str, list[str]],
    *,
    anchor: str,
) -> dict[str, list[str]]:
    """Step 3a as its own coroutine (fail-open) — gathered together with
    Step 3b in deduplicate_before_map_reduce() since text and code
    candidates are strictly independent."""
    if not suspect_groups:
        return {}
    trace(f"PRE_MAP_DEDUP step3a ▶ | suspect_groups={len(suspect_groups)}")
    t0 = time.monotonic()
    try:
        result = await _run_bulk_gate(suspect_groups, [], context_by_id, anchor=anchor)
    except Exception as exc:
        trace(
            f"PRE_MAP_DEDUP step3a ✗ | {time.monotonic() - t0:.2f}s | "
            f"{type(exc).__name__}: {exc}"
        )
        return {}
    trace(f"PRE_MAP_DEDUP step3a ✓ | {time.monotonic() - t0:.2f}s")
    return result


async def _run_step3b_code_dedup(
    code_ids: list[str],
    candidates: list[PreMapCandidate],
    *,
    anchor: str,
) -> dict[str, list[str]]:
    """Step 3b as its own coroutine (fail-open) — gathered together with
    Step 3a. Any error just contributes nothing; code candidates stay
    CANONICAL exactly as Step 3a would have left them."""
    if not code_ids:
        return {}
    from knowledge_engine.src.deduplication.code_deduplicator import (
        deduplicate_code_candidates,
    )

    trace(f"PRE_MAP_DEDUP step3b ▶ | code_candidates={len(code_ids)}")
    t0 = time.monotonic()
    code_id_set = set(code_ids)
    code_candidates = [c for c in candidates if c.id in code_id_set]
    try:
        result = await deduplicate_code_candidates(code_candidates, anchor=anchor)
    except Exception as exc:
        trace(f"PRE_MAP_DEDUP code_dedup ✗ | {type(exc).__name__}: {exc}")
        result = {}
    trace(f"PRE_MAP_DEDUP step3b ✓ | {time.monotonic() - t0:.2f}s")
    return result


async def deduplicate_before_map_reduce(
    candidates: list[PreMapCandidate],
    *,
    top_k: int | None = None,
    cosine_threshold: float | None = None,
    min_chars: int = 20,
    anchor: str = "",
) -> PreMapDedupResult:
    """Runs the full 4-step pipeline (see module docstring) over one batch of
    candidates and returns a CANONICAL/ALIAS decision per candidate id.
    Fail-open: BGE/Lite errors never remove a candidate — worst case,
    everyone stays CANONICAL and MAP+REDUCE runs for all of them, exactly as
    if this module were not in the call path."""
    result = PreMapDedupResult()
    if not candidates:
        return result

    k = top_k if top_k is not None else PRE_MAP_DEDUP_TOP_K
    threshold = (
        cosine_threshold
        if cosine_threshold is not None
        else PRE_MAP_DEDUP_COSINE_THRESHOLD
    )

    # Step 1a: извлечение сырых юнитов (разделение текст/код), без LLM-вызовов.
    # Fail-open по кандидату: упавший детектор/экстрактор/AST-парсер не должен
    # прервать весь батч — этот кандидат просто остаётся без сырых юнитов
    # (пустой контекст => никогда не кластеризуется => остаётся автономным/CANONICAL).
    raw_units_by_id: dict[str, list[str]] = {}
    is_code_by_id: dict[str, bool] = {}
    for c in candidates:
        try:
            is_code = (
                c.is_code
                if c.is_code is not None
                else detect_code_content(c.url, c.text, c.method)
            )
            is_code_by_id[c.id] = is_code
            if is_code:
                extracts = _ast_semantic_extracts(c.text, c.url, min_chars=min_chars)
                if not extracts:
                    extracts = _code_paragraphs_from_raw_text(
                        c.text, min_chars=min_chars
                    )
                raw_units_by_id[c.id] = extracts
            else:
                raw_units_by_id[c.id] = _extract_paragraphs(
                    c.text, c.url, min_chars=min_chars
                )
        except Exception as exc:
            trace(
                f"PRE_MAP_DEDUP context_extract ✗ | {c.id[:60]} | "
                f"{type(exc).__name__}: {exc}"
            )
            is_code_by_id[c.id] = bool(c.is_code)
            raw_units_by_id[c.id] = []

    # Step 1b: Triage (Group Batching) — сырые юниты ВСЕХ кандидатов
    # классифицируются CORE/CONTEXT/DROP ОДНИМ вызовом Flash Lite (откат на
    # поштучные вызовы внутри, если объединённый payload переразмерен или
    # вызов упал — см. докстринг _flash_lite_triage_core_units_batch).
    trace(f"PRE_MAP_DEDUP step1b_triage ▶ | candidates={len(raw_units_by_id)}")
    t_1b = time.monotonic()
    core_by_id = await _flash_lite_triage_core_units_batch(
        raw_units_by_id, label="pre_map_dedup"
    )
    trace(f"PRE_MAP_DEDUP step1b_triage ✓ | {time.monotonic() - t_1b:.2f}s")

    # Step 1c: MMR (текст) / Head-3+Tail-3 извлечение полных тел (код) по
    # CORE-юнитам. Код игнорирует top_k — head/tail отбор фиксированная
    # эвристика (см. _bulk_gate_code_context), не настраивается конфигом.
    context_by_id: dict[str, list[str]] = {}
    for c in candidates:
        core_units = core_by_id.get(c.id, [])
        if is_code_by_id[c.id]:
            try:
                context_by_id[c.id] = _bulk_gate_code_context(
                    c.text, c.url, core_units, min_chars=min_chars
                )
            except Exception as exc:
                trace(
                    f"PRE_MAP_DEDUP bulk_gate_code_context ✗ | {c.id[:60]} | "
                    f"{type(exc).__name__}: {exc}"
                )
                context_by_id[c.id] = core_units
        else:
            context_by_id[c.id] = await asyncio.to_thread(
                _mmr_top_by_centroid, core_units, top_k=k
            )

    text_ids = [c.id for c in candidates if not is_code_by_id[c.id]]
    code_ids = [c.id for c in candidates if is_code_by_id[c.id]]

    # Step 2: BGE-кластеризация (только текст, 0 LLM-вызовов).
    doc_vectors: dict[str, list[float]] = {}
    for cid in text_ids:
        vec = await asyncio.to_thread(_pool_vector, context_by_id.get(cid) or [])
        if vec is not None:
            doc_vectors[cid] = vec
    groups = _cluster_text_candidates(doc_vectors, threshold=threshold)
    suspect_groups = [g for g in groups if len(g) > 1]
    trace(
        f"PRE_MAP_DEDUP cluster ✓ | text={len(text_ids)} code={len(code_ids)} "
        f"groups={len(groups)} suspect_groups={len(suspect_groups)} "
        f"threshold={threshold:.2f}"
    )

    # Step 3a/3b: Flash Lite Bulk Gate (TEXT suspect groups) и изолированная
    # дедупликация кода (code_deduplicator.py, весь контекст README/дерева +
    # свой вызов Flash Lite) запускаются ОДНОВРЕМЕННО через asyncio.gather —
    # текстовые и кодовые кандидаты строго независимы, ждать завершения
    # одного шага перед стартом другого не нужно (см. log_profiler.py: до
    # этой правки шаги шли строго последовательно, ~7s + ~13s подряд).
    step3a_map, step3b_map = await asyncio.gather(
        _run_step3a_bulk_gate(suspect_groups, context_by_id, anchor=anchor),
        _run_step3b_code_dedup(code_ids, candidates, anchor=anchor),
    )
    raw_map: dict[str, list[str]] = dict(step3a_map)
    for canonical_id, aliases in step3b_map.items():
        raw_map.setdefault(canonical_id, [])
        raw_map[canonical_id].extend(aliases)

    valid_ids = {c.id for c in candidates}
    clean_map = _sanitize_canonical_map(raw_map, valid_ids)

    # Step 4: пулинг канонических (Canonical Pooling).
    alias_of: dict[str, str] = {
        alias_id: canonical_id
        for canonical_id, aliases in clean_map.items()
        for alias_id in aliases
    }
    for c in candidates:
        if c.id in alias_of:
            result.decisions[c.id] = PreMapDedupDecision(
                id=c.id,
                is_canonical=False,
                canonical_id=alias_of[c.id],
                is_code=is_code_by_id[c.id],
                context_units=context_by_id.get(c.id, []),
            )
        else:
            result.decisions[c.id] = PreMapDedupDecision(
                id=c.id,
                is_canonical=True,
                canonical_id=c.id,
                is_code=is_code_by_id[c.id],
                context_units=context_by_id.get(c.id, []),
            )
    result.alias_map = clean_map

    trace(
        f"PRE_MAP_DEDUP ✓ | candidates={len(candidates)} "
        f"canonical={len(candidates) - len(alias_of)} alias={len(alias_of)} "
        f"canonical_groups={len(clean_map)}"
    )
    return result
