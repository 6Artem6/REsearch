"""Map-Reduce spatial summarizer (Gemma 4 cloud API)."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel

from knowledge_engine.config import (
    BLOG_SPATIAL_MAP_MAX_TOKENS,
    BLOG_SPATIAL_MAP_PROVIDER,
    BLOG_SPATIAL_TIMEOUT_SEC,
    GEMMA_FALLBACK_MAX_RPM,
    GEMMA_FALLBACK_MODEL,
    GEMMA_MAP_FIXED_MINUTE_PACING,
    GEMMA_MAP_FORCE_PER_MODEL_LIMITS,
    GEMMA_MAX_RPM,
    GEMMA_MAX_TPM,
    GEMMA_PRIMARY_MAX_RPM,
    GEMMA_PRIMARY_MODEL,
    GEMMA_REDUCE_MAX_OUTPUT_TOKENS,
    GEMMA_TARGET_TPM_SAFETY_CAP,
    MAP_DISPATCH_WINDOW_SEC,
    MAP_FACT_BUDGET_MAX,
    MAP_FACT_BUDGET_MIN,
    MAX_CONCURRENT_MAP_REQUESTS,
    REDUCE_STRATEGY,
    gemma_cloud_api_key_available,
    map_pipeline_concurrency,
)
from knowledge_engine.db.rag_chunks_schema import map_window_chunk_id
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.extraction import (
    SCOPE_TAGGING_PROMPT_RULES,
    KnowledgeAtom,
    reattach_source_chunk_ids_from_raw,
)
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    BlogArticleSummaryResponse,
    DeduplicatedAtomsResponse,
    FinalArticleSummaryResponse,
    MapWindowResponse,
    WindowDiagramCheck,
    final_to_legacy_summary,
    normalize_final_knowledge,
    normalize_map_knowledge,
)
from knowledge_engine.services.article_ingestion.map_diagram_attach import (
    build_attached_diagrams_block,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
    estimate_text_tokens,
    split_annotated_text_by_tokens,
)
from knowledge_engine.services.llm.gemma_client import (
    GemmaCloudClient,
    GemmaModelSlot,
    RateLimitedLLMClient,
    resolve_gemma_map_max_output_tokens,
)
from knowledge_engine.services.llm.rate_limiter import await_next_minute_window
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.run_log import trace

T = TypeVar("T", bound=BaseModel)

# Curriculum deep_blogs counts extracts ≥ 120 words. MAP/REDUCE still persists
# the summary; this threshold only explains why the hit is not counted as deep.
_POST_MAP_DEEP_WORDS = 120


def _post_map_thin_reason(final: FinalArticleSummaryResponse) -> str | None:
    """Why a successful REDUCE would not count toward deep_blogs (not a hard drop)."""
    takes = [
        str(t or "").strip()
        for t in (final.key_takeaways or [])
        if str(t or "").strip()
    ]
    exec_s = (final.executive_summary or "").strip()
    words = sum(len(t.split()) for t in takes) + (len(exec_s.split()) if exec_s else 0)
    atom_n = len(final.knowledge_atoms or [])
    if words >= _POST_MAP_DEEP_WORDS:
        return None
    if not takes and not exec_s:
        return f"empty takeaways/executive_summary after REDUCE " f"(atoms={atom_n})"
    return (
        f"extract_words={words} < {_POST_MAP_DEEP_WORDS} "
        "(takeaways/executive_summary too short for deep_blogs)"
    )


# Stable system prefix for provider/vLLM KV-cache hits across all MAP windows.
# Must stay free of per-chunk dynamics (chunk text goes only in the user message).
_MAP_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "CRITICAL: DO NOT output <thought> tags or any reasoning steps. "
    "Start your response IMMEDIATELY with the open curly bracket `{` and "
    "output ONLY pure, valid JSON.\n"
    "Ensure all backslashes inside JSON strings (e.g., in LaTeX math or quotes) "
    "are properly escaped with double backslashes (`\\\\`).\n\n"
    "You analyze ONE window of an annotated technical article "
    "(paragraph_inspector / map).\n"
    "Paragraphs [P_n], figures [FIG_m]. Read <window_text> first, then use "
    "<diagram_context> only to refine facts already present in the text.\n"
    "Produce a dense window_summary: integrate facts, numbers, axes, and "
    "architectural details from diagrams into the summary text "
    "(not as a separate figure list).\n"
    "window_role — short role tag (2–6 words), e.g. «Benchmarks», «Architecture».\n"
    "Do not select figures for VLM — they are already processed.\n"
    f"{SCOPE_TAGGING_PROMPT_RULES}\n"
    "You MUST fill knowledge_atoms — aim for the TARGET_FACTS count given in "
    "<article_context> (a soft target scaled to this window's size, not a "
    "hard cap; fewer is fine if the window genuinely has less to extract): "
    "{scope: PRINCIPLE|MECHANIC|INSTANCE, statement, context_quote, cluster_key}.\n"
    "cluster_key — a short (1-3 word) lowercase entity/topic tag for this atom, "
    "e.g. 'cpylex', 'gil_lock' (used downstream to group near-duplicate atoms "
    "before dedup — pick the specific subject the statement is about, not a "
    "generic category).\n"
    "source_chunk_ids is attached by the pipeline from CHUNK_ID — leave [] or omit.\n"
    "required_diagrams MUST be an empty array []. "
    "If non-empty, each item MUST be an object "
    "{figure_id, referenced_paragraphs, reason} with figure_id like FIG_1 — "
    "never a bare string, never an invented camelCase diagram name.\n"
    "JSON: window_role, window_summary, knowledge_atoms; required_diagrams — []."
)

_MAP_SYSTEM_CODE = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "CRITICAL: DO NOT output <thought> tags or any reasoning steps. "
    "Start your response IMMEDIATELY with the open curly bracket `{` and "
    "output ONLY pure, valid JSON.\n"
    "Ensure all backslashes inside JSON strings (e.g., in C escapes or quotes) "
    "are properly escaped with double backslashes (`\\\\`).\n\n"
    "You analyze ONE window of a source-code file or raw technical document "
    "(no HTML article chrome).\n"
    "Focus on control flow, data structures, concurrency, APIs, and invariants "
    "visible in this window. Preserve exact identifiers, macros, and signatures.\n"
    "Produce a dense window_summary of the mechanics in this slice.\n"
    "window_role — short role tag (2–6 words), e.g. «Locking», «Eval loop».\n"
    f"{SCOPE_TAGGING_PROMPT_RULES}\n"
    "You MUST fill knowledge_atoms — aim for the TARGET_FACTS count given in "
    "<article_context> (a soft target scaled to this window's size, not a "
    "hard cap; fewer is fine if the window genuinely has less to extract): "
    "{scope: PRINCIPLE|MECHANIC|INSTANCE, statement, context_quote, cluster_key}.\n"
    "cluster_key — a short (1-3 word) lowercase entity/topic tag for this atom, "
    "e.g. 'cpylex', 'gil_lock' (used downstream to group near-duplicate atoms "
    "before dedup — pick the specific subject the statement is about, not a "
    "generic category).\n"
    "source_chunk_ids is attached by the pipeline from CHUNK_ID — leave [] or omit.\n"
    "Do not select figures for VLM and do not invent conceptual diagram names.\n"
    "required_diagrams MUST be an empty array []. "
    "If non-empty, each item MUST be an object "
    "{figure_id, referenced_paragraphs, reason} with figure_id like FIG_1 — "
    "never a bare string, never an invented camelCase diagram name.\n"
    "JSON: window_role, window_summary, knowledge_atoms; required_diagrams — []."
)

_REDUCE_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Build the final executive_summary from all window_summary blocks "
    "(in window index order).\n"
    "Figures are already explained inside window text — do not drop them when compressing.\n"
    f"{SCOPE_TAGGING_PROMPT_RULES}\n"
    "Aggregate knowledge_atoms from all windows, PRESERVING original scope tags "
    "(Reduce must not rewrite PRINCIPLE↔INSTANCE).\n"
    "key_takeaways — 3–7 compressed synthesis lines of the form «[SCOPE: …] …» "
    "(not a dump of knowledge_atoms; the full catalog stays in knowledge_atoms).\n"
    "Strictly follow the <critical_reduce_rules> block at the end of the user message.\n"
    "target_diagrams_for_vlm — always an empty array [].\n"
    "JSON conforming to FinalArticleSummaryResponse."
)

# Phase 1 (two_phase): atom deduplication only — stable module constant for KV-cache.
_REDUCE_DEDUP_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "CRITICAL: DO NOT output <thought> tags. Start with `{` and output ONLY JSON.\n\n"
    "You deduplicate knowledge atoms extracted from different windows of ONE article.\n"
    "Input: raw_knowledge_atoms (full KnowledgeAtom objects: scope, statement, "
    "context_quote, source_chunk_ids).\n"
    "Rules:\n"
    "1. Do NOT drop any unique fact or detail — completeness over brevity.\n"
    "2. When two atoms describe the same claim, MERGE into one fuller statement "
    "that keeps all exact numbers, formulas, model/library names, and parameters.\n"
    "3. When merging duplicate or overlapping atoms, UNION their source_chunk_ids "
    "into one unique list — preserve references to ALL source chunks.\n"
    "4. Prefer the stronger scope when reconciling the same principle "
    "(PRINCIPLE > MECHANIC > INSTANCE).\n"
    "5. Return full KnowledgeAtom objects (not bare strings).\n"
    "6. context_quote may be the best supporting quote among merges (or empty).\n"
    f"{SCOPE_TAGGING_PROMPT_RULES}\n"
    "JSON schema: DeduplicatedAtomsResponse "
    "{{ knowledge_atoms: [{{scope, statement, context_quote, source_chunk_ids}}] }}."
)

# Phase 2 (two_phase): executive synthesis — atoms come from phase 1, not new extraction.
_REDUCE_SYNTHESIS_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "CRITICAL: DO NOT output <thought> tags. Start with `{` and output ONLY JSON.\n\n"
    "You write the article passport (executive_summary + key_takeaways) for a "
    "technical paper.\n"
    "Inputs: (1) clean_knowledge_atoms — already deduplicated facts; "
    "(2) window_summary blocks — narrative scaffolding only.\n"
    "Rules:\n"
    "1. Do NOT extract new facts from window_summary text. "
    "All factual claims must come from clean_knowledge_atoms.\n"
    "2. Your knowledge_atoms output MUST always be an empty array []. This "
    "OVERRIDES any 'fill knowledge_atoms' instruction below — that generic "
    "rule is for MAP/dedup calls, not this one. "
    "The pipeline substitutes the verified phase-1 clean_knowledge_atoms list "
    "after your response — do not re-emit them: it wastes output tokens and "
    "risks truncating your JSON mid-array on atom-rich articles.\n"
    "3. Use window_summary ONLY as structural context to write "
    "executive_summary (1–2 coherent paragraphs) and key_takeaways "
    "(3–7 lines of the form «[SCOPE: …] …»).\n"
    "4. Keep exact numbers / names from atoms inside INSTANCE takeaways; "
    "do not promote them to industry-wide standards.\n"
    "5. target_diagrams_for_vlm — always [].\n"
    f"{SCOPE_TAGGING_PROMPT_RULES}\n"
    "JSON conforming to FinalArticleSummaryResponse."
)

_CRITICAL_REDUCE_RULES = (
    "1. Keep exact numbers, units, benchmark and model names inside "
    "[SCOPE: INSTANCE]; do not present them as industry-wide standards in "
    "executive_summary.\n"
    "2. Deduplicate facts: one canonical claim; merge overlapping windows without "
    "repeats; do not downgrade the winning scope tag "
    "(PRINCIPLE > MECHANIC > INSTANCE when reconciling the same principle).\n"
    "3. Architecture and figures belong inside executive_summary, not a separate FIG list.\n"
    "4. key_takeaways: 3–7 compressed synthesis items prefixed with "
    "[SCOPE: PRINCIPLE|MECHANIC|INSTANCE]; "
    "experiment numbers / libraries / limits — INSTANCE only. "
    "Do not dump the full knowledge_atoms catalog into takeaways.\n"
    "5. knowledge_atoms is the full fact catalog (separate from key_takeaways).\n"
    "6. target_diagrams_for_vlm — always []."
)


_MAP_PARALLEL_HINT_LOGGED = False


def _log_map_parallel_hint(
    use_gemma: bool, map_provider: str, map_model_label: str
) -> None:
    global _MAP_PARALLEL_HINT_LOGGED
    if _MAP_PARALLEL_HINT_LOGGED:
        return
    _MAP_PARALLEL_HINT_LOGGED = True
    conc = map_pipeline_concurrency()
    _ = use_gemma
    trace(
        f"BLOG_SPATIAL map-reduce | backend=gemma provider={map_provider} "
        f"model={map_model_label} concurrency={conc} "
        f"fixed_minute={GEMMA_MAP_FIXED_MINUTE_PACING} "
        f"tpm_cap={GEMMA_TARGET_TPM_SAFETY_CAP} "
        f"unified_pool={GEMMA_MAP_FORCE_PER_MODEL_LIMITS}"
    )


def _norm_fig(raw: str) -> str:
    t = (raw or "").strip().upper()
    if t.startswith("FIG_"):
        return t
    if t.startswith("FIG"):
        return f"FIG_{t[3:].lstrip('_')}"
    return t


def _norm_p(raw: str) -> str:
    t = (raw or "").strip().upper()
    if t.startswith("P_"):
        return t
    return f"P_{t.lstrip('P_')}"


def merge_window_diagrams(
    map_results: list[MapWindowResponse],
) -> list[WindowDiagramCheck]:
    by_fig: dict[str, WindowDiagramCheck] = {}
    for mr in map_results:
        for d in mr.required_diagrams or []:
            fid = _norm_fig(d.figure_id)
            if not fid.startswith("FIG_"):
                continue
            paras = [_norm_p(p) for p in d.referenced_paragraphs or []]
            if fid not in by_fig:
                by_fig[fid] = WindowDiagramCheck(
                    figure_id=fid,
                    referenced_paragraphs=[],
                    reason=(d.reason or "").strip(),
                )
            existing = by_fig[fid]
            existing.referenced_paragraphs.extend(paras)
            if len(d.reason or "") > len(existing.reason):
                existing.reason = (d.reason or "").strip()
    out: list[WindowDiagramCheck] = []
    for item in by_fig.values():
        item.referenced_paragraphs = sorted(set(item.referenced_paragraphs))
        if item.referenced_paragraphs or item.reason:
            out.append(item)
    return out


@dataclass
class MapReduceArticleJob:
    job_id: str
    title: str
    url: str
    windows: list[TokenWindowChunk]
    all_figure_ids: list[str] = field(default_factory=list)
    figure_registry: object | None = None
    trust_score: float = 1.0
    source_kind: str = "article"
    anchor_index_map: dict[str, dict[str, object]] = field(default_factory=dict)
    unverified_citations: list[str] = field(default_factory=list)
    consensus_nodes: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
            apply_chunk_anchors_to_windows,
        )

        self.anchor_index_map = apply_chunk_anchors_to_windows(
            self.windows, url=self.url
        )


def _map_system_for_job(job: MapReduceArticleJob) -> str:
    if (job.source_kind or "article") == "source_code":
        return _MAP_SYSTEM_CODE
    return _MAP_SYSTEM


@dataclass
class MapReduceJobOutcome:
    """Final REDUCE passport + per-window MAP results (for LanceDB provenance)."""

    final: FinalArticleSummaryResponse | None
    map_results: list[MapWindowResponse | None] = field(default_factory=list)
    unverified_citations: list[str] = field(default_factory=list)
    anchor_index_map: dict[str, dict[str, object]] = field(default_factory=dict)
    consensus_nodes: list[object] = field(default_factory=list)


@dataclass
class _ArticleMapState:
    job: MapReduceArticleJob
    results: list[MapWindowResponse | None]
    pending: int
    map_started: bool = False


def _strip_redundant_article_header(body: str, article_title: str) -> str:
    text = (body or "").strip()
    title = (article_title or "").strip()[:300]
    if title:
        for prefix in (f"Article: {title}", f"Article:{title}"):
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip("\n")
                break
    return text


def _wrap_diagram_context(attached: str) -> str:
    block = (attached or "").strip()
    if not block:
        return ""
    return f"<diagram_context>\n{block}\n</diagram_context>"


# Rough per-atom output footprint (scope + statement + context_quote +
# source_chunk_ids as JSON) — calibration heuristic, not a measured constant.
# Only used to keep a proportionally-large target_facts from overshooting
# GEMMA_MAP_MAX_OUTPUT_TOKENS; MAP_FACT_BUDGET_MIN/MAX are the real (env) caps.
_EST_OUTPUT_TOKENS_PER_ATOM = 110


def dynamic_target_facts(
    window_text: str, diagram_block: str = ""
) -> tuple[int, int, int]:
    """Adaptive per-window knowledge_atoms target — proportional to window
    size instead of the same fixed range for every window. Returns
    (input_tokens, target_facts, projected_out_tokens)."""
    input_tokens = estimate_text_tokens(f"{window_text}\n{diagram_block}".strip())
    ratio = min(1.0, max(0.0, input_tokens / max(1, BLOG_SPATIAL_MAP_MAX_TOKENS)))
    target = int(
        round(MAP_FACT_BUDGET_MIN + ratio * (MAP_FACT_BUDGET_MAX - MAP_FACT_BUDGET_MIN))
    )
    target = max(MAP_FACT_BUDGET_MIN, min(MAP_FACT_BUDGET_MAX, target))
    output_cap = resolve_gemma_map_max_output_tokens()
    projected_out = target * _EST_OUTPUT_TOKENS_PER_ATOM
    if projected_out > output_cap:
        target = max(MAP_FACT_BUDGET_MIN, output_cap // _EST_OUTPUT_TOKENS_PER_ATOM)
        projected_out = target * _EST_OUTPUT_TOKENS_PER_ATOM
    return input_tokens, target, projected_out


def _prompt_for_window(job: MapReduceArticleJob, w: TokenWindowChunk) -> str:
    attached = (w.attached_diagrams or "").strip()
    if not attached:
        attached = build_attached_diagrams_block(
            w.body,
            job.figure_registry,  # type: ignore[arg-type]
            extra_figure_ids=w.figure_ids,
        )
    section = (w.section_heading or "").strip() or "—"
    window_text = _strip_redundant_article_header(w.body, job.title)
    doc_id = VectorStore.doc_id_for_url(job.url)
    chunk_id = map_window_chunk_id(doc_id, w.window_index)
    from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
        maybe_prepend_chunk_anchor,
        ordinal_for_window,
        strip_context_anchor_prefix,
    )

    window_text = strip_context_anchor_prefix(window_text)
    ordinal = ordinal_for_window(w.window_index, job.anchor_index_map)
    window_text = maybe_prepend_chunk_anchor(window_text, ordinal)
    diagram_block = _wrap_diagram_context(attached)
    _in_tok, target_facts, _proj_out = dynamic_target_facts(window_text, diagram_block)
    parts = [
        "<article_context>",
        f"ARTICLE_TITLE: {job.title[:300]}",
        f"ARTICLE_URL: {job.url[:500]}",
        f"SECTION: {section}",
        f"WINDOW_INDEX: {w.window_index}",
        f"CHUNK_ID: {chunk_id}",
        f"TARGET_FACTS: {target_facts} (soft target, proportional to window size — not a hard cap)",
        "</article_context>",
        "",
        "<window_text>",
        window_text,
        "</window_text>",
    ]
    if diagram_block:
        parts.extend(["", diagram_block])
    return "\n".join(parts)


def _default_window_role() -> str:
    return "общий контекст"


def _format_reduce_summaries_block(
    windows: list[TokenWindowChunk],
    map_results: list[MapWindowResponse | None],
    *,
    include_atoms: bool = True,
    index_map: dict[str, dict[str, object]] | None = None,
) -> str:
    from knowledge_engine import config as ke_config
    from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
        ordinal_for_window,
    )

    inject = bool(ke_config.CHUNK_ANCHOR_INJECTION)
    sections: list[str] = []
    current_section: str | None = None
    for w, m in zip(windows, map_results):
        if m is None:
            continue
        sec = (w.section_heading or "").strip() or "—"
        if sec != current_section:
            sections.append(f"## Section: {sec}")
            current_section = sec
        role = (m.window_role or "").strip() or _default_window_role()
        summary = (m.window_summary or "").strip()
        if inject:
            aid = f"A{ordinal_for_window(w.window_index, index_map)}"
            block = f"### [{aid}] Window {w.window_index} [{role}]\n{summary}"
        else:
            block = f"### Window {w.window_index} [{role}]\n{summary}"
        if include_atoms:
            atoms = list(m.knowledge_atoms or [])
            if atoms:
                tagged = "\n".join(f"- {a.format_tagged()}" for a in atoms[:16])
                block += f"\n\nknowledge_atoms:\n{tagged}"
        sections.append(block)
    return "\n\n".join(sections)


def _collect_raw_knowledge_atoms(
    map_results: list[MapWindowResponse | None],
) -> list[KnowledgeAtom]:
    pooled: list[KnowledgeAtom] = []
    for m in map_results:
        if m is None:
            continue
        pooled.extend(list(m.knowledge_atoms or []))
    return pooled


def _format_atoms_json_block(atoms: list[KnowledgeAtom]) -> str:
    payload = {
        "raw_knowledge_atoms": [
            {
                "scope": a.scope.value,
                "statement": a.statement,
                "context_quote": a.context_quote or "",
                "source_chunk_ids": list(a.source_chunk_ids or []),
            }
            for a in atoms
        ]
    }
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_atoms_compact_block(atoms: list[KnowledgeAtom]) -> str:
    """Scope+statement only — for the synthesis prompt, which only needs to
    read atoms to write prose (executive_summary/key_takeaways); the model
    never re-emits them (final.knowledge_atoms is reattached post-hoc from
    clean_atoms). context_quote/source_chunk_ids are provenance details the
    dedup phase needs but synthesis doesn't — dropping them here cuts the
    input of the heaviest, slowest REDUCE call without touching output."""
    lines = [f"- [{a.scope.value}] {a.statement}" for a in atoms]
    return "\n".join(lines)


def _build_reduce_user_prompt(job: MapReduceArticleJob, summaries_block: str) -> str:
    trust = float(job.trust_score if job.trust_score is not None else 1.0)
    return (
        "<article_context>\n"
        f"ARTICLE_TITLE: {job.title[:300]}\n"
        f"ARTICLE_URL: {job.url[:500]}\n"
        f"ARTICLE_TRUST_SCORE: {trust:.3f}\n"
        "</article_context>\n\n"
        "## Window summaries (ordered; prefer higher-trust article claims)\n"
        f"{summaries_block}\n\n"
        "<critical_reduce_rules>\n"
        f"{_CRITICAL_REDUCE_RULES}\n"
        "</critical_reduce_rules>"
    )


def _build_dedup_user_prompt(
    job: MapReduceArticleJob, atoms: list[KnowledgeAtom]
) -> str:
    return (
        "<article_context>\n"
        f"ARTICLE_TITLE: {job.title[:300]}\n"
        f"ARTICLE_URL: {job.url[:500]}\n"
        "</article_context>\n\n"
        "<raw_knowledge_atoms>\n"
        f"{_format_atoms_json_block(atoms)}\n"
        "</raw_knowledge_atoms>\n\n"
        "Return DeduplicatedAtomsResponse JSON with merged knowledge_atoms."
    )


def _build_synthesis_user_prompt(
    job: MapReduceArticleJob,
    *,
    clean_atoms: list[KnowledgeAtom],
    summaries_block: str,
    summaries_in_cache: bool = False,
) -> str:
    trust = float(job.trust_score if job.trust_score is not None else 1.0)
    atoms_block = _format_atoms_compact_block(clean_atoms)
    if summaries_in_cache:
        scaffold = (
            "## window_summary scaffolding is in cached_content "
            "(do NOT mine new facts from it).\n"
        )
    else:
        scaffold = (
            "## window_summary scaffolding (context only — do NOT mine new facts)\n"
            f"{summaries_block}\n\n"
        )
    return (
        "<article_context>\n"
        f"ARTICLE_TITLE: {job.title[:300]}\n"
        f"ARTICLE_URL: {job.url[:500]}\n"
        f"ARTICLE_TRUST_SCORE: {trust:.3f}\n"
        "</article_context>\n\n"
        "<clean_knowledge_atoms>\n"
        f"{atoms_block}\n"
        "</clean_knowledge_atoms>\n\n"
        f"{scaffold}"
        "Write executive_summary (1–2 paragraphs) and key_takeaways (3–7 tagged lines). "
        "Set knowledge_atoms to the clean list (scope labels only may be normalized). "
        "target_diagrams_for_vlm=[]."
    )


def _resolve_map_provider() -> tuple[bool, str]:
    """Always Gemma Cloud — Ollama MAP backend is removed."""
    _ = BLOG_SPATIAL_MAP_PROVIDER
    return True, "gemma_cloud"


async def _try_cached_structured_reduce(
    job: MapReduceArticleJob,
    *,
    system: str,
    cache_content: str,
    user_prompt: str,
    schema: type[T],
    max_tokens: int | None = None,
) -> T | None:
    """REDUCE через Gemini cached_content; None → GemmaCloudClient."""
    from knowledge_engine import config as ke_config

    if not ke_config.MIGRATION_USE_CONTEXT_CACHING:
        return None
    if not (cache_content or "").strip():
        return None
    try:
        from knowledge_engine.services.llm.ingest_context_cache_manager import (
            IngestContextCacheManager,
        )

        doc_id = VectorStore.doc_id_for_url(job.url)
        mgr = IngestContextCacheManager()
        return await asyncio.to_thread(
            mgr.generate_structured,
            doc_id=doc_id,
            system_instruction=system,
            cache_content=cache_content,
            user_prompt=user_prompt,
            schema=schema,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        trace(f"INGEST_CACHE REDUCE fallback | {type(exc).__name__}: {exc}")
        return None


async def _structured_reduce_call(
    system: str,
    prompt: str,
    schema: type[T],
    *,
    label: str,
    http_client: httpx.AsyncClient,
    gemma_rl: RateLimitedLLMClient | None,
    max_tokens: int | None = None,
) -> T | None:
    out_tokens = (
        max_tokens if max_tokens is not None else GEMMA_REDUCE_MAX_OUTPUT_TOKENS
    )
    if gemma_rl is not None:
        return await gemma_rl.post_structured(
            system,
            prompt,
            schema,
            label=label,
            client=http_client,
            max_tokens=out_tokens,
            # REDUCE gets priority: ignore the TPM/RPM headroom reserved
            # away from MAP callers so document completion never queues
            # behind MAP's own usage of the same slot.
            priority=True,
        )
    # AUDIT FIX: this used to call complete_structured() with no limiter at
    # all — complete_structured()'s own `limiter` kwarg only reconciles
    # AFTER the call, never blocks admission — so this path (dead in the
    # current pipeline, since map_reduce_jobs_pooled_async always builds a
    # gemma_rl, but a real hole for any other caller) must acquire against
    # the SAME registry entry build_gemma_model_slots() uses for the
    # primary model before sending the request, for 100% counter coverage.
    from knowledge_engine.services.llm.rate_limiter import get_gemma_rate_limiter

    client = GemmaCloudClient()
    est = client.estimate_request_tokens(
        system, prompt, schema=schema, max_output_tokens=out_tokens
    )
    fallback_limiter = get_gemma_rate_limiter(slot=f"gemma:{client.model}")
    await fallback_limiter.acquire(est, model=client.model, priority=True)
    return await client.complete_structured(
        system,
        prompt,
        schema,
        label=label,
        client=http_client,
        limiter=fallback_limiter,
        max_tokens=out_tokens,
    )


async def _run_legacy_reduce(
    job: MapReduceArticleJob,
    map_results: list[MapWindowResponse | None],
    *,
    http_client: httpx.AsyncClient,
    gemma_rl: RateLimitedLLMClient | None = None,
) -> FinalArticleSummaryResponse | None:
    """Single-call REDUCE (A/B baseline)."""
    summaries_block = _format_reduce_summaries_block(
        job.windows,
        map_results,
        include_atoms=True,
        index_map=job.anchor_index_map,
    )
    reduce_prompt = _build_reduce_user_prompt(job, summaries_block)
    backend = "gemma"
    model = GEMMA_PRIMARY_MODEL
    trace(
        f"BLOG_SPATIAL reduce ▶ legacy | backend={backend} model={model} "
        f"| {job.url[:55]}"
    )
    cached = await _try_cached_structured_reduce(
        job,
        system=_REDUCE_SYSTEM,
        cache_content=summaries_block,
        user_prompt=_build_reduce_user_prompt(
            job,
            "(Window summaries are supplied via cached_content.)",
        ),
        schema=FinalArticleSummaryResponse,
    )
    if cached is not None:
        return cached
    return await _structured_reduce_call(
        _REDUCE_SYSTEM,
        reduce_prompt,
        FinalArticleSummaryResponse,
        label=f"reduce_legacy/{job.job_id[:20]}",
        http_client=http_client,
        gemma_rl=gemma_rl,
    )


async def _run_two_phase_reduce(
    job: MapReduceArticleJob,
    map_results: list[MapWindowResponse | None],
    *,
    http_client: httpx.AsyncClient,
    gemma_rl: RateLimitedLLMClient | None = None,
) -> FinalArticleSummaryResponse | None:
    """Phase1 atom dedup → Phase2 executive synthesis (lighter schemas)."""
    backend = "gemma"
    model = GEMMA_PRIMARY_MODEL
    raw_atoms = _collect_raw_knowledge_atoms(map_results)
    summaries_only = _format_reduce_summaries_block(
        job.windows,
        map_results,
        include_atoms=False,
        index_map=job.anchor_index_map,
    )

    clean_atoms = list(raw_atoms)
    consensus_applied = False
    if raw_atoms:
        from knowledge_engine.services.deduplication.entity_consensus_engine import (
            apply_entity_consensus_to_atoms,
            claim_dedup_is_enabled,
        )

        if claim_dedup_is_enabled():
            try:
                collapsed = await apply_entity_consensus_to_atoms(
                    raw_atoms,
                    index_map=job.anchor_index_map,
                    http_client=http_client,
                    gemma_rl=gemma_rl,
                )
            except Exception as exc:
                trace(
                    f"BLOG_SPATIAL reduce ⚠ entity_consensus failed | "
                    f"{type(exc).__name__}: {exc}"
                )
                collapsed = None
            if collapsed is not None:
                clean_atoms, nodes = collapsed
                job.consensus_nodes = list(nodes)
                consensus_applied = True
                trace(
                    f"BLOG_SPATIAL reduce ✓ entity_consensus | "
                    f"atoms_out={len(clean_atoms)} nodes={len(nodes)}"
                )
    if raw_atoms and not consensus_applied:
        trace(
            f"BLOG_SPATIAL reduce ▶ two_phase/dedup | backend={backend} "
            f"model={model} atoms_in={len(raw_atoms)} | {job.url[:55]}"
        )
        dedup = await _structured_reduce_call(
            _REDUCE_DEDUP_SYSTEM,
            _build_dedup_user_prompt(job, raw_atoms),
            DeduplicatedAtomsResponse,
            label=f"reduce_dedup/{job.job_id[:20]}",
            http_client=http_client,
            gemma_rl=gemma_rl,
            max_tokens=min(2048, GEMMA_REDUCE_MAX_OUTPUT_TOKENS),
        )
        if dedup is not None and (dedup.knowledge_atoms or []):
            clean_atoms = list(dedup.knowledge_atoms)
            trace(
                f"BLOG_SPATIAL reduce ✓ two_phase/dedup | atoms_out={len(clean_atoms)}"
            )
        else:
            trace(
                "BLOG_SPATIAL reduce ⚠ two_phase/dedup failed — "
                "using pooled MAP atoms"
            )
    elif not raw_atoms:
        trace("BLOG_SPATIAL reduce ⚠ two_phase/dedup skip | no MAP atoms")

    # Provenance: union source_chunk_ids even if Gemma dropped them on merge.
    clean_atoms = reattach_source_chunk_ids_from_raw(clean_atoms, raw_atoms)

    trace(
        f"BLOG_SPATIAL reduce ▶ two_phase/synthesis | backend={backend} "
        f"atoms={len(clean_atoms)} | {job.url[:55]}"
    )
    synth_full = _build_synthesis_user_prompt(
        job, clean_atoms=clean_atoms, summaries_block=summaries_only
    )
    # Output no longer echoes knowledge_atoms (rule 2 above forces []) — its size
    # is now independent of atom count, so the dedup phase's smaller cap is safe
    # and avoids the truncation risk a full-width 4096 cap gave atom-rich articles.
    synth_max_tokens = min(2048, GEMMA_REDUCE_MAX_OUTPUT_TOKENS)
    cached = await _try_cached_structured_reduce(
        job,
        system=_REDUCE_SYNTHESIS_SYSTEM,
        cache_content=summaries_only,
        user_prompt=_build_synthesis_user_prompt(
            job,
            clean_atoms=clean_atoms,
            summaries_block=summaries_only,
            summaries_in_cache=True,
        ),
        schema=FinalArticleSummaryResponse,
        max_tokens=synth_max_tokens,
    )
    final = cached
    if final is None:
        final = await _structured_reduce_call(
            _REDUCE_SYNTHESIS_SYSTEM,
            synth_full,
            FinalArticleSummaryResponse,
            label=f"reduce_synth/{job.job_id[:20]}",
            http_client=http_client,
            gemma_rl=gemma_rl,
            max_tokens=synth_max_tokens,
        )
    if final is None:
        return None
    # Phase-2 contract: factual atoms come from phase 1 (or pooled MAP).
    if clean_atoms:
        final.knowledge_atoms = list(clean_atoms)
    return final


async def run_reduce(
    job: MapReduceArticleJob,
    map_results: list[MapWindowResponse | None],
    *,
    http_client: httpx.AsyncClient,
    gemma_rl: RateLimitedLLMClient | None = None,
) -> FinalArticleSummaryResponse | None:
    """REDUCE dispatcher — ``REDUCE_STRATEGY`` = two_phase | legacy."""
    strategy = (REDUCE_STRATEGY or "two_phase").strip().lower()
    if strategy == "legacy":
        return await _run_legacy_reduce(
            job, map_results, http_client=http_client, gemma_rl=gemma_rl
        )
    return await _run_two_phase_reduce(
        job, map_results, http_client=http_client, gemma_rl=gemma_rl
    )


def _annotate_reduce_anchor_citations(
    job: MapReduceArticleJob,
    final: FinalArticleSummaryResponse,
) -> FinalArticleSummaryResponse:
    from knowledge_engine.services.validators.anchor_validator import (
        validate_and_annotate_anchors,
    )

    valid = {str(k) for k in (job.anchor_index_map or {})}
    unverified: list[str] = []
    text, extra = validate_and_annotate_anchors(final.executive_summary or "", valid)
    final.executive_summary = text
    unverified.extend(extra)
    takes: list[str] = []
    for item in final.key_takeaways or []:
        marked, extra = validate_and_annotate_anchors(item, valid)
        takes.append(marked)
        unverified.extend(extra)
    final.key_takeaways = takes
    job.unverified_citations = list(dict.fromkeys(unverified))
    return final


async def _reduce_final_from_maps(
    job: MapReduceArticleJob,
    map_results: list[MapWindowResponse | None],
    *,
    http_client: httpx.AsyncClient,
    gemma_rl: RateLimitedLLMClient | None = None,
) -> FinalArticleSummaryResponse | None:
    valid_maps = [m for m in map_results if m is not None]
    if not valid_maps:
        trace(f"BLOG_SPATIAL map ✗ | all windows failed | {job.url[:50]}")
        return None

    final = await run_reduce(
        job,
        map_results,
        http_client=http_client,
        gemma_rl=gemma_rl,
    )
    if final is None:
        trace(f"BLOG_SPATIAL reduce ✗ | failed | {job.url[:50]}")
        return None

    final.target_diagrams_for_vlm = []
    # Если Reduce потерял atoms — собрать с MAP-окон
    if not (final.knowledge_atoms or []):
        pooled: list[KnowledgeAtom] = _collect_raw_knowledge_atoms(map_results)
        final.knowledge_atoms = pooled
    else:
        # Legacy / synthesis may drop provenance — restore from MAP atoms.
        final.knowledge_atoms = reattach_source_chunk_ids_from_raw(
            final.knowledge_atoms,
            _collect_raw_knowledge_atoms(map_results),
        )
    final = normalize_final_knowledge(final)
    final = _annotate_reduce_anchor_citations(job, final)

    trace(
        f"BLOG_SPATIAL map-reduce ✓ | backend={'gemma' if gemma_rl else 'ollama'} "
        f"strategy={REDUCE_STRATEGY} | {job.url[:50]} windows={len(valid_maps)} "
        f"takeaways={len(final.key_takeaways)} "
        f"atoms={len(final.knowledge_atoms or [])}"
    )
    thin = _post_map_thin_reason(final)
    if thin:
        trace(f"[Triage Post-MAP] Dropped {job.url} due to: {thin}")
    return final


async def map_reduce_jobs_pooled_async(
    jobs: list[MapReduceArticleJob],
    *,
    force_gemma_cloud: bool = False,
) -> dict[str, MapReduceJobOutcome]:
    """MAP: общий пул чанков всех статей; REDUCE: по готовности каждой статьи."""
    if not jobs:
        return {}

    if not gemma_cloud_api_key_available():
        trace("BLOG_SPATIAL map-reduce ⊘ | Gemma Cloud API key unset")
        return {j.job_id: MapReduceJobOutcome(final=None, map_results=[]) for j in jobs}
    use_gemma, map_provider = True, (
        "gemma_cloud_forced" if force_gemma_cloud else "gemma_cloud"
    )

    map_model_label = f"{GEMMA_PRIMARY_MODEL}→{GEMMA_FALLBACK_MODEL}"
    backend = "gemma"
    _log_map_parallel_hint(use_gemma, map_provider, map_model_label)
    # Unified for every MAP provider/model — no per-backend concurrency fork.
    map_concurrency = map_pipeline_concurrency()
    total_chunks = sum(len(j.windows) for j in jobs)
    trace(
        f"BLOG_SPATIAL map-reduce ▶ | backend={backend} "
        f"BLOG_SPATIAL_MAP_PROVIDER={(BLOG_SPATIAL_MAP_PROVIDER or 'gemma_cloud')!r} "
        f"resolved={map_provider} model={map_model_label} "
        f"articles={len(jobs)} chunks={total_chunks} concurrency={map_concurrency}"
    )

    states: list[_ArticleMapState] = []
    for job in jobs:
        n = len(job.windows)
        states.append(
            _ArticleMapState(
                job=job,
                results=[None] * n,
                pending=n,
            )
        )

    # Hard cap for all MAP backends: Semaphore(MAX_CONCURRENT_MAP_REQUESTS).
    map_sem = asyncio.Semaphore(MAX_CONCURRENT_MAP_REQUESTS)
    # LATENCY AUDIT: was capped at 2 regardless of batch size — with 4
    # articles finishing MAP around the same time, the 3rd/4th had to wait
    # for an EARLIER article's whole REDUCE phase (consensus + synthesis,
    # tens of seconds) to free a slot even though their own MAP work was
    # long done. Confirmed live: one article's REDUCE_START was delayed
    # ~30s purely on this semaphore, not on TPM/RPM (REDUCE calls run with
    # priority=True and had free headroom). 4 covers a typical batch
    # without unbounded fan-out for larger ones.
    reduce_sem = asyncio.Semaphore(max(1, min(4, map_concurrency)))
    reduce_queue: asyncio.Queue[_ArticleMapState] = asyncio.Queue()
    finals: dict[str, MapReduceJobOutcome] = {
        j.job_id: MapReduceJobOutcome(final=None, map_results=[]) for j in jobs
    }

    timeout = httpx.Timeout(BLOG_SPATIAL_TIMEOUT_SEC)
    gemma_rl = (
        RateLimitedLLMClient(
            # BUG FIXED: was `GEMMA_MAP_FORCE_PER_MODEL_LIMITS and
            # GEMMA_MAP_FIXED_MINUTE_PACING` — with the documented default
            # (fixed_minute=False, the modern sliding-window dispatcher),
            # that AND collapsed to False regardless of
            # GEMMA_MAP_FORCE_PER_MODEL_LIMITS, silently handing both model
            # slots the SAME shared limiter (see config.py comment: this
            # flag must ignore GEMMA_QUOTA_SHARED for MAP, unconditionally).
            # acquire_parallel_wave's dual-basket split then always skipped
            # the second slot (its seen_limiters dedup guard treats a
            # shared limiter as "already tried"), capping every MAP wave at
            # a single in-flight request instead of two — confirmed via
            # perf_debug.log: MAP windows for the same node ran fully
            # sequentially (1875s / 1325s vs a ~480s baseline) with the
            # fallback model never once appearing in a wave-reserve trace.
            map_parallel_streams=GEMMA_MAP_FORCE_PER_MODEL_LIMITS,
        )
        if use_gemma
        else None
    )

    async def reduce_worker() -> None:
        while True:
            state = await reduce_queue.get()
            try:
                async with reduce_sem:
                    trace(f"[REDUCE_START] | {state.job.url[:60]}")
                    final = await _reduce_final_from_maps(
                        state.job,
                        state.results,
                        http_client=http_client,
                        gemma_rl=gemma_rl,
                    )
                    trace(
                        f"[REDUCE_DONE] | {state.job.url[:60]} | "
                        f"final={'ok' if final is not None else 'failed'}"
                    )
                finals[state.job.job_id] = MapReduceJobOutcome(
                    final=final,
                    map_results=list(state.results),
                    unverified_citations=list(state.job.unverified_citations),
                    anchor_index_map=dict(state.job.anchor_index_map),
                    consensus_nodes=list(state.job.consensus_nodes),
                )
            finally:
                reduce_queue.task_done()

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        # STREAMING AUDIT FIX: this stayed at 2 when reduce_sem above was
        # raised 2->4 — real concurrency is min(worker task count,
        # reduce_sem cap), so only 2 of the 4 permitted slots could ever be
        # occupied. Raising reduce_sem alone did nothing once >2 articles'
        # MAP finished close together; must match reduce_sem's cap here.
        reduce_workers = max(1, min(4, map_concurrency))
        workers = [asyncio.create_task(reduce_worker()) for _ in range(reduce_workers)]

        async def _on_chunk_done(
            state: _ArticleMapState,
            window_index: int,
            result: MapWindowResponse | None,
        ) -> None:
            if result is not None:
                doc_id = VectorStore.doc_id_for_url(state.job.url)
                chunk_id = map_window_chunk_id(doc_id, window_index)
                result = normalize_map_knowledge(result, source_chunk_id=chunk_id)
            state.results[window_index] = result
            state.pending -= 1
            w_total = len(state.job.windows)
            n_atoms = len(result.knowledge_atoms or []) if result else 0
            trace(
                f"BLOG_SPATIAL map ✓ | chunk ready "
                f"{window_index + 1}/{w_total} article={state.job.url[:45]} "
                f"pending={state.pending} atoms={n_atoms}"
            )
            if state.pending <= 0:
                trace(f"[MAP_DONE] | {state.job.url[:60]} | windows={w_total}")
                await reduce_queue.put(state)

        async def _map_gemma_chunk_preacquired(
            state: _ArticleMapState,
            w: TokenWindowChunk,
            slot: GemmaModelSlot,
            *,
            http_client: httpx.AsyncClient,
        ) -> tuple[MapWindowResponse | None, int]:
            if not state.map_started:
                state.map_started = True
                trace(
                    f"[MAP_START] | {state.job.url[:60]} | windows={len(state.job.windows)}"
                )
            map_system = _map_system_for_job(state.job)
            async with map_sem:
                prompt = _prompt_for_window(state.job, w)
                inp, out_cap, _ = gemma_rl.estimate_budget(  # type: ignore[union-attr]
                    map_system, prompt, MapWindowResponse
                )
                _fb_in, target_facts, projected_out = dynamic_target_facts(w.body)
                trace(
                    f"BLOG_SPATIAL map ▶ | backend=gemma model={slot.model} "
                    f"chunk {w.window_index + 1}/{len(state.job.windows)} "
                    f"article={state.job.url[:40]} "
                    f"[est in={inp}+out_cap={out_cap}]"
                )
                trace(
                    f"BLOG_SPATIAL map fact-budget ▶ | "
                    f"{w.window_index + 1}/{len(state.job.windows)} "
                    f"facts={target_facts} input_tokens={_fb_in}/{BLOG_SPATIAL_MAP_MAX_TOKENS} "
                    f"projected_out={projected_out}/{out_cap}"
                )
                out, usage = await gemma_rl.post_structured_preacquired(  # type: ignore[union-attr]
                    slot,
                    map_system,
                    prompt,
                    MapWindowResponse,
                    label=f"map/{state.job.job_id[:16]}/w{w.window_index}",
                    client=http_client,
                    max_tokens=resolve_gemma_map_max_output_tokens(
                        inp, projected_out=projected_out
                    ),
                )
                return out, usage

        async def _run_gemma_map_fixed_minute(
            work: list[tuple[_ArticleMapState, TokenWindowChunk]],
        ) -> None:
            """
            Общая очередь MAP-чанков; каждую UTC-минуту — до двух полных батчей
            (по одному на модель, до safety cap), без привязки задач к модели.
            """
            assert gemma_rl is not None
            slots = gemma_rl.model_slots
            if not slots:
                return

            @dataclass
            class _MapWork:
                state: _ArticleMapState
                window: TokenWindowChunk
                est_tokens: int

            q: deque[_MapWork] = deque()
            for st, w in work:
                prompt = _prompt_for_window(st.job, w)
                inp, out_cap, total = gemma_rl.estimate_budget(
                    _map_system_for_job(st.job), prompt, MapWindowResponse
                )
                # estimate_budget already uses adaptive out_cap; keep total for TPM pack.
                _ = (inp, out_cap)
                q.append(_MapWork(state=st, window=w, est_tokens=total))

            cap = max(1000, GEMMA_TARGET_TPM_SAFETY_CAP)
            hard_tpm = GEMMA_MAX_TPM
            n_slots = len(slots)

            def _rpm_for_slot(slot: GemmaModelSlot) -> int:
                if slot.label == "primary":
                    return max(1, GEMMA_PRIMARY_MAX_RPM)
                if slot.label == "fallback":
                    return max(1, GEMMA_FALLBACK_MAX_RPM)
                return max(1, GEMMA_MAX_RPM)

            rpm_limits = [_rpm_for_slot(s) for s in slots]

            def _pack_minute_batches() -> tuple[list[list[_MapWork]], list[int]]:
                batches: list[list[_MapWork]] = [[] for _ in range(n_slots)]
                batch_ests = [0] * n_slots
                while q:
                    item = q[0]
                    est = item.est_tokens
                    best: int | None = None
                    for idx in range(n_slots):
                        if len(batches[idx]) >= rpm_limits[idx]:
                            continue
                        new_total = batch_ests[idx] + est
                        if batches[idx] and new_total > cap:
                            continue
                        if not batches[idx] and est > cap:
                            best = idx
                            break
                        if new_total <= cap or (not batches[idx] and est > hard_tpm):
                            if best is None or batch_ests[idx] < batch_ests[best]:
                                best = idx
                    if best is None:
                        break
                    q.popleft()
                    batches[best].append(item)
                    batch_ests[best] += est
                return batches, batch_ests

            async def _fire_batch(
                slot: GemmaModelSlot,
                batch: list[_MapWork],
                batch_est: int,
                *,
                align_sleep: float,
            ) -> None:
                if not batch:
                    return
                trace(
                    f"BLOG_SPATIAL gemma minute batch ▶ | model={slot.model} "
                    f"chunks={len(batch)} est_tpm≈{batch_est} "
                    f"cap={cap} align_sleep={align_sleep:.1f}s "
                    f"dispatch_window={MAP_DISPATCH_WINDOW_SEC:.0f}s"
                )

                async def _one(item: _MapWork) -> MapWindowResponse | None:
                    out, _usage = await _map_gemma_chunk_preacquired(
                        item.state, item.window, slot, http_client=http_client
                    )
                    if out is not None:
                        return out
                    # Instant failover: the primary attempt (incl. its own
                    # internal retries + Flash-Lite fallback in gemma_client.py)
                    # is already exhausted — try the other pool model once
                    # immediately rather than accepting the loss for this window.
                    alt = next((s for s in slots if s is not slot), None)
                    if alt is None:
                        return None
                    trace(
                        f"BLOG_SPATIAL gemma instant failover ▶ | "
                        f"{slot.model} → {alt.model} | "
                        f"chunk {item.window.window_index + 1}/{len(item.state.job.windows)}"
                    )
                    out, _usage = await _map_gemma_chunk_preacquired(
                        item.state, item.window, alt, http_client=http_client
                    )
                    return out

                # Paced dispatcher: only meaningful when a batch has MORE items
                # than can run concurrently at once (MAX_CONCURRENT_MAP_REQUESTS).
                # Within that limit, spacing launches adds pure critical-path
                # latency for zero rate-limit benefit: _pack_minute_batches
                # already sized this batch to fit the model's TPM/RPM safety
                # cap for the WHOLE minute regardless of when within it the
                # requests fire, and map_sem already caps concurrent in-flight
                # calls — nothing here needs an artificial sleep to stay safe.
                # BUG FIXED: this used to apply delay = WINDOW/n unconditionally
                # for any n > 1, even n=2 against concurrency=4 (the common case
                # observed every run) — pure added wall time with no TPM benefit,
                # a real contributor to the MAP-phase latency this audit targets.
                n = len(batch)
                delay = (
                    MAP_DISPATCH_WINDOW_SEC / n
                    if n > MAX_CONCURRENT_MAP_REQUESTS
                    else 0.0
                )
                tasks: list[asyncio.Task] = []
                for i, item in enumerate(batch):
                    tasks.append(asyncio.create_task(_one(item)))
                    if i + 1 < n and delay > 0:
                        await asyncio.sleep(delay)
                results = await asyncio.gather(*tasks)
                if not GEMMA_MAP_FIXED_MINUTE_PACING:
                    usage_est = [it.est_tokens for it in batch]
                    await gemma_rl.reconcile_batch_usage(  # type: ignore[union-attr]
                        slot,
                        usage_est,
                        batch_est,
                    )
                for item, out in zip(batch, results):
                    await _on_chunk_done(
                        item.state,
                        item.window.window_index,
                        out,
                    )

            while q:
                slept = await await_next_minute_window()
                batches, batch_ests = _pack_minute_batches()
                if not any(batches):
                    if q:
                        lone = q.popleft()
                        batches[0] = [lone]
                        batch_ests[0] = lone.est_tokens
                    else:
                        break
                await asyncio.gather(
                    *[
                        _fire_batch(
                            slots[i],
                            batches[i],
                            batch_ests[i],
                            align_sleep=slept if i == 0 else 0.0,
                        )
                        for i in range(n_slots)
                    ]
                )

        async def _one(
            item: tuple[_ArticleMapState, TokenWindowChunk],
            _slot: GemmaModelSlot,
        ) -> tuple[MapWindowResponse | None, int]:
            st, w = item
            return await _map_gemma_chunk_preacquired(
                st, w, _slot, http_client=http_client
            )

        async def _dispatch_group(
            items: list[tuple[_ArticleMapState, TokenWindowChunk]],
            slot: GemmaModelSlot,
            reserved_tpm: int,
            event: list | None = None,
        ) -> None:
            """One (slot, k)-admission group from a single wave, run as its
            OWN background task — see BUG FIXED note on _run_gemma_map_waves
            below for why this must not be awaited inline by the wave loop.

            ``event`` is the SPECIFIC reservation this group holds in the
            slot's limiter (from acquire_parallel_wave). It must be passed
            straight through to reconcile — DEEP AUDIT FIX: with groups now
            running concurrently, a later wave can admit a second group onto
            the SAME slot's limiter before this one's HTTP call finishes.
            Reconciling by deque position ("the last reservation") then
            corrupts whichever group happens to be last at the moment THIS
            one's response lands — not necessarily this group's own entry —
            which silently reinflates or deflates an unrelated in-flight
            reservation instead of this one."""
            assert gemma_rl is not None
            try:
                results = await asyncio.gather(*[_one(item, slot) for item in items])
            except Exception as exc:
                trace(
                    f"BLOG_SPATIAL map ✗ | dispatch group failed | {slot.model} | {exc}"
                )
                results = [(None, 0) for _ in items]
            real_usage = [usage for _out, usage in results]
            await gemma_rl.reconcile_batch_usage(
                slot, real_usage, reserved_tpm, event=event
            )
            for (st, w), (out, _usage) in zip(items, results):
                await _on_chunk_done(st, w.window_index, out)

        async def _run_gemma_map_waves(
            work: list[tuple[_ArticleMapState, TokenWindowChunk]],
        ) -> None:
            """
            BUG FIXED: this used to `await asyncio.gather(...)` on the WHOLE
            wave (every slot's admitted items together) before requesting the
            next wave — so a fast model's slot, done in e.g. 25s, sat
            completely idle waiting for a slow model's call in the SAME wave
            (e.g. 65s) to finish before this loop would even ASK for more
            work on the fast slot's behalf. Confirmed live: up to ~30s of
            dead time on the faster slot between waves, while AI Studio still
            showed it disproportionately closer to its own TPM ceiling than
            the slower model — the slow model's wall-clock latency, not its
            token budget, was the real constraint, and the old lockstep wave
            barrier wasted the fast model's spare capacity waiting on it.

            Now each (slot, k) admission group from `acquire_parallel_wave`
            runs as its own independent background task (`_dispatch_group`);
            the loop immediately tries to admit the NEXT wave instead of
            waiting for any group to finish. This is safe without any extra
            bookkeeping: `try_acquire_parallel` reserves each group's TPM/RPM
            in the limiter the moment it's admitted (before the HTTP call
            even starts), so a later `acquire_parallel_wave` call already
            sees that reservation and won't over-admit a still-busy slot; the
            existing `map_sem` semaphore (MAX_CONCURRENT_MAP_REQUESTS) still
            caps how many HTTP calls are actually in flight process-wide.
            """
            assert gemma_rl is not None
            pos = 0
            in_flight: list[asyncio.Task[None]] = []
            while pos < len(work):
                batch = work[pos : pos + map_concurrency]
                # DEEP AUDIT FIX: this used to call estimate_request_tokens()
                # with no max_output_tokens, so it fell back to the FLAT
                # resolve_gemma_map_max_output_tokens(inp) — the pre-Step-1
                # worst-case cap — even after Step 1 taught the ACTUAL call
                # (_map_gemma_chunk_preacquired, below) to size max_tokens off
                # dynamic_target_facts()'s projected_out. The two diverged:
                # the wave-admission reservation stayed pinned at inp+4096
                # (matches the audit's est_tpm=8248 exactly: ~4152 input +
                # 4096 flat cap) while real usage shrank toward inp+projected
                # out. The sliding window filled up on the STALE estimate long
                # before real TPM usage ever approached the ceiling — this is
                # the "wave full at 8.2k, cloud shows 4.6k" lockout. Mirror
                # the same projected_out-aware cap used for the real call.
                ests = []
                for st, w in batch:
                    system_ = _map_system_for_job(st.job)
                    prompt_ = _prompt_for_window(st.job, w)
                    inp, _flat_cap, _ = gemma_rl.estimate_budget(
                        system_, prompt_, MapWindowResponse
                    )
                    _fb_in, _target_facts, projected_out = dynamic_target_facts(w.body)
                    out_cap = resolve_gemma_map_max_output_tokens(
                        inp, projected_out=projected_out
                    )
                    ests.append(inp + out_cap)
                # Dual-basket: acquire_parallel_wave reserves proportionally
                # across the pool based on each slot's CURRENT free TPM/RPM
                # in the sliding window — plan = [(slot, k, reserved), ...],
                # first k items of `batch` go to plan[0]'s slot, next k to
                # plan[1]'s slot, etc. The part exceeding the primary
                # (31B)'s free capacity lands instantly on the next slot
                # (26B) instead of waiting for the primary to free up.
                plan = await gemma_rl.acquire_parallel_wave(
                    ests,
                    max_parallel=map_concurrency,
                )
                total_k = sum(k for _slot, k, _reserved, _event in plan)
                if total_k <= 0:
                    if in_flight:
                        # An in-flight group finishing frees real TPM/RPM
                        # room (via reconcile) — wait for that instead of a
                        # blind fixed sleep when we already have work moving.
                        done, still_pending = await asyncio.wait(
                            in_flight,
                            timeout=0.25,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        in_flight = list(still_pending)
                        for t in done:
                            t.result()
                    else:
                        await asyncio.sleep(0.25)
                    continue

                wave = batch[:total_k]
                idx = 0
                for slot, k, reserved_tpm, event in plan:
                    group_items = wave[idx : idx + k]
                    idx += k
                    in_flight.append(
                        asyncio.create_task(
                            _dispatch_group(group_items, slot, reserved_tpm, event)
                        )
                    )
                pos += total_k

                still_running: list[asyncio.Task[None]] = []
                for t in in_flight:
                    if t.done():
                        t.result()  # propagate any unexpected exception now
                    else:
                        still_running.append(t)
                in_flight = still_running

            if in_flight:
                await asyncio.gather(*in_flight)

        map_tasks: list[asyncio.Task[None]] = []
        if gemma_rl is not None:
            work_items: list[tuple[_ArticleMapState, TokenWindowChunk]] = []
            for state in states:
                for w in state.job.windows:
                    work_items.append((state, w))
            map_tasks.append(
                asyncio.create_task(
                    _run_gemma_map_fixed_minute(work_items)
                    if GEMMA_MAP_FIXED_MINUTE_PACING
                    else _run_gemma_map_waves(work_items)
                )
            )

        if map_tasks:
            await asyncio.gather(*map_tasks)
        await reduce_queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    return finals


def build_article_map_reduce_job(
    annotated_markdown: str,
    *,
    title: str,
    url: str,
    all_figure_ids: list[str] | None = None,
    figure_registry: object | None = None,
    source_kind: str = "article",
) -> tuple[MapReduceArticleJob, list[TokenWindowChunk]] | None:
    """Chunk + build the MapReduceArticleJob — everything map_reduce_jobs_pooled_async
    needs, but WITHOUT running it. Pure sync CPU work (no LLM calls), factored out
    of map_reduce_summarize_blog_outcome_async so callers can build several jobs
    and submit them together in ONE pooled batch (BATCH POOLING AGGREGATION task)
    instead of one map_reduce_jobs_pooled_async([job]) call per document."""
    body = (annotated_markdown or "").strip()
    if len(body) < 80:
        trace("BLOG_SPATIAL map-reduce ⊘ | annotated body too short")
        return None

    windows = split_annotated_text_by_tokens(
        body,
        title=title,
        all_figure_ids=all_figure_ids or [],
        figure_registry=figure_registry,  # type: ignore[arg-type]
    )
    if not windows:
        windows = [TokenWindowChunk(window_index=0, body=body)]

    from knowledge_engine.ingest.pipeline_audit import pipeline_audit

    joined = "\n\n".join((w.body or "") for w in windows)
    pipeline_audit(
        "Chunk",
        url,
        joined,
        extra=f"windows={len(windows)} source_kind={source_kind}",
    )
    pipeline_audit("MAP", url, body, extra=f"annotated → {len(windows)} windows")

    job = MapReduceArticleJob(
        job_id=url,
        title=title,
        url=url,
        windows=windows,
        all_figure_ids=list(all_figure_ids or []),
        figure_registry=figure_registry,
        source_kind=source_kind or "article",
    )
    return job, windows


async def map_reduce_summarize_blog_outcome_async(
    annotated_markdown: str,
    *,
    title: str,
    url: str,
    all_figure_ids: list[str] | None = None,
    figure_registry: object | None = None,
    source_kind: str = "article",
) -> tuple[MapReduceJobOutcome | None, list[TokenWindowChunk]]:
    built = build_article_map_reduce_job(
        annotated_markdown,
        title=title,
        url=url,
        all_figure_ids=all_figure_ids,
        figure_registry=figure_registry,
        source_kind=source_kind,
    )
    if built is None:
        return None, []
    job, windows = built
    results = await map_reduce_jobs_pooled_async([job])
    return results.get(job.job_id), windows


async def map_reduce_summarize_blog_async(
    annotated_markdown: str,
    *,
    title: str,
    url: str,
    all_figure_ids: list[str] | None = None,
    figure_registry: object | None = None,
    source_kind: str = "article",
) -> FinalArticleSummaryResponse | None:
    outcome, _windows = await map_reduce_summarize_blog_outcome_async(
        annotated_markdown,
        title=title,
        url=url,
        all_figure_ids=all_figure_ids,
        figure_registry=figure_registry,
        source_kind=source_kind,
    )
    return outcome.final if outcome else None


def map_reduce_summarize_blog_outcome(
    annotated_markdown: str,
    *,
    title: str,
    url: str,
    all_figure_ids: list[str] | None = None,
    figure_registry: object | None = None,
    source_kind: str = "article",
) -> tuple[MapReduceJobOutcome | None, list[TokenWindowChunk]]:
    import asyncio

    kwargs = dict(
        annotated_markdown=annotated_markdown,
        title=title,
        url=url,
        all_figure_ids=all_figure_ids,
        figure_registry=figure_registry,
        source_kind=source_kind,
    )
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                map_reduce_summarize_blog_outcome_async(**kwargs),
            ).result()
    except RuntimeError:
        return asyncio.run(map_reduce_summarize_blog_outcome_async(**kwargs))


def map_reduce_summarize_blog(
    annotated_markdown: str,
    *,
    title: str,
    url: str,
    all_figure_ids: list[str] | None = None,
    figure_registry: object | None = None,
    source_kind: str = "article",
) -> FinalArticleSummaryResponse | None:
    outcome, _windows = map_reduce_summarize_blog_outcome(
        annotated_markdown,
        title=title,
        url=url,
        all_figure_ids=all_figure_ids,
        figure_registry=figure_registry,
        source_kind=source_kind,
    )
    return outcome.final if outcome else None


async def summarize_blog_article_spatial_async(
    annotated_markdown: str,
    *,
    title: str,
    url: str,
    all_figure_ids: list[str] | None = None,
) -> BlogArticleSummaryResponse | None:
    final = await map_reduce_summarize_blog_async(
        annotated_markdown,
        title=title,
        url=url,
        all_figure_ids=all_figure_ids,
    )
    if final is None:
        return None
    return final_to_legacy_summary(final)


def summarize_blog_article_spatial(
    annotated_markdown: str,
    *,
    title: str,
    url: str,
    all_figure_ids: list[str] | None = None,
) -> BlogArticleSummaryResponse | None:
    final = map_reduce_summarize_blog(
        annotated_markdown,
        title=title,
        url=url,
        all_figure_ids=all_figure_ids,
    )
    if final is None:
        return None
    return final_to_legacy_summary(final)
