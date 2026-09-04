"""Per-turn knowledge_atoms RAG for tutor dialog (no parent-chunk expand)."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from knowledge_engine.config import (
    DIALOG_ATOMS_ENABLED,
    DIALOG_ATOMS_MIN_SCORE,
    DIALOG_ATOMS_TOP_K,
)
from knowledge_engine.db.knowledge_atoms_schema import (
    COL_SCOPE,
    COL_SOURCE_CHUNK_IDS,
    COL_STATEMENT,
)
from knowledge_engine.schemas.extraction import (
    KnowledgeAtom,
    ScopeType,
    coerce_scope_type,
)
from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import atom_key
from knowledge_engine.ui.run_log import trace

_CODE_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:код|code|syntax|snippet|функци\w*|function|implement\w*|реализац\w*|"
    r"source\s*code|how\s+to\s+code|api\s*call)\b|"
    r"покажи\s+код|приведи\s+код|пример\s+кода|show\s+(?:me\s+)?(?:the\s+)?code"
    r")",
    re.I,
)

# Sticky chip / control stubs that must not dominate deep_analysis RAG query.
_GENERIC_FOCUS_RE = re.compile(
    r"(?:"
    r"задачк\w*\s+со\s+зв[её]здочк|"
    r"deep\s*mastery|"
    r"\[mode:\s*deep_analysis\]|"
    r"^разбери\s+механик|"
    r"^разбери\s+архитектур"
    r")",
    re.I,
)

DIALOG_PREFERRED_SCOPES = frozenset({ScopeType.PRINCIPLE, ScopeType.MECHANIC})
DIALOG_DETAIL_SCOPES = frozenset({ScopeType.INSTANCE})

# Aliases beyond ScopeType enum (future / LLM typos).
_SCOPE_ALIASES_PREFERRED = {
    "CONCEPT": ScopeType.PRINCIPLE,
    "PRINCIPLES": ScopeType.PRINCIPLE,
}
_SCOPE_ALIASES_DETAIL = {
    "IMPLEMENTATION": ScopeType.INSTANCE,
    "CODE_DETAILS": ScopeType.INSTANCE,
    "CODE": ScopeType.INSTANCE,
}


@dataclass
class DialogAtomsRetrieveResult:
    """Retrieve payload for tutor / deep_analysis."""

    block: str = ""
    atom_keys: list[str] = field(default_factory=list)
    atom_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    rag_exhausted: bool = False
    unseen_count: int = 0


def detect_code_intent(user_query: str) -> bool:
    """True when the user explicitly asks for code / implementation details."""
    raw = (user_query or "").strip()
    if not raw:
        return False
    return bool(_CODE_INTENT_RE.search(raw))


def is_generic_dialog_focus(text: str) -> bool:
    """True for chip stubs that should not drive sticky RAG queries."""
    raw = (text or "").strip()
    if not raw:
        return True
    if len(raw) < 12:
        return True
    return bool(_GENERIC_FOCUS_RE.search(raw))


def build_dialog_atoms_query(
    node: Any,
    user_msg: str,
    *,
    focus_hint: str = "",
) -> str:
    """
    Build retrieval query: prefer dialog/subconcept focus over sticky chip text.
    """
    title = str(getattr(node, "title", "") or "").strip()
    hint = (focus_hint or "").strip()
    msg = (user_msg or "").strip()
    parts: list[str] = []
    if title:
        parts.append(title)
    if hint and not is_generic_dialog_focus(hint):
        parts.append(hint)
    if msg and not is_generic_dialog_focus(msg):
        if msg.lower() != hint.lower():
            parts.append(msg)
    elif hint and hint not in parts:
        parts.append(hint)
    return " ".join(p for p in parts if p).strip()


def _normalize_dialog_scope(raw: object) -> ScopeType:
    if isinstance(raw, ScopeType):
        return raw
    text = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in _SCOPE_ALIASES_PREFERRED:
        return _SCOPE_ALIASES_PREFERRED[text]
    if text in _SCOPE_ALIASES_DETAIL:
        return _SCOPE_ALIASES_DETAIL[text]
    return coerce_scope_type(raw)


def filter_atoms_for_dialog(
    atoms: list[KnowledgeAtom],
    *,
    allow_instance: bool,
    limit: int | None = None,
) -> list[KnowledgeAtom]:
    """
    Prefer PRINCIPLE/MECHANIC; INSTANCE only when ``allow_instance`` (code intent).
    """
    preferred: list[KnowledgeAtom] = []
    details: list[KnowledgeAtom] = []
    for atom in atoms:
        scope = (
            atom.scope
            if isinstance(atom.scope, ScopeType)
            else _normalize_dialog_scope(atom.scope)
        )
        if scope in DIALOG_PREFERRED_SCOPES:
            preferred.append(
                atom
                if atom.scope == scope
                else atom.model_copy(update={"scope": scope})
            )
        elif scope in DIALOG_DETAIL_SCOPES and allow_instance:
            details.append(
                atom
                if atom.scope == scope
                else atom.model_copy(update={"scope": scope})
            )
    out = preferred + details
    cap = limit if limit is not None else DIALOG_ATOMS_TOP_K
    if limit is None:
        return out[: max(1, int(cap))]
    # Explicit None-like "no cap" for exclude pipeline: pass a huge limit.
    return out[: max(1, int(cap))]


def filter_atoms_for_dialog_uncapped(
    atoms: list[KnowledgeAtom],
    *,
    allow_instance: bool,
) -> list[KnowledgeAtom]:
    """Same scope preference as dialog filter, without top_k truncation."""
    preferred: list[KnowledgeAtom] = []
    details: list[KnowledgeAtom] = []
    for atom in atoms:
        scope = (
            atom.scope
            if isinstance(atom.scope, ScopeType)
            else _normalize_dialog_scope(atom.scope)
        )
        if scope in DIALOG_PREFERRED_SCOPES:
            preferred.append(
                atom
                if atom.scope == scope
                else atom.model_copy(update={"scope": scope})
            )
        elif scope in DIALOG_DETAIL_SCOPES and allow_instance:
            details.append(
                atom
                if atom.scope == scope
                else atom.model_copy(update={"scope": scope})
            )
    return preferred + details


def format_dialog_atoms_block(
    atoms: list[KnowledgeAtom],
    *,
    cite_r_index: bool = False,
) -> str:
    """Compact fact block for tutor dialog (no parent chunk text).

    When ``cite_r_index`` is True (Deep Analysis), label facts as ``[R1]…[Rn]``
    so the isolated prompt can require inline RAG citations.
    """
    if not atoms:
        return ""
    if cite_r_index:
        lines = [
            "### RAG MATERIAL / dialog atoms (cite inline as [R1]…[Rn])",
            "Each line is a retrieved knowledge atom from mapped sources / "
            "map-reduce extracts. Cite the same [Rx] when you use that fact.",
        ]
        idx = 0
        for atom in atoms:
            stmt = (atom.statement or "").strip()
            if not stmt:
                continue
            idx += 1
            scope = (
                atom.scope.value
                if isinstance(atom.scope, ScopeType)
                else _normalize_dialog_scope(atom.scope).value
            )
            lines.append(f"[R{idx}] ({scope}): {stmt}")
        if idx == 0:
            return ""
        return "\n".join(lines)

    lines = ["### dialog_knowledge_atoms"]
    for atom in atoms:
        stmt = (atom.statement or "").strip()
        if not stmt:
            continue
        scope = (
            atom.scope.value
            if isinstance(atom.scope, ScopeType)
            else _normalize_dialog_scope(atom.scope).value
        )
        lines.append(f"[ФАКТ ({scope})]: {stmt}")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def _rows_to_atoms(rows: list[dict[str, Any]]) -> list[KnowledgeAtom]:
    out: list[KnowledgeAtom] = []
    for row in rows:
        stmt = str(row.get(COL_STATEMENT) or "").strip()
        if len(stmt) < 8:
            continue
        chunk_ids: list[str] = []
        raw_chunks = row.get(COL_SOURCE_CHUNK_IDS)
        if isinstance(raw_chunks, str) and raw_chunks.strip():
            s = raw_chunks.strip()
            if s.startswith("["):
                try:
                    import json

                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        chunk_ids = [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    chunk_ids = [s]
            else:
                chunk_ids = [s]
        elif isinstance(raw_chunks, (list, tuple, set)):
            chunk_ids = [str(x).strip() for x in raw_chunks if str(x).strip()]
        try:
            out.append(
                KnowledgeAtom(
                    scope=_normalize_dialog_scope(row.get(COL_SCOPE)),
                    statement=stmt[:2000],
                    context_quote=None,
                    source_chunk_ids=chunk_ids,
                )
            )
        except Exception:
            continue
    return out


def _exclude_atoms(
    atoms: list[KnowledgeAtom],
    exclude_keys: set[str],
) -> tuple[list[KnowledgeAtom], list[KnowledgeAtom]]:
    """Split into (unseen, excluded_hits). Never lower score to refill."""
    if not exclude_keys:
        return list(atoms), []
    unseen: list[KnowledgeAtom] = []
    excluded: list[KnowledgeAtom] = []
    for atom in atoms:
        key = atom_key(atom.statement or "")
        if key and key in exclude_keys:
            excluded.append(atom)
        else:
            unseen.append(atom)
    return unseen, excluded


def _row_stable_ids(row: dict[str, Any]) -> tuple[str, list[str]]:
    from knowledge_engine.db.knowledge_atoms_schema import COL_ID
    from knowledge_engine.services.vector_store import VectorStore

    rid = str(row.get(COL_ID) or "").strip()
    chunks = VectorStore._row_chunk_ids(row)
    if not chunks and rid:
        chunks = [rid]
    return rid, chunks


def retrieve_dialog_knowledge_atoms_detailed(
    user_msg: str,
    node: Any,
    curriculum_id: str = "",
    *,
    store: Any | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    force_allow_instance: bool = False,
    cite_r_index: bool = False,
    exclude_keys: list[str] | set[str] | None = None,
    exclude_chunk_ids: list[str] | set[str] | None = None,
    exclude_atom_ids: list[str] | set[str] | None = None,
    focus_hint: str = "",
    query_override: str = "",
    lambda_mult: float = 1.0,
    query_noise: float = 0.0,
    stochastic_sample: bool = False,
    pool_mult: int = 3,
    rng_seed: int | None = None,
) -> DialogAtomsRetrieveResult:
    """
    Sync retrieve + filter + format for tutor turn.

    Under deep_analysis: pass ``exclude_chunk_ids`` / diversity knobs from
    SessionMemory. Do NOT lower ``min_score`` when excluded; if no unseen
    atoms remain → ``rag_exhausted``.
    """
    if not DIALOG_ATOMS_ENABLED:
        return DialogAtomsRetrieveResult(rag_exhausted=False)

    query = (query_override or "").strip() or build_dialog_atoms_query(
        node, user_msg, focus_hint=focus_hint
    )
    if not query:
        return DialogAtomsRetrieveResult(rag_exhausted=False)

    from knowledge_engine.services.vector_store import VectorStore

    vs = store or VectorStore()
    allowed: list[str] | None = None
    cid = (curriculum_id or "").strip()
    if cid and node is not None:
        try:
            from knowledge_engine.services.lecture_rag_source_scope import (
                mapped_doc_ids_for_node,
            )

            mapped = mapped_doc_ids_for_node(cid, node)
            if mapped:
                allowed = list(mapped)
        except Exception as exc:
            trace(f"DIALOG_ATOMS scope skip | {exc}")

    cap = top_k if top_k is not None else DIALOG_ATOMS_TOP_K
    floor = min_score if min_score is not None else DIALOG_ATOMS_MIN_SCORE
    excl = {str(k).strip() for k in (exclude_keys or []) if str(k).strip()}
    excl_chunks = [str(k).strip() for k in (exclude_chunk_ids or []) if str(k).strip()]
    excl_atom_ids = [str(k).strip() for k in (exclude_atom_ids or []) if str(k).strip()]
    fetch_limit = max(cap * 4, cap + len(excl) + len(excl_chunks) + 8)

    search_kwargs: dict[str, Any] = {
        "limit": fetch_limit,
        "allowed_doc_ids": allowed,
        "min_score": float(floor),
        "exclude_ids": excl_atom_ids or None,
        "exclude_chunk_ids": excl_chunks or None,
        "lambda_mult": float(lambda_mult),
        "query_noise": float(query_noise),
        "stochastic_sample": bool(stochastic_sample),
        "pool_mult": int(pool_mult),
        "rng_seed": rng_seed,
    }
    # This function only runs via LangGraph's sync-node thread dispatch (no
    # event loop of its own) — a single local asyncio.run() per Qdrant call
    # is the legitimate sync/async boundary here, not a nested-loop bridge.
    rows = asyncio.run(vs.search_knowledge_atoms(query, **search_kwargs))
    if not rows and allowed:
        try:
            from knowledge_engine.services.lecture_rag_source_scope import (
                collect_curriculum_library_urls,
            )

            urls = collect_curriculum_library_urls(cid)
            lib_ids = [
                VectorStore.doc_id_for_url(u) for u in urls if u.startswith("http")
            ]
            if lib_ids:
                search_kwargs["allowed_doc_ids"] = lib_ids
                rows = asyncio.run(vs.search_knowledge_atoms(query, **search_kwargs))
        except Exception as exc:
            trace(f"DIALOG_ATOMS library fallback skip | {exc}")

    row_meta: list[tuple[dict[str, Any], KnowledgeAtom]] = []
    for row in rows:
        atoms = _rows_to_atoms([row])
        if not atoms:
            continue
        row_meta.append((row, atoms[0]))

    allow_instance = bool(force_allow_instance) or detect_code_intent(
        " ".join(p for p in [focus_hint, user_msg] if p)
    )
    filtered_pairs: list[tuple[dict[str, Any], KnowledgeAtom]] = []
    for row, atom in row_meta:
        scope = (
            atom.scope
            if isinstance(atom.scope, ScopeType)
            else _normalize_dialog_scope(atom.scope)
        )
        if scope in DIALOG_PREFERRED_SCOPES:
            filtered_pairs.append((row, atom))
        elif scope in DIALOG_DETAIL_SCOPES and allow_instance:
            filtered_pairs.append((row, atom))

    unseen_pairs: list[tuple[dict[str, Any], KnowledgeAtom]] = []
    excluded_hits = 0
    for row, atom in filtered_pairs:
        key = atom_key(atom.statement or "")
        if excl and key and key in excl:
            excluded_hits += 1
            continue
        unseen_pairs.append((row, atom))

    selected_pairs = unseen_pairs[: max(0, int(cap))]
    rag_exhausted = (
        bool(excl or excl_chunks or excl_atom_ids) and len(selected_pairs) == 0
    )

    selected = [a for _, a in selected_pairs]
    keys = [
        atom_key(a.statement or "") for a in selected if (a.statement or "").strip()
    ]
    keys = [k for k in keys if k]
    atom_ids: list[str] = []
    chunk_ids: list[str] = []
    for row, _atom in selected_pairs:
        rid, chunks = _row_stable_ids(row)
        if rid:
            atom_ids.append(rid)
        for c in chunks:
            if c and c not in chunk_ids:
                chunk_ids.append(c)

    block = format_dialog_atoms_block(selected, cite_r_index=cite_r_index)
    if block or rag_exhausted or excl or excl_chunks:
        trace(
            f"DIALOG_ATOMS ✓ | n={len(selected)} unseen={len(unseen_pairs)} "
            f"excluded_keys={excluded_hits} excl_chunks={len(excl_chunks)} "
            f"exhausted={rag_exhausted} λ={lambda_mult:.2f} "
            f"noise={query_noise:.3f} stochastic={stochastic_sample} "
            f"code_intent={allow_instance} cite_r={cite_r_index} "
            f"scoped={bool(allowed)} query_len={len(query)}"
        )
    return DialogAtomsRetrieveResult(
        block=block,
        atom_keys=keys,
        atom_ids=atom_ids,
        chunk_ids=chunk_ids,
        rag_exhausted=rag_exhausted,
        unseen_count=len(unseen_pairs),
    )


def retrieve_dialog_knowledge_atoms(
    user_msg: str,
    node: Any,
    curriculum_id: str = "",
    *,
    store: Any | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    force_allow_instance: bool = False,
    cite_r_index: bool = False,
    exclude_keys: list[str] | set[str] | None = None,
    exclude_chunk_ids: list[str] | set[str] | None = None,
    exclude_atom_ids: list[str] | set[str] | None = None,
    focus_hint: str = "",
    query_override: str = "",
    lambda_mult: float = 1.0,
    query_noise: float = 0.0,
    stochastic_sample: bool = False,
    pool_mult: int = 3,
    rng_seed: int | None = None,
) -> str:
    """Backward-compatible wrapper — returns formatted block only."""
    result = retrieve_dialog_knowledge_atoms_detailed(
        user_msg,
        node,
        curriculum_id,
        store=store,
        top_k=top_k,
        min_score=min_score,
        force_allow_instance=force_allow_instance,
        cite_r_index=cite_r_index,
        exclude_keys=exclude_keys,
        exclude_chunk_ids=exclude_chunk_ids,
        exclude_atom_ids=exclude_atom_ids,
        focus_hint=focus_hint,
        query_override=query_override,
        lambda_mult=lambda_mult,
        query_noise=query_noise,
        stochastic_sample=stochastic_sample,
        pool_mult=pool_mult,
        rng_seed=rng_seed,
    )
    return result.block
