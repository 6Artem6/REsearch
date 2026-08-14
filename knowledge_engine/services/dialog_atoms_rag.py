"""Per-turn knowledge_atoms RAG for tutor dialog (no parent-chunk expand)."""

from __future__ import annotations

import re
from typing import Any

from knowledge_engine.config import (
    DIALOG_ATOMS_ENABLED,
    DIALOG_ATOMS_MIN_SCORE,
    DIALOG_ATOMS_TOP_K,
)
from knowledge_engine.db.knowledge_atoms_schema import (
    COL_SCOPE,
    COL_STATEMENT,
)
from knowledge_engine.schemas.extraction import (
    KnowledgeAtom,
    ScopeType,
    coerce_scope_type,
)
from knowledge_engine.ui.run_log import trace

_CODE_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:код|code|syntax|snippet|функци\w*|function|implement\w*|реализац\w*|"
    r"source\s*code|how\s+to\s+code|api\s*call)\b|"
    r"покажи\s+код|приведи\s+код|пример\s+кода|show\s+(?:me\s+)?(?:the\s+)?code"
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


def detect_code_intent(user_query: str) -> bool:
    """True when the user explicitly asks for code / implementation details."""
    raw = (user_query or "").strip()
    if not raw:
        return False
    return bool(_CODE_INTENT_RE.search(raw))


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
    return out[: max(1, int(cap))]


def format_dialog_atoms_block(atoms: list[KnowledgeAtom]) -> str:
    """Compact fact block for tutor dialog (no parent chunk text)."""
    if not atoms:
        return ""
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
        try:
            out.append(
                KnowledgeAtom(
                    scope=_normalize_dialog_scope(row.get(COL_SCOPE)),
                    statement=stmt[:2000],
                    context_quote=None,
                    source_chunk_ids=[],  # never expand parents in dialog
                )
            )
        except Exception:
            continue
    return out


def retrieve_dialog_knowledge_atoms(
    user_msg: str,
    node: Any,
    curriculum_id: str = "",
    *,
    store: Any | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> str:
    """
    Sync retrieve + filter + format for tutor turn.

    Scoped to mapped node doc_ids when available; never expands rag_chunks.
    """
    if not DIALOG_ATOMS_ENABLED:
        return ""
    focus = (user_msg or "").strip()
    title = str(getattr(node, "title", "") or "").strip()
    query = " ".join(p for p in [title, focus] if p).strip()
    if not query:
        return ""

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
    # Over-fetch slightly so scope filter still fills top_k.
    rows = vs.search_knowledge_atoms(
        query,
        limit=max(cap * 2, cap + 4),
        allowed_doc_ids=allowed,
        min_score=float(floor),
    )
    if not rows and allowed:
        # Fallback: curriculum library atoms if mapped docs empty of hits.
        try:
            from knowledge_engine.services.lecture_rag_source_scope import (
                collect_curriculum_library_urls,
            )

            urls = collect_curriculum_library_urls(cid)
            lib_ids = [
                VectorStore.doc_id_for_url(u) for u in urls if u.startswith("http")
            ]
            if lib_ids:
                rows = vs.search_knowledge_atoms(
                    query,
                    limit=max(cap * 2, cap + 4),
                    allowed_doc_ids=lib_ids,
                    min_score=float(floor),
                )
        except Exception as exc:
            trace(f"DIALOG_ATOMS library fallback skip | {exc}")

    atoms = _rows_to_atoms(rows)
    allow_instance = detect_code_intent(focus)
    filtered = filter_atoms_for_dialog(atoms, allow_instance=allow_instance, limit=cap)
    block = format_dialog_atoms_block(filtered)
    if block:
        trace(
            f"DIALOG_ATOMS ✓ | n={len(filtered)} code_intent={allow_instance} "
            f"scoped={bool(allowed)} query_len={len(query)}"
        )
    return block
