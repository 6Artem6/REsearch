"""Selection Explainer context: Target Anchor + invariants + causal facts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from knowledge_engine.config import (
    EXPLAIN_ANCHOR_FALLBACK_ENABLED,
    EXPLAIN_ANCHOR_FALLBACK_TOP_K,
    EXPLAIN_ATOMS_ENABLED,
    EXPLAIN_ATOMS_MIN_SCORE,
    EXPLAIN_ATOMS_TOP_K,
    EXPLAIN_CAUSAL_FACTS_TOP_K,
    LIGHT_RAG_MIN_COSINE_SIM,
)
from knowledge_engine.db.knowledge_atoms_schema import (
    COL_DOC_ID,
    COL_SCOPE,
    COL_STATEMENT,
)
from knowledge_engine.db.rag_chunks_schema import (
    COL_CHUNK_INDEX,
    COL_CHUNK_TEXT,
    COL_CHUNKS_IN_DOC,
)
from knowledge_engine.db.rag_chunks_schema import COL_DOC_ID as RAG_COL_DOC_ID
from knowledge_engine.db.rag_chunks_schema import (
    COL_TITLE,
    COL_URL,
)
from knowledge_engine.schemas.extraction import KnowledgeAtom, ScopeType
from knowledge_engine.services.dialog_atoms_rag import (
    _normalize_dialog_scope,
    detect_code_intent,
    filter_atoms_for_dialog,
)
from knowledge_engine.ui.run_log import trace

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_DOC_ID_BOOST = 0.15


@dataclass
class ExplainContextBundle:
    """Additive context blocks for node selection explain."""

    resolved_r_chunks: list[dict[str, Any]] = field(default_factory=list)
    anchor_block: str = ""
    invariants_block: str = ""
    causal_block: str = ""
    prefer_doc_ids: list[str] = field(default_factory=list)


def _token_set(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if len(w) > 2}


def _mapped_doc_ids(curriculum_id: str, node: Any) -> list[str] | None:
    cid = (curriculum_id or "").strip()
    if not cid or node is None:
        return None
    try:
        from knowledge_engine.services.lecture_rag_source_scope import (
            mapped_doc_ids_for_node,
        )

        mapped = mapped_doc_ids_for_node(cid, node)
        return list(mapped) if mapped else None
    except Exception as exc:
        trace(f"EXPLAIN_BUNDLE scope skip | {exc}")
        return None


def _prefer_doc_ids_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for row in chunks:
        did = str(row.get("doc_id") or "").strip()
        if did and did not in out:
            out.append(did)
    return out


def _format_anchor_block(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    from knowledge_engine.services.lecture_rag_context import (
        format_highlight_rag_chunks_block,
    )

    body = format_highlight_rag_chunks_block(rows)
    if not body:
        return ""
    return f"### target_anchor\n{body}"


def format_explain_invariants_block(atoms: list[KnowledgeAtom]) -> str:
    if not atoms:
        return ""
    lines = ["### fundamental_invariants"]
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


def format_causal_facts_block(facts: list[str]) -> str:
    clean = [f.strip() for f in facts if (f or "").strip()]
    if not clean:
        return ""
    lines = ["### causal_facts"]
    for fact in clean:
        lines.append(f"- {fact}")
    return "\n".join(lines)


def _rows_to_atoms_keep_score(
    rows: list[dict[str, Any]],
) -> list[tuple[float, KnowledgeAtom, str]]:
    out: list[tuple[float, KnowledgeAtom, str]] = []
    for row in rows:
        stmt = str(row.get(COL_STATEMENT) or "").strip()
        if len(stmt) < 8:
            continue
        try:
            atom = KnowledgeAtom(
                scope=_normalize_dialog_scope(row.get(COL_SCOPE)),
                statement=stmt[:2000],
                context_quote=None,
                source_chunk_ids=[],
            )
        except Exception:
            continue
        score = float(row.get("_score") or 0.0)
        did = str(row.get(COL_DOC_ID) or "").strip()
        out.append((score, atom, did))
    return out


def retrieve_explain_invariants(
    query: str,
    *,
    curriculum_id: str = "",
    node: Any = None,
    prefer_doc_ids: list[str] | None = None,
    store: Any | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> str:
    """PRINCIPLE/MECHANIC atoms for Explainer (limit 2–3)."""
    if not EXPLAIN_ATOMS_ENABLED:
        return ""
    q = (query or "").strip()
    if not q:
        return ""
    # Require curriculum scope so unit tests / bare payload builds stay offline.
    if not (curriculum_id or "").strip():
        return ""

    from knowledge_engine.services.vector_store import VectorStore

    try:
        vs = store or VectorStore()
    except Exception as exc:
        trace(f"EXPLAIN_ATOMS store skip | {exc}")
        return ""

    allowed = _mapped_doc_ids(curriculum_id, node)
    cap = top_k if top_k is not None else EXPLAIN_ATOMS_TOP_K
    floor = min_score if min_score is not None else EXPLAIN_ATOMS_MIN_SCORE
    prefer = {d for d in (prefer_doc_ids or []) if d}

    try:
        rows = vs.search_knowledge_atoms(
            q,
            limit=max(cap * 3, cap + 4),
            allowed_doc_ids=allowed,
            min_score=float(floor),
        )
    except Exception as exc:
        trace(f"EXPLAIN_ATOMS search skip | {exc}")
        return ""

    if not rows and allowed:
        try:
            from knowledge_engine.services.lecture_rag_source_scope import (
                collect_curriculum_library_urls,
            )

            urls = collect_curriculum_library_urls((curriculum_id or "").strip())
            lib_ids = [
                VectorStore.doc_id_for_url(u) for u in urls if u.startswith("http")
            ]
            if lib_ids:
                rows = vs.search_knowledge_atoms(
                    q,
                    limit=max(cap * 3, cap + 4),
                    allowed_doc_ids=lib_ids,
                    min_score=float(floor),
                )
        except Exception as exc:
            trace(f"EXPLAIN_ATOMS library fallback skip | {exc}")

    scored = _rows_to_atoms_keep_score(rows)
    if prefer:
        scored = [
            (sc + (_DOC_ID_BOOST if did in prefer else 0.0), atom, did)
            for sc, atom, did in scored
        ]
        scored.sort(key=lambda t: t[0], reverse=True)

    atoms = [atom for _, atom, _ in scored]
    allow_instance = detect_code_intent(q)
    filtered = filter_atoms_for_dialog(atoms, allow_instance=allow_instance, limit=cap)
    block = format_explain_invariants_block(filtered)
    if block:
        trace(
            f"EXPLAIN_ATOMS ✓ | n={len(filtered)} code_intent={allow_instance} "
            f"scoped={bool(allowed)} prefer={len(prefer)}"
        )
    return block


def _rag_rows_to_inspector(
    rows: list[dict[str, Any]], *, start_index: int = 1
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = str(row.get(COL_CHUNK_TEXT) or "").strip()
        if len(text) < 20:
            continue
        idx = start_index + i
        out.append(
            {
                "rag_id": f"AF{idx}",
                "title": str(row.get(COL_TITLE) or f"anchor_chunk_{idx}")[:200],
                "url": str(row.get(COL_URL) or "").strip(),
                "chunk_index": int(row.get(COL_CHUNK_INDEX) or 0),
                "chunks_in_doc": int(row.get(COL_CHUNKS_IN_DOC) or 0),
                "chunk_text": text[:6000],
                "doc_id": str(row.get(RAG_COL_DOC_ID) or "").strip()[:64],
            }
        )
    return out


def retrieve_anchor_fallback_chunks(
    query: str,
    *,
    curriculum_id: str = "",
    node: Any = None,
    store: Any | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Vector search on rag_chunks when lecture [R*] lookup is empty."""
    if not EXPLAIN_ANCHOR_FALLBACK_ENABLED:
        return []
    q = (query or "").strip()
    if not q:
        return []
    allowed = _mapped_doc_ids(curriculum_id, node)
    if allowed is None and not (curriculum_id or "").strip():
        return []

    from knowledge_engine.services.vector_store import VectorStore

    try:
        vs = store or VectorStore()
    except Exception as exc:
        trace(f"EXPLAIN_ANCHOR store skip | {exc}")
        return []

    cap = top_k if top_k is not None else EXPLAIN_ANCHOR_FALLBACK_TOP_K
    try:
        rows = vs.search_rag_chunk_rows(
            q,
            limit=max(cap * 3, 8),
            allowed_doc_ids=allowed,
        )
    except Exception as exc:
        trace(f"EXPLAIN_ANCHOR search skip | {exc}")
        return []

    inspector = _rag_rows_to_inspector(rows[: max(1, int(cap))])
    if inspector:
        trace(f"EXPLAIN_ANCHOR fallback ✓ | n={len(inspector)}")
    return inspector


def _light_rag_causal_sync(query: str, limit: int) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    try:
        from knowledge_engine.src.memory.light_rag import LightRAG

        rag = LightRAG()
        if rag._table_name not in rag._db.table_names():
            return []
        if rag.count_indexed_rows_sync() == 0:
            return []
        vector = rag._embed_sync(q)
        hits = rag._search_rows_sync(
            vector,
            limit,
            kinds=frozenset({"profile", "fact"}),
            min_cosine=LIGHT_RAG_MIN_COSINE_SIM,
        )
    except Exception as exc:
        trace(f"EXPLAIN_CAUSAL light_rag skip | {exc}")
        return []
    out: list[str] = []
    for _sim, text, _meta in hits:
        t = (text or "").strip()
        if len(t) >= 12 and t not in out:
            out.append(t[:800])
        if len(out) >= limit:
            break
    return out


def _profile_overlap_facts(rag_profile: str, focus: str, limit: int) -> list[str]:
    profile = (rag_profile or "").strip()
    if not profile:
        return []
    focus_tokens = _token_set(focus)
    if not focus_tokens:
        return []
    candidates: list[tuple[int, str]] = []
    for raw in re.split(r"[\n•\-]+", profile):
        line = raw.strip(" .-•\t")
        if len(line) < 20:
            continue
        overlap = len(focus_tokens & _token_set(line))
        if overlap <= 0:
            continue
        candidates.append((overlap, line[:800]))
    candidates.sort(key=lambda t: t[0], reverse=True)
    out: list[str] = []
    for _, line in candidates:
        if line not in out:
            out.append(line)
        if len(out) >= limit:
            break
    return out


def retrieve_causal_facts(
    query: str,
    *,
    rag_profile: str = "",
    top_k: int | None = None,
    use_light_rag: bool = True,
) -> str:
    """Selection-scoped causal / boundary facts (LightRAG, then profile overlap)."""
    cap = top_k if top_k is not None else EXPLAIN_CAUSAL_FACTS_TOP_K
    q = (query or "").strip()
    facts: list[str] = []
    if use_light_rag and q:
        facts = _light_rag_causal_sync(q, cap)
    if len(facts) < cap:
        for line in _profile_overlap_facts(rag_profile, q, cap):
            if line not in facts:
                facts.append(line)
            if len(facts) >= cap:
                break
    block = format_causal_facts_block(facts[: max(1, int(cap))] if facts else [])
    if block:
        trace(f"EXPLAIN_CAUSAL ✓ | n={len(facts[:cap])}")
    return block


def build_explain_context_bundle(
    *,
    selected_text: str,
    user_question: str,
    surrounding_paragraph: str = "",
    resolved_r_chunks: list[dict[str, Any]] | None = None,
    rag_profile: str = "",
    curriculum_id: str = "",
    node: Any = None,
    node_title: str = "",
    store: Any | None = None,
) -> ExplainContextBundle:
    """
    Assemble Target Anchor (+ optional rag_chunks fallback), invariants, causal facts.

    Failures are swallowed so Explainer stays usable without LanceDB/Ollama.
    """
    selected = (selected_text or "").strip()
    question = (user_question or "").strip()
    title = (node_title or str(getattr(node, "title", "") or "")).strip()
    focus_query = " ".join(p for p in [title, selected, question] if p).strip()

    resolved = list(resolved_r_chunks or [])
    anchor_rows = list(resolved)
    if not anchor_rows:
        try:
            anchor_rows = retrieve_anchor_fallback_chunks(
                focus_query or selected,
                curriculum_id=curriculum_id,
                node=node,
                store=store,
            )
        except Exception as exc:
            trace(f"EXPLAIN_BUNDLE anchor skip | {exc}")
            anchor_rows = []

    prefer = _prefer_doc_ids_from_chunks(anchor_rows)
    invariants = ""
    try:
        invariants = retrieve_explain_invariants(
            focus_query or selected,
            curriculum_id=curriculum_id,
            node=node,
            prefer_doc_ids=prefer,
            store=store,
        )
    except Exception as exc:
        trace(f"EXPLAIN_BUNDLE invariants skip | {exc}")

    causal = ""
    try:
        # LightRAG embed only when curriculum-scoped (API path); else profile overlap.
        causal = retrieve_causal_facts(
            focus_query or selected,
            rag_profile=rag_profile,
            use_light_rag=bool((curriculum_id or "").strip()),
        )
    except Exception as exc:
        trace(f"EXPLAIN_BUNDLE causal skip | {exc}")

    return ExplainContextBundle(
        resolved_r_chunks=resolved,
        anchor_block=_format_anchor_block(anchor_rows),
        invariants_block=invariants,
        causal_block=causal,
        prefer_doc_ids=prefer,
    )
