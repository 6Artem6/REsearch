"""Lightweight validation: non-empty queries + verbatim acronym preservation."""

from __future__ import annotations

import re

from knowledge_engine.src.state import ValidatedQuerySpec

_USER_ACRONYM_RE = re.compile(r"\b[A-Z]{2,5}\b")


def extract_user_acronyms(user_query: str) -> list[str]:
    """Uppercase acronyms 2–5 letters from the original prompt."""
    text = user_query or ""
    seen: set[str] = set()
    out: list[str] = []
    for match in _USER_ACRONYM_RE.finditer(text):
        token = match.group(0)
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _acronym_present(acronym: str, text: str) -> bool:
    return (
        re.search(rf"\b{re.escape(acronym)}\b", text or "", re.IGNORECASE) is not None
    )


def validate_search_queries(
    user_query: str,
    spec: ValidatedQuerySpec,
) -> ValidatedQuerySpec:
    """
    Structural checks only:
    - search_queries non-empty;
    - each user acronym (MCP, RAG, GUI, …) appears verbatim in every search query.
    """
    acronyms = extract_user_acronyms(user_query)
    preserved = list(spec.preserved_terms or [])
    for a in acronyms:
        if a not in preserved:
            preserved.append(a)

    queries = [q.strip() for q in spec.search_queries if q and q.strip()]
    formal = (spec.cs_formal_query or "").strip()

    if not queries:
        fallback = formal or user_query.strip()
        queries = [fallback[:240]] if fallback else []

    if not queries:
        queries = [user_query.strip()[:240]]

    fixed: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        q = " ".join(raw.split())
        for acronym in acronyms:
            if not _acronym_present(acronym, q):
                q = f"{q} {acronym}"
        if q and q not in seen:
            seen.add(q)
            fixed.append(q)

    keywords = [k.strip() for k in spec.target_keywords if k and k.strip()]
    for a in acronyms:
        if not any(_acronym_present(a, k) for k in keywords):
            keywords.append(a)
    keywords = keywords[:5]

    return ValidatedQuerySpec(
        cs_formal_query=formal or spec.cs_formal_query,
        target_keywords=keywords,
        search_queries=fixed[:3],
        preserved_terms=preserved[:12],
    )
