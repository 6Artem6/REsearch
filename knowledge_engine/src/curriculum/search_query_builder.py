"""Трансформация expansion_vector → запросы для академического и практического поиска."""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge_engine.src.curriculum.curriculum_search_sites import (
    CURRICULUM_PRIORITY_ENGINEERING_SITES,
)

# Доп. домены из ТЗ (site: whitelist для CSE / DDGS)
_CURRICULUM_PRACTICAL_SITE_WHITELIST: tuple[str, ...] = (
    *CURRICULUM_PRIORITY_ENGINEERING_SITES,
    "highload.ru",
    "cloudflare.com/blog",
)

_EN_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "by",
        "from",
        "as",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "study",
        "studying",
        "approach",
        "approaches",
        "using",
        "use",
        "based",
        "how",
        "what",
        "when",
        "where",
        "why",
        "изучение",
        "подход",
        "подходы",
        "снижение",
        "при",
        "для",
        "как",
        "что",
        "это",
        "или",
        "и",
        "в",
        "на",
        "по",
        "с",
        "из",
        "от",
        "до",
        "при",
        "об",
        "о",
    }
)

_TERM_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}(?:\s+[a-zA-Z][a-zA-Z0-9\-]{2,}){0,3}")
_QUOTED_RE = re.compile(r'"([^"]{4,80})"')


@dataclass(frozen=True)
class BuiltSearchQueries:
    academic_query: str
    practical_query: str
    keywords: tuple[str, ...]


def _extract_keywords(expansion_vector: str, max_terms: int = 6) -> list[str]:
    text = (expansion_vector or "").strip()
    if not text:
        return []

    seen: set[str] = set()
    terms: list[str] = []

    for m in _QUOTED_RE.finditer(text):
        phrase = m.group(1).strip().lower()
        if phrase and phrase not in seen:
            seen.add(phrase)
            terms.append(phrase)

    for m in _TERM_RE.finditer(text):
        phrase = m.group(0).strip().lower()
        words = phrase.split()
        if all(w in _EN_STOP for w in words):
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        terms.append(phrase)

    if not terms:
        rough = re.sub(r"[^\w\s\-]", " ", text.lower())
        for w in rough.split():
            if len(w) >= 4 and w not in _EN_STOP and w not in seen:
                seen.add(w)
                terms.append(w)

    return terms[:max_terms]


def _format_academic_query(keywords: list[str]) -> str:
    picked = keywords[:4]
    if not picked:
        return ""
    parts: list[str] = []
    for t in picked:
        if " " in t:
            parts.append(f'"{t}"')
        else:
            parts.append(t)
    return " ".join(parts)


def _format_practical_query(keywords: list[str]) -> str:
    sites = _CURRICULUM_PRACTICAL_SITE_WHITELIST
    site_clause = " OR ".join(f"site:{s}" for s in sites)
    kw = " ".join(keywords[:6])
    if not kw:
        kw = "engineering architecture"
    return f"({site_clause}) {kw}"


def build_search_queries(expansion_vector: str) -> BuiltSearchQueries:
    """
    academic_query — 2–4 английских термина для Semantic Scholar / arXiv.
    practical_query — site: whitelist + keywords для Google CSE / DDGS.
    """
    keywords = _extract_keywords(expansion_vector)
    academic = _format_academic_query(keywords)
    practical = _format_practical_query(keywords)
    if not academic:
        academic = (expansion_vector or "").strip()[:120]
    return BuiltSearchQueries(
        academic_query=academic,
        practical_query=practical,
        keywords=tuple(keywords),
    )


def build_fallback_quote_queries(learning_goal: str, *, max_queries: int = 6) -> list[str]:
    """Fallback если Lite не вернула JSON: кавычечные термы + site: по приоритетным доменам."""
    keywords = _extract_keywords(learning_goal)
    kw = " ".join(keywords[:4])
    if not kw:
        kw = (learning_goal or "").strip()[:100]
    seen: set[str] = set()
    out: list[str] = []

    def add(q: str) -> None:
        q = (q or "").strip()
        if len(q) < 6 or q in seen:
            return
        seen.add(q)
        out.append(q)

    for t in keywords[:3]:
        if " " in t:
            add(f'"{t}" {kw}')
        else:
            add(f'"{t}"')
    for site in _CURRICULUM_PRACTICAL_SITE_WHITELIST[:4]:
        add(f"site:{site} {kw}")
    if not out:
        add(kw or "distributed systems engineering")
    return out[:max(1, max_queries)]


def build_practical_searxng_queries(
    expansion_vector: str,
    *,
    max_queries: int = 8,
) -> list[str]:
    """
    Набор запросов для SearXNG: один combined site:OR + site: по приоритетным доменам.
    Короткие запросы надёжнее для bing/google engines в SearXNG, чем один гигантский OR.
    """
    built = build_search_queries(expansion_vector)
    kw = " ".join(built.keywords[:4])
    if not kw:
        kw = (expansion_vector or "").strip()[:100]
    seen: set[str] = set()
    out: list[str] = []

    def add(q: str) -> None:
        q = (q or "").strip()
        if len(q) < 8 or q in seen:
            return
        seen.add(q)
        out.append(q)

    add(built.practical_query)
    for site in _CURRICULUM_PRACTICAL_SITE_WHITELIST[:6]:
        add(f"site:{site} {kw}")
    add(f"{kw} distributed systems engineering blog")
    return out[:max(1, max_queries)]

