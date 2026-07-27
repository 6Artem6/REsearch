"""v0.6: Smart Targeted Search — операторы запросов для SearXNG / Bing / Google."""

from __future__ import annotations

import re

from knowledge_engine.config import SMART_QUERY_SYNTAX_ENABLED

# Платформы инженерных блогов и репозиториев (не дублировать в каждом векторе LLM)
_ENGINEERING_SITE_FILTER = (
    "(site:github.io OR site:substack.com OR site:dev.to OR site:arxiv.org)"
)

_EXCLUSION_GUARD = (
    "-inurl:cart -inurl:shop -site:ikea.com -site:amazon.com "
    "-site:ikea.ru -site:avito.ru"
)

_SITE_FILTER_RE = re.compile(r"\bsite:", re.I)
_EXCLUSION_MARKER = "-inurl:cart"


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def apply_exclusion_guard(query: str) -> str:
    q = query.strip()
    if not q or _EXCLUSION_MARKER in q:
        return q
    return _normalize_spaces(f"{q} {_EXCLUSION_GUARD}")


def apply_engineering_site_filter(query: str) -> str:
    """Deep Engineering / Architecture: блоги и репозитории."""
    q = query.strip()
    if not q or _SITE_FILTER_RE.search(q):
        return apply_exclusion_guard(q)
    return apply_exclusion_guard(f"{q} {_ENGINEERING_SITE_FILTER}")


def apply_smart_query_syntax(query: str, engineering: bool = True) -> str:
    """Полная пост-обработка одного поискового вектора."""
    if not SMART_QUERY_SYNTAX_ENABLED:
        return query.strip()
    q = query.strip()
    if not q:
        return q
    if engineering:
        return apply_engineering_site_filter(q)
    return apply_exclusion_guard(q)


def apply_smart_query_syntax_batch(
    queries: list[str],
    engineering: bool = True,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        q = apply_smart_query_syntax(raw, engineering=engineering)
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out
