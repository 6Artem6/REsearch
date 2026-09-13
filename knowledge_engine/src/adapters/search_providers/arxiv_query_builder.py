"""Build arXiv Atom `search_query` + sort/paging params from structured fields.

Spec: https://info.arxiv.org/help/api/user-manual.html (§3.1.1, §5.1)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlencode

_CAT_RE = re.compile(r"^[a-z]+(\.[A-Za-z0-9\-]+)?$", re.I)
_SAFE_TERM_RE = re.compile(r'^[^:"()]+$')


@dataclass
class ArxivQueryParams:
    title_keywords: list[str] = field(default_factory=list)
    abstract_keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    start_year: int | None = None
    end_year: int | None = None
    sort_by: str = "relevance"  # relevance | submittedDate | lastUpdatedDate
    sort_order: str = "descending"  # ascending | descending
    start: int = 0

    def has_precision(self) -> bool:
        return bool(
            self.title_keywords
            or self.abstract_keywords
            or self.categories
            or self.exclude_terms
            or self.start_year
            or self.end_year
        )

    @classmethod
    def from_mapping(cls, raw: Any) -> ArxivQueryParams:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if hasattr(raw, "model_dump"):
            data = raw.model_dump()
        elif isinstance(raw, dict):
            data = raw
        else:
            return cls()
        return cls(
            title_keywords=_as_str_list(data.get("title_keywords")),
            abstract_keywords=_as_str_list(data.get("abstract_keywords")),
            categories=_as_str_list(data.get("categories")),
            exclude_terms=_as_str_list(data.get("exclude_terms")),
            start_year=_as_optional_int(data.get("start_year")),
            end_year=_as_optional_int(data.get("end_year")),
            sort_by=str(data.get("sort_by") or "relevance").strip() or "relevance",
            sort_order=str(data.get("sort_order") or "descending").strip()
            or "descending",
            start=max(0, int(data.get("start") or 0)),
        )


@dataclass(frozen=True)
class BuiltArxivQuery:
    search_query: str
    sort_by: str | None
    sort_order: str | None
    start: int
    max_results: int | None = None

    def as_query_params(self, *, max_results: int = 5) -> dict[str, str | int]:
        params: dict[str, str | int] = {
            "search_query": self.search_query,
            "start": self.start,
            "max_results": int(self.max_results or max_results),
        }
        if self.sort_by:
            params["sortBy"] = self.sort_by
        if self.sort_order:
            params["sortOrder"] = self.sort_order
        return params

    def encode(self, *, max_results: int = 5) -> str:
        return urlencode(self.as_query_params(max_results=max_results))


def _as_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        t = re.sub(r"\s+", " ", str(item or "").strip())
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _as_optional_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _quote_term(term: str) -> str:
    t = re.sub(r"\s+", " ", (term or "").strip())
    t = t.replace('"', "")
    if not t:
        return ""
    if " " in t or not _SAFE_TERM_RE.match(t):
        return f'"{t}"'
    return t


def _field_or_clause(prefix: str, terms: Sequence[str]) -> str:
    parts = [_quote_term(t) for t in terms]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return f"{prefix}:{parts[0]}"
    inner = " OR ".join(f"{prefix}:{p}" for p in parts)
    return f"({inner})"


def _normalize_category(cat: str) -> str:
    c = (cat or "").strip()
    if c.lower().startswith("cat:"):
        c = c[4:].strip()
    return c


def _category_clause(categories: Sequence[str]) -> str:
    cats: list[str] = []
    seen: set[str] = set()
    for raw in categories:
        c = _normalize_category(raw)
        if not c or not _CAT_RE.match(c):
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        cats.append(c)
    if not cats:
        return ""
    if len(cats) == 1:
        return f"cat:{cats[0]}"
    return "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"


def _submitted_date_clause(start_year: int | None, end_year: int | None) -> str:
    if start_year is None and end_year is None:
        return ""
    y0 = int(start_year) if start_year is not None else 1991
    y1 = int(end_year) if end_year is not None else 2100
    if y0 > y1:
        y0, y1 = y1, y0
    y0 = max(1991, min(2100, y0))
    y1 = max(1991, min(2100, y1))
    return f"submittedDate:[{y0:04d}01010000 TO {y1:04d}12312359]"


def _exclude_clause(terms: Sequence[str]) -> str:
    parts: list[str] = []
    for t in terms:
        q = _quote_term(t)
        if q:
            parts.append(f"ANDNOT all:{q}")
    return " ".join(parts)


class ArxivQueryBuilder:
    """Compose precision `search_query` strings for export.arxiv.org."""

    VALID_SORT_BY = frozenset({"relevance", "submittedDate", "lastUpdatedDate"})
    VALID_SORT_ORDER = frozenset({"ascending", "descending"})

    def __init__(self, params: ArxivQueryParams | Any | None = None) -> None:
        self.params = ArxivQueryParams.from_mapping(params)

    def build_search_query(self, *, free_text_fallback: str = "") -> str:
        p = self.params
        clauses: list[str] = []
        ti = _field_or_clause("ti", p.title_keywords)
        abs_ = _field_or_clause("abs", p.abstract_keywords)
        cat = _category_clause(p.categories)
        if ti:
            clauses.append(ti)
        if abs_:
            clauses.append(abs_)
        if cat:
            clauses.append(cat)
        date_clause = _submitted_date_clause(p.start_year, p.end_year)
        if date_clause:
            clauses.append(date_clause)

        if not clauses:
            fb = (free_text_fallback or "").strip()
            if fb.lower().startswith(("all:", "ti:", "abs:", "cat:", "au:")):
                core = fb
            elif fb:
                core = f"all:{_quote_term(fb) or fb}"
            else:
                return ""
        else:
            core = " AND ".join(clauses)

        excl = _exclude_clause(p.exclude_terms)
        if excl:
            return f"{core} {excl}".strip()
        return core

    def build(
        self,
        *,
        free_text_fallback: str = "",
        start: int | None = None,
        max_results: int | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> BuiltArxivQuery:
        p = self.params
        sb = (sort_by or p.sort_by or "relevance").strip()
        so = (sort_order or p.sort_order or "descending").strip()
        if sb not in self.VALID_SORT_BY:
            sb = "relevance"
        if so not in self.VALID_SORT_ORDER:
            so = "descending"
        # relevance ordering ignores sortOrder in practice; still pass for API completeness
        start_i = p.start if start is None else max(0, int(start))
        return BuiltArxivQuery(
            search_query=self.build_search_query(free_text_fallback=free_text_fallback),
            sort_by=sb,
            sort_order=so,
            start=start_i,
            max_results=max_results,
        )


def heuristic_arxiv_params_from_keywords(
    keywords: Sequence[str],
    *,
    free_text: str = "",
) -> ArxivQueryParams:
    """Deterministic fallback when the LLM pass did not return arxiv_params."""
    kws = _as_str_list(list(keywords)[:6])
    if not kws and free_text:
        rough = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", free_text)
        kws = _as_str_list(rough[:6])
    return ArxivQueryParams(
        title_keywords=kws[:2],
        abstract_keywords=kws[:4],
        categories=[],
        exclude_terms=["survey", "homework"],
        sort_by="relevance",
        sort_order="descending",
    )
