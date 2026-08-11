"""OpenAlex-based source trust scores for academic works (DOI / arXiv)."""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import quote, unquote

import httpx

from knowledge_engine.config import (
    OPENALEX_CONCURRENCY,
    OPENALEX_DAILY_LIMIT,
    OPENALEX_MAILTO,
    OPENALEX_TIMEOUT_SEC,
    OPENALEX_TRUST_ENABLED,
    PACKAGE_ROOT,
    RAG_TRUST_HARD_CUTOFF,
    RAG_TRUST_HARD_MIN_SIM,
    RAG_TRUST_HARD_MIN_TRUST,
)
from knowledge_engine.ui.run_log import trace

_ARXIV_ID_RE = re.compile(
    r"(?:arxiv(?:\.org)?[\/:](?:abs|pdf|html)\/)?(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)
_ARXIV_ID_LOOSE_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_DOI_RE = re.compile(
    r"\b(10\.\d{4,9}/[^\s\"'<>\[\]{}]+)",
    re.I,
)
_DOI_TRAIL_PUNCT = re.compile(r"[.,;:)\]}>]+$")

_DOC_SOURCE_TYPES = frozenset(
    {
        "doc",
        "docs",
        "documentation",
        "vendor_doc",
        "official_docs",
        "api_reference",
        "reference",
    }
)
_DOC_URL_HINTS = (
    "docs.",
    "/docs/",
    "documentation",
    "readthedocs",
    "developer.mozilla",
    "developers.google",
)

_QUOTA_PATH: Path = (PACKAGE_ROOT / ".runs" / "openalex_quota_state.json").resolve()
_QUOTA_LOCK = threading.Lock()
_SOFT_FALLBACK = 0.3
_ARXIV_QUOTA_FALLBACK = _SOFT_FALLBACK  # backward-compatible alias


def final_retrieval_score(
    vector_similarity: float,
    trust_score: float | None = None,
) -> float:
    """Rank key for RAG context: sim × trust (default trust=1)."""
    trust = 1.0 if trust_score is None else float(trust_score)
    trust = max(0.0, min(1.0, trust))
    return float(vector_similarity) * trust


def coerce_trust_score(raw: Any, *, default: float = 1.0) -> float:
    if raw is None:
        return float(default)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return float(default)


def passes_trust_hard_cutoff(
    vector_similarity: float,
    trust_score: float | None,
    *,
    min_trust: float | None = None,
    min_sim_if_low_trust: float | None = None,
) -> bool:
    """
    Hard reject: trust < min_trust AND vector_sim < min_sim → drop.

    Keep when trust is adequate OR similarity is exceptionally high.
    Missing trust defaults to 1.0 (legacy rows).
    """
    if not RAG_TRUST_HARD_CUTOFF:
        return True
    trust = coerce_trust_score(trust_score, default=1.0)
    thr_trust = RAG_TRUST_HARD_MIN_TRUST if min_trust is None else float(min_trust)
    thr_sim = (
        RAG_TRUST_HARD_MIN_SIM
        if min_sim_if_low_trust is None
        else float(min_sim_if_low_trust)
    )
    if trust >= thr_trust:
        return True
    return float(vector_similarity) >= thr_sim


def filter_rows_trust_hard_cutoff(
    rows: Sequence[dict[str, Any]],
    *,
    sim_key: str = "_cosine_raw",
    trust_key: str = "_trust_score",
) -> tuple[list[dict[str, Any]], int]:
    """
    Early-exit filter right after raw vector hits are scored.

    Call BEFORE CE/MMR, Lite Map, or heavy context assembly.
    Rows must already expose raw similarity + trust (or COL_TRUST_SCORE).
    """
    from knowledge_engine.db.rag_chunks_schema import COL_TRUST_SCORE

    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        sim = float(row.get(sim_key) if row.get(sim_key) is not None else 0.0)
        trust_raw = row.get(trust_key)
        if trust_raw is None:
            trust_raw = row.get(COL_TRUST_SCORE)
        if not passes_trust_hard_cutoff(sim, trust_raw):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def filter_candidates_trust_hard_cutoff(
    candidates: Sequence[Any],
) -> tuple[list[Any], int]:
    """
    Belt-and-suspenders early exit on LectureContextCandidate-like objects
    before CE/MMR or prompt assembly. Candidates without vector_similarity
    set (non-rag sources) pass through unchanged.
    """
    kept: list[Any] = []
    dropped = 0
    for c in candidates:
        trust = getattr(c, "trust_score", None)
        sim = getattr(c, "vector_similarity", None)
        # Only enforce when we have a real retrieval similarity signal
        if sim is None or float(sim) <= 0.0:
            kept.append(c)
            continue
        if not passes_trust_hard_cutoff(float(sim), trust):
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped


def looks_like_vendor_doc(url: str, source_type: str | None = None) -> bool:
    st = (source_type or "").strip().lower()
    if st in _DOC_SOURCE_TYPES or "documentation" in st:
        return True
    u = (url or "").lower()
    return any(h in u for h in _DOC_URL_HINTS)


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _empty_quota_state() -> dict[str, Any]:
    return {
        "day_utc": _utc_day(),
        "requests_today": 0,
        "daily_limit": int(OPENALEX_DAILY_LIMIT),
    }


def _quota_try_consume(*, n: int = 1) -> bool:
    """Cross-process daily counter. False → over limit (use arXiv fallback)."""
    limit = max(0, int(OPENALEX_DAILY_LIMIT))
    if limit <= 0:
        return True
    _QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _QUOTA_LOCK:
        with open(_QUOTA_PATH, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                try:
                    state = json.loads(raw) if raw else _empty_quota_state()
                except Exception:
                    state = _empty_quota_state()
                if not isinstance(state, dict):
                    state = _empty_quota_state()
                if state.get("day_utc") != _utc_day():
                    state = _empty_quota_state()
                used = int(state.get("requests_today") or 0)
                if used + n > limit:
                    return False
                state["requests_today"] = used + n
                state["daily_limit"] = limit
                state["day_utc"] = _utc_day()
                f.seek(0)
                f.truncate()
                f.write(json.dumps(state, ensure_ascii=False, indent=2))
                f.flush()
                return True
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def openalex_quota_snapshot() -> dict[str, Any]:
    """Read-only view for tests/diagnostics."""
    if not _QUOTA_PATH.is_file():
        return _empty_quota_state()
    try:
        raw = json.loads(_QUOTA_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return _empty_quota_state()


class OpenAlexEvaluator:
    """Enrich academic sources via OpenAlex (DOI or arXiv) → trust_score."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        email: str | None = None,
        *,
        timeout_sec: float | None = None,
        enabled: bool | None = None,
        concurrency: int | None = None,
    ):
        mail = (email if email is not None else OPENALEX_MAILTO).strip()
        self.email = mail or "dev@knowledge-engine.local"
        self.timeout_sec = float(
            OPENALEX_TIMEOUT_SEC if timeout_sec is None else timeout_sec
        )
        self.enabled = OPENALEX_TRUST_ENABLED if enabled is None else bool(enabled)
        self.concurrency = max(
            1, int(OPENALEX_CONCURRENCY if concurrency is None else concurrency)
        )
        self.headers = {
            "User-Agent": f"KnowledgeEngine/1.0 (mailto:{self.email})",
            "Accept": "application/json",
        }
        self._cache: dict[str, float] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def extract_arxiv_id(url_or_id: str) -> Optional[str]:
        text = (url_or_id or "").strip()
        if not text:
            return None
        m = _ARXIV_ID_RE.search(text) or _ARXIV_ID_LOOSE_RE.search(text)
        return m.group(1) if m else None

    @staticmethod
    def extract_doi(url_or_text: str) -> Optional[str]:
        """Return bare DOI (10.xxxx/...) from URL / metadata blob, if any."""
        text = unquote((url_or_text or "").strip())
        if not text:
            return None
        lower = text.lower()
        if "doi.org/" in lower:
            tail = text.split("doi.org/", 1)[-1]
            tail = tail.split("?", 1)[0].split("#", 1)[0]
            if tail.lower().startswith("doi/"):
                tail = tail[4:]
            doi = _DOI_TRAIL_PUNCT.sub("", tail.strip())
            if doi.lower().startswith("10."):
                return doi
        m = _DOI_RE.search(text)
        if not m:
            return None
        doi = _DOI_TRAIL_PUNCT.sub("", m.group(1).strip())
        # Drop URL path junk after DOI
        doi = doi.split("?", 1)[0].split("#", 1)[0]
        doi = _DOI_TRAIL_PUNCT.sub("", doi)
        return doi if doi.lower().startswith("10.") else None

    @staticmethod
    def is_published_venue(data: dict[str, Any]) -> bool:
        locations = list(data.get("locations") or [])
        primary = data.get("primary_location")
        if isinstance(primary, dict):
            locations = [primary, *locations]
        for loc in locations:
            if not isinstance(loc, dict):
                continue
            if (loc.get("version") or "").strip() == "publishedVersion":
                return True
            src = loc.get("source") or {}
            if not isinstance(src, dict):
                continue
            src_type = (src.get("type") or "").strip().lower()
            if src_type in {"journal", "conference"}:
                return True
        return False

    @staticmethod
    def is_open_access(data: dict[str, Any]) -> bool:
        oa = data.get("open_access")
        if isinstance(oa, dict) and oa.get("is_oa") is True:
            return True
        primary = data.get("primary_location")
        if isinstance(primary, dict) and primary.get("is_oa") is True:
            return True
        return bool(data.get("is_oa"))

    @classmethod
    def score_from_openalex_payload(cls, data: dict[str, Any]) -> float:
        citations = int(data.get("cited_by_count") or 0)
        citations = max(0, citations)
        published = cls.is_published_venue(data)
        is_oa = cls.is_open_access(data)
        if published:
            w_source = 0.85
        elif is_oa:
            w_source = 0.6
        else:
            w_source = 0.5
        c_sat = 40.0
        alpha = 0.3
        log_citations = math.log(1.0 + citations)
        log_sat = math.log(1.0 + c_sat)
        citation_factor = min(1.0, log_citations / log_sat) if log_sat > 0 else 0.0
        score = w_source * (alpha + (1.0 - alpha) * citation_factor)
        return round(max(0.0, min(1.0, score)), 3)

    def _cache_get(self, key: str) -> float | None:
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_set(self, key: str, value: float) -> None:
        with self._cache_lock:
            self._cache[key] = value

    def warm_cache(self, mapping: dict[str, float]) -> None:
        with self._cache_lock:
            self._cache.update(mapping)

    def _short_circuit_trust(
        self,
        source_url_or_id: str,
        *,
        is_doc: bool,
    ) -> float | None:
        if not self.enabled:
            return 1.0
        raw = source_url_or_id or ""
        if is_doc or looks_like_vendor_doc(raw):
            return 1.0
        return None

    def _work_path_for_doi(self, doi: str) -> str:
        doi_url = f"https://doi.org/{doi}"
        return f"{self.BASE_URL}/{quote(doi_url, safe='')}"

    def _resolve_lookup(
        self,
        source_url_or_id: str,
        *,
        doi: str | None = None,
    ) -> tuple[str, str] | None:
        """
        Returns (cache_key, doi_for_openalex) or None when soft-fallback.
        Prefer explicit/URL DOI; else arXiv → 10.48550/arXiv.{id}.
        """
        resolved_doi = (doi or "").strip() or self.extract_doi(source_url_or_id)
        if resolved_doi:
            m = re.match(r"10\.48550/arxiv\.(.+)$", resolved_doi, re.I)
            if m:
                resolved_doi = f"10.48550/arXiv.{m.group(1)}"
            return f"doi:{resolved_doi.lower()}", resolved_doi

        arxiv_id = self.extract_arxiv_id(source_url_or_id)
        if arxiv_id:
            arxiv_doi = f"10.48550/arXiv.{arxiv_id}"
            return f"arxiv:{arxiv_id}", arxiv_doi
        return None

    def _http_score_for_doi(self, doi: str) -> float:
        if not _quota_try_consume(n=1):
            trace(
                f"OPENALEX quota ⊘ | daily limit — fallback trust="
                f"{_SOFT_FALLBACK:.2f} | doi:{doi[:48]}"
            )
            return _SOFT_FALLBACK

        url = self._work_path_for_doi(doi)
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.get(url, headers=self.headers)
                if res.status_code != 200:
                    return 0.25
                return self.score_from_openalex_payload(res.json())
        except Exception:
            return 0.3

    async def _http_score_for_doi_async(
        self,
        doi: str,
        *,
        client: httpx.AsyncClient,
    ) -> float:
        if not _quota_try_consume(n=1):
            trace(
                f"OPENALEX quota ⊘ | daily limit — fallback trust="
                f"{_SOFT_FALLBACK:.2f} | doi:{doi[:48]}"
            )
            return _SOFT_FALLBACK

        url = self._work_path_for_doi(doi)
        try:
            res = await client.get(url, headers=self.headers)
            if res.status_code != 200:
                return 0.25
            return self.score_from_openalex_payload(res.json())
        except Exception:
            return 0.3

    def _http_score_for_arxiv(self, arxiv_id: str) -> float:
        return self._http_score_for_doi(f"10.48550/arXiv.{arxiv_id}")

    async def _http_score_for_arxiv_async(
        self,
        arxiv_id: str,
        *,
        client: httpx.AsyncClient,
    ) -> float:
        return await self._http_score_for_doi_async(
            f"10.48550/arXiv.{arxiv_id}",
            client=client,
        )

    def fetch_trust_score_sync(
        self,
        source_url_or_id: str,
        is_doc: bool = False,
        *,
        doi: str | None = None,
    ) -> float:
        early = self._short_circuit_trust(source_url_or_id, is_doc=is_doc)
        if early is not None:
            return early

        lookup = self._resolve_lookup(source_url_or_id, doi=doi)
        if lookup is None:
            return _SOFT_FALLBACK

        cache_key, work_doi = lookup
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        score = self._http_score_for_doi(work_doi)
        self._cache_set(cache_key, score)
        return score

    async def fetch_trust_score(
        self,
        source_url_or_id: str,
        is_doc: bool = False,
        *,
        client: httpx.AsyncClient | None = None,
        doi: str | None = None,
    ) -> float:
        early = self._short_circuit_trust(source_url_or_id, is_doc=is_doc)
        if early is not None:
            return early

        lookup = self._resolve_lookup(source_url_or_id, doi=doi)
        if lookup is None:
            return _SOFT_FALLBACK

        cache_key, work_doi = lookup
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if client is None:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as owned:
                score = await self._http_score_for_doi_async(work_doi, client=owned)
        else:
            score = await self._http_score_for_doi_async(work_doi, client=client)
        self._cache_set(cache_key, score)
        return score

    async def fetch_trust_scores_batch(
        self,
        items: Sequence[tuple[str, str | None] | str],
        *,
        concurrency: int | None = None,
    ) -> list[float]:
        """Parallel OpenAlex lookups with a concurrency semaphore."""
        norm: list[tuple[str, str | None]] = []
        for item in items:
            if isinstance(item, str):
                norm.append((item, None))
            else:
                url, st = item
                norm.append((url or "", st))

        if not norm:
            return []

        sem = asyncio.Semaphore(max(1, int(concurrency or self.concurrency)))

        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:

            async def _one(url: str, source_type: str | None) -> float:
                async with sem:
                    return await self.fetch_trust_score(
                        url,
                        is_doc=looks_like_vendor_doc(url, source_type),
                        client=client,
                    )

            return list(
                await asyncio.gather(
                    *[_one(u, st) for u, st in norm],
                    return_exceptions=False,
                )
            )


_default_evaluator: OpenAlexEvaluator | None = None
_default_lock = threading.Lock()


def get_openalex_evaluator() -> OpenAlexEvaluator:
    global _default_evaluator
    with _default_lock:
        if _default_evaluator is None:
            _default_evaluator = OpenAlexEvaluator()
        return _default_evaluator


def resolve_source_trust_score(
    url: str,
    *,
    source_type: str | None = None,
    evaluator: OpenAlexEvaluator | None = None,
) -> float:
    """One-shot trust for document ingest (cached per arXiv id)."""
    ev = evaluator or get_openalex_evaluator()
    return ev.fetch_trust_score_sync(
        url or "",
        is_doc=looks_like_vendor_doc(url or "", source_type),
    )


async def prefetch_trust_scores_async(
    urls: Sequence[str],
    *,
    source_types: Sequence[str | None] | None = None,
    evaluator: OpenAlexEvaluator | None = None,
    concurrency: int | None = None,
) -> dict[str, float]:
    """Warm OpenAlex cache for a batch of ingest URLs (parallel)."""
    ev = evaluator or get_openalex_evaluator()
    items: list[tuple[str, str | None]] = []
    for i, url in enumerate(urls):
        u = (url or "").strip()
        if not u:
            continue
        st = None
        if source_types is not None and i < len(source_types):
            st = source_types[i]
        items.append((u, st))
    if not items:
        return {}
    scores = await ev.fetch_trust_scores_batch(items, concurrency=concurrency)
    out: dict[str, float] = {}
    for (url, _), score in zip(items, scores):
        out[url] = float(score)
    trace(f"OPENALEX prefetch ✓ | urls={len(out)}")
    return out
