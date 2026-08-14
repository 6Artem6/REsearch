"""Unified async arXiv Atom API client with rate limit + backoff."""

from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlencode

import httpx

from knowledge_engine.config import (
    ARXIV_API_URL,
    ARXIV_BACKOFF_BASE_SEC,
    ARXIV_ID_LIST_CHUNK,
    ARXIV_MAX_RETRIES,
    ARXIV_TIMEOUT_SEC,
)
from knowledge_engine.services.search.arxiv_rate_limit import (
    acquire_arxiv_slot_async,
    arxiv_pause_before_retry_async,
)
from knowledge_engine.ui.run_log import trace

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
_RETRYABLE_STATUS = frozenset({403, 429, 503})
_ARXIV_ID_FROM_ENTRY = re.compile(r"arxiv\.org/abs/([^/\s]+)", re.I)
_VERSION_SUFFIX = re.compile(r"v\d+$", re.I)


@dataclass(frozen=True)
class ArxivEntry:
    arxiv_id: str
    title: str
    abstract: str
    entry_id: str
    pdf_url: str
    published: str = ""
    updated: str = ""
    primary_category: str = ""

    @property
    def abs_url(self) -> str:
        if self.entry_id:
            return self.entry_id
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return ""


def normalize_arxiv_id(raw: str) -> str:
    """Strip URL wrappers / .pdf / version suffix → bare arXiv id for id_list."""
    text = (raw or "").strip()
    if not text:
        return ""
    text = text.replace("https://", "").replace("http://", "")
    text = text.replace("arxiv.org/abs/", "").replace("arxiv.org/pdf/", "")
    text = text.replace("export.arxiv.org/abs/", "")
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    text = text.strip().strip("/")
    text = _VERSION_SUFFIX.sub("", text)
    return text.strip()


def _parse_atom_entries(xml_text: str) -> list[ArxivEntry]:
    body = (xml_text or "").strip()
    if not body or body.startswith("<!DOCTYPE") or "<feed" not in body[:800]:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []

    out: list[ArxivEntry] = []
    for entry in root.findall("a:entry", _ATOM_NS):
        title = (
            entry.findtext("a:title", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        title = re.sub(r"\s+", " ", title)
        abstract = (
            entry.findtext("a:summary", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        entry_id = (
            entry.findtext("a:id", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        published = (
            entry.findtext("a:published", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        updated = (
            entry.findtext("a:updated", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        primary = ""
        for cat in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
            primary = (cat.attrib.get("term") or "").strip()
            if primary:
                break
        m = _ARXIV_ID_FROM_ENTRY.search(entry_id)
        arxiv_id = normalize_arxiv_id(m.group(1) if m else "")
        if not arxiv_id and entry_id:
            arxiv_id = normalize_arxiv_id(entry_id.rsplit("/", 1)[-1])
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
        if not title and not arxiv_id:
            continue
        out.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                entry_id=entry_id,
                pdf_url=pdf_url,
                published=published,
                updated=updated,
                primary_category=primary,
            )
        )
    return out


class ArxivClient:
    """Single entry point for arXiv Atom search and id_list hydrate."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        timeout_sec: float | None = None,
        max_retries: int | None = None,
        backoff_base_sec: float | None = None,
        id_list_chunk: int | None = None,
    ) -> None:
        self.api_url = (api_url or ARXIV_API_URL).rstrip("?")
        self.timeout_sec = float(
            ARXIV_TIMEOUT_SEC if timeout_sec is None else timeout_sec
        )
        self.max_retries = max(
            0, int(ARXIV_MAX_RETRIES if max_retries is None else max_retries)
        )
        self.backoff_base_sec = float(
            ARXIV_BACKOFF_BASE_SEC if backoff_base_sec is None else backoff_base_sec
        )
        self.id_list_chunk = max(
            1, int(ARXIV_ID_LIST_CHUNK if id_list_chunk is None else id_list_chunk)
        )

    def _backoff_seconds(self, attempt: int) -> float:
        base = max(0.0, self.backoff_base_sec) * (2**attempt)
        jitter = random.uniform(0.0, 0.5)
        return base + jitter

    async def _get_atom(self, params: dict[str, str | int]) -> list[ArxivEntry]:
        query = urlencode(params)
        url = f"{self.api_url}?{query}"
        last_exc: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self.timeout_sec,
            follow_redirects=True,
        ) as client:
            for attempt in range(self.max_retries + 1):
                await acquire_arxiv_slot_async()
                try:
                    resp = await client.get(url)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= self.max_retries:
                        break
                    wait = self._backoff_seconds(attempt)
                    trace(
                        f"arXiv API ⊘ transport {exc} — backoff {wait:.2f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await arxiv_pause_before_retry_async(wait)
                    continue

                if resp.status_code in _RETRYABLE_STATUS:
                    if attempt >= self.max_retries:
                        trace(
                            f"arXiv API ✗ HTTP {resp.status_code} after retries | "
                            f"{query[:120]}"
                        )
                        return []
                    wait = self._backoff_seconds(attempt)
                    trace(
                        f"arXiv API ⊘ HTTP {resp.status_code} — backoff {wait:.2f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await arxiv_pause_before_retry_async(wait)
                    continue

                try:
                    resp.raise_for_status()
                except Exception as exc:
                    trace(f"arXiv API ✗ HTTP {resp.status_code} | {exc}")
                    return []

                entries = _parse_atom_entries(resp.text)
                if not entries and (
                    resp.text.strip().startswith("<!DOCTYPE")
                    or "<feed" not in resp.text[:500]
                ):
                    trace(f"arXiv API ✗ non-atom body (HTTP {resp.status_code})")
                    return []
                return entries

        if last_exc is not None:
            trace(f"arXiv API ✗ {last_exc}")
        return []

    async def search(
        self,
        *,
        search_query: str,
        start: int = 0,
        max_results: int = 5,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> list[ArxivEntry]:
        q = (search_query or "").strip()
        if not q:
            return []
        params: dict[str, str | int] = {
            "search_query": q,
            "start": max(0, int(start)),
            "max_results": max(1, int(max_results)),
        }
        if sort_by:
            params["sortBy"] = sort_by
        if sort_order:
            params["sortOrder"] = sort_order
        trace(f"arXiv API ▶ search | {q[:100]}")
        entries = await self._get_atom(params)
        trace(f"arXiv API ✓ search papers={len(entries)}")
        return entries

    async def search_with_params(
        self,
        params: Any,
        *,
        free_text_fallback: str = "",
        start: int | None = None,
        max_results: int = 5,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> list[ArxivEntry]:
        """Precision search via ArxivQueryBuilder (falls back to all: free text)."""
        from knowledge_engine.services.search.arxiv_query_builder import (
            ArxivQueryBuilder,
        )

        built = ArxivQueryBuilder(params).build(
            free_text_fallback=free_text_fallback,
            start=start,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        if not built.search_query:
            return []
        return await self.search(
            search_query=built.search_query,
            start=built.start,
            max_results=max_results,
            sort_by=built.sort_by,
            sort_order=built.sort_order,
        )

    async def fetch_by_ids(
        self,
        ids: Sequence[str],
        *,
        chunk_size: int | None = None,
    ) -> list[ArxivEntry]:
        """Batch metadata via id_list=ID1,ID2,... (chunks of ≤50 by default)."""
        norm: list[str] = []
        seen: set[str] = set()
        for raw in ids:
            aid = normalize_arxiv_id(raw)
            if not aid or aid.lower() in seen:
                continue
            seen.add(aid.lower())
            norm.append(aid)
        if not norm:
            return []

        size = max(1, int(chunk_size or self.id_list_chunk))
        out: list[ArxivEntry] = []
        for i in range(0, len(norm), size):
            chunk = norm[i : i + size]
            params: dict[str, str | int] = {
                "id_list": ",".join(chunk),
                "start": 0,
                "max_results": len(chunk),
            }
            trace(f"arXiv API ▶ id_list | n={len(chunk)}")
            entries = await self._get_atom(params)
            out.extend(entries)
        trace(f"arXiv API ✓ id_list papers={len(out)}/{len(norm)}")
        return out


_default_client: ArxivClient | None = None


def get_arxiv_client() -> ArxivClient:
    global _default_client
    if _default_client is None:
        _default_client = ArxivClient()
    return _default_client
