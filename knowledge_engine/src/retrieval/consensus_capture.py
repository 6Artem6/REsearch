"""Перехват JSON от Consensus + парсинг публикаций из ответов API."""

from __future__ import annotations

import json
import re
from typing import Any, List
from urllib.parse import urlparse

from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper
from knowledge_engine.ui.run_log import trace

_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.I)


def _pick_str(obj: dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _pick_doi(obj: dict[str, Any]) -> str:
    doi = _pick_str(obj, "doi", "DOI")
    if doi:
        return doi
    ext = obj.get("externalIds")
    if isinstance(ext, dict):
        doi = str(ext.get("DOI") or ext.get("doi") or "").strip()
        if doi:
            return doi
    return ""


def _paper_from_dict(obj: dict[str, Any]) -> ScholarPaper | None:
    title = _pick_str(obj, "title", "paperTitle", "name", "displayName")
    abstract = _pick_str(obj, "abstract", "snippet", "summary", "tldr")
    tldr_raw = obj.get("tldr")
    tldr = ""
    if isinstance(tldr_raw, dict):
        tldr = str(tldr_raw.get("text") or "").strip()
    elif isinstance(tldr_raw, str):
        tldr = tldr_raw.strip()

    doi = _pick_doi(obj)
    url = _pick_str(obj, "url", "sourceUrl", "link", "paperUrl", "source_url")
    pdf = ""
    oa = obj.get("openAccessPdf")
    if isinstance(oa, dict):
        pdf = str(oa.get("url") or "").strip()
    pdf = pdf or _pick_str(obj, "pdfUrl", "pdf_url")

    if doi and not url:
        url = f"https://doi.org/{doi}"
    if not title and not url and not abstract:
        return None
    if not title:
        title = url or "publication"
    # Навигация / UI labels из JSON API (не публикации)
    if not url and not doi and len(abstract) < 40 and len(title.split()) <= 3:
        return None

    year_raw = obj.get("year")
    year = int(year_raw) if isinstance(year_raw, int) else None
    venue = _pick_str(obj, "venue", "journal")

    return ScholarPaper(
        paper_id=_pick_str(obj, "paperId", "id", "paper_id") or doi or url,
        title=title[:500],
        year=year,
        tldr=tldr,
        abstract=abstract[:8000],
        venue=venue,
        pdf_url=pdf,
        source_url=url,
        source="consensus_api",
    )


def _walk_json(node: Any, out: List[ScholarPaper], seen: set[str]) -> None:
    if isinstance(node, dict):
        p = _paper_from_dict(node)
        if p:
            key = (p.source_url or p.title).lower()
            if key and key not in seen:
                seen.add(key)
                out.append(p)
        for v in node.values():
            _walk_json(v, out, seen)
    elif isinstance(node, list):
        for item in node:
            _walk_json(item, out, seen)


def papers_from_json_payload(data: Any) -> List[ScholarPaper]:
    out: List[ScholarPaper] = []
    seen: set[str] = set()
    _walk_json(data, out, seen)
    return out


def papers_from_json_text(text: str) -> List[ScholarPaper]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    papers = papers_from_json_payload(data)
    if papers:
        trace(f"Consensus ✓ JSON papers={len(papers)}")
    return papers


def papers_from_json_text_relaxed(text: str) -> List[ScholarPaper]:
    papers = papers_from_json_text(text)
    if papers:
        return papers
    # эвристика: вытащить DOI/arXiv из сырого JSON-текста
    urls = _URL_RE.findall(text)
    dois = _DOI_RE.findall(text)
    out: List[ScholarPaper] = []
    for doi in dois:
        out.append(
            ScholarPaper(
                title=f"DOI {doi}",
                source_url=f"https://doi.org/{doi}",
                source="consensus_api_relaxed",
            )
        )
    for u in urls:
        if "arxiv.org" in u or "doi.org" in u or "semanticscholar.org" in u:
            out.append(
                ScholarPaper(
                    title=u[:120],
                    source_url=u.rstrip('",'),
                    source="consensus_api_relaxed",
                )
            )
    return out


def is_generic_consensus_url(url: str) -> bool:
    u = (url or "").strip()
    if not u or "consensus.app" not in u:
        return False
    path = urlparse(u).path.strip("/")
    if not path or path == "home":
        return True
    if path.startswith("search") and "/papers/" not in path and "/paper/" not in path:
        return True
    return False


def normalize_paper_urls(papers: List[ScholarPaper]) -> List[ScholarPaper]:
    cleaned: List[ScholarPaper] = []
    for p in papers:
        url = (p.source_url or "").strip()
        if is_generic_consensus_url(url):
            cleaned.append(p.model_copy(update={"source_url": ""}))
            continue
        cleaned.append(p)
    return cleaned
