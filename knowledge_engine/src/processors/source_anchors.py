"""Сквозные source anchors [S1], [S2] для Consensus → L2 → Reasoner → UI."""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Mapping
from urllib.parse import urlparse

from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper

_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]")
_MD_SOURCE_ANCHOR_LINK_RE = re.compile(r"\[\[S(\d+)\]\]\([^)]*\)")


def strip_source_anchor_tags(text: str) -> str:
    """Убрать [S1], [[S1]](url) — для fact nuggets в памяти и UI без сносок."""
    if not text:
        return text
    s = _MD_SOURCE_ANCHOR_LINK_RE.sub("", text)
    s = _SOURCE_TAG_RE.sub("", s)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def strip_source_anchor_tags_list(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        clean = strip_source_anchor_tags(item)
        if clean:
            out.append(clean)
    return out


SOURCE_ANCHOR_RETENTION_PROMPT = (
    "SOURCE ANCHORS: The context includes a SOURCE REGISTRY with ids S1, S2, S3, …\n"
    "DO NOT remove or summarize away inline source tags like [S1], [S2]. "
    "Every chunk, concept, tradeoff point, factual claim, metric, or mechanism MUST retain "
    "its original source anchor [Sx] immediately after the claim. "
    "If a statement comes from multiple papers, list all tags without spaces: [S1][S3]. "
    "Use ONLY ids present in the SOURCE REGISTRY — never invent new ids."
)

REASONER_SOURCE_ATTRIBUTION_PROMPT = (
    "SOURCE ATTRIBUTION (mandatory):\n"
    "1. Every factual claim, technical mechanism, algorithm recommendation, complexity bound, "
    "or trade-off statement in user_final_answer MUST cite inline tags [S1], [S2], … "
    "from the SOURCE REGISTRY.\n"
    "2. Use ONLY source ids that appear in the registry / valid_docs / scholarly_papers blocks.\n"
    "3. At the very END of user_final_answer, add a section "
    "## Источники (Source registry) with one Markdown bullet per registry entry, "
    "format: `- [S1] Authors, Title, DOI or URL as markdown link`.\n"
    "4. Keep inline [Sx] tags in the body even when the reference list is present."
)


def _normalize_url(url: str) -> str:
    u = (url or "").strip().lower()
    if not u:
        return ""
    parsed = urlparse(u)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _doi_from_paper(p: Mapping[str, Any]) -> str:
    doi = (p.get("doi") or "").strip()
    if doi:
        return doi
    ext = p.get("external_ids") or {}
    if isinstance(ext, dict):
        return str(ext.get("DOI") or ext.get("doi") or "").strip()
    return ""


def _authors_line(p: Mapping[str, Any]) -> str:
    authors = p.get("authors")
    if isinstance(authors, list):
        names = [
            str(a.get("name", a) if isinstance(a, dict) else a) for a in authors[:4]
        ]
        return ", ".join(n for n in names if n)
    if isinstance(authors, str) and authors.strip():
        return authors.strip()
    return "et al."


def build_source_registry(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Назначить S1… и записать source_anchor в каждый paper dict."""
    registry: List[Dict[str, Any]] = []
    seen: set[str] = set()
    idx = 0
    for raw in papers:
        if not isinstance(raw, dict):
            continue
        url = (raw.get("source_url") or raw.get("url") or "").strip()
        norm = _normalize_url(url)
        title = (raw.get("title") or "paper").strip()
        key = norm or title.lower()
        if key in seen:
            for entry in registry:
                if _normalize_url(entry.get("url") or "") == norm or (
                    not norm and entry.get("title", "").lower() == title.lower()
                ):
                    raw["source_anchor"] = entry["id"]
                    break
            continue
        seen.add(key)
        idx += 1
        sid = f"S{idx}"
        doi = _doi_from_paper(raw)
        entry = {
            "id": sid,
            "tag": f"[{sid}]",
            "title": title,
            "url": url,
            "doi": doi,
            "authors": _authors_line(raw),
            "year": raw.get("year"),
            "snippet": (
                raw.get("abstract") or raw.get("tldr") or raw.get("snippet") or ""
            )[:800],
            "venue": raw.get("venue") or "",
        }
        registry.append(entry)
        raw["source_anchor"] = sid
    return registry


def url_to_source_id_map(registry: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for entry in registry:
        sid = entry.get("id") or ""
        url = entry.get("url") or ""
        if sid and url:
            out[_normalize_url(url)] = sid
    return out


def resolve_source_anchor_for_url(
    url: str,
    url_map: Mapping[str, str],
    papers: List[Dict[str, Any]],
) -> str:
    sid = url_map.get(_normalize_url(url))
    if sid:
        return sid
    for p in papers:
        if _normalize_url(p.get("source_url") or p.get("url") or "") == _normalize_url(
            url
        ):
            return str(p.get("source_anchor") or "")
    return ""


def format_registry_for_prompt(registry: List[Dict[str, Any]]) -> str:
    if not registry:
        return "(SOURCE REGISTRY empty)"
    lines = ["### SOURCE REGISTRY (cite inline as [S1], [S2], …)"]
    for entry in registry:
        sid = entry.get("id") or "?"
        title = entry.get("title") or "paper"
        authors = entry.get("authors") or "et al."
        url = entry.get("url") or "—"
        doi = entry.get("doi") or ""
        doi_part = f" | DOI: {doi}" if doi else ""
        snippet = (entry.get("snippet") or "")[:400]
        lines.append(
            f"[{sid}] Title: {title} | Authors: {authors}{doi_part} | URL: {url}\n"
            f"   Summary: {snippet}"
        )
    return "\n".join(lines)


def format_papers_block_with_anchors(papers: List[ScholarPaper]) -> str:
    if not papers:
        return "(нет статей в retrieval)"
    blocks: List[str] = []
    for i, p in enumerate(papers, start=1):
        sid = f"S{i}"
        lines = [
            f"[{sid}] {p.title}",
            f"Year: {p.year or '—'} | Venue: {p.venue or '—'} | Citations: {p.citation_count}",
            f"URL: {p.source_url}",
        ]
        if p.pdf_url:
            lines.append(f"Open PDF: {p.pdf_url}")
        if p.tldr:
            lines.append(f"TLDR: {p.tldr}")
        if p.abstract:
            lines.append(f"Abstract: {p.abstract[:2500]}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def format_valid_docs_for_reasoner(
    docs: List[Dict[str, Any]],
    registry: List[Dict[str, Any]],
) -> str:
    if registry:
        by_url = url_to_source_id_map(registry)
        lines: List[str] = []
        for d in docs:
            url = (d.get("url") or "").strip()
            sid = d.get("source_anchor") or by_url.get(_normalize_url(url)) or "?"
            title = d.get("title") or "paper"
            snippet = (d.get("snippet") or "")[:800]
            lines.append(f"[{sid}] {title}\n   URL: {url}\n   {snippet}")
        return "\n".join(lines) if lines else "(нет valid_docs)"
    return "(нет реестра — см. scholarly_papers)"


def expand_source_tags_to_markdown_links(
    text: str,
    registry: List[Dict[str, Any]] | None,
) -> str:
    if not text or not registry:
        return text
    by_id = {str(e.get("id")): e for e in registry if e.get("id")}

    def _repl(m: re.Match[str]) -> str:
        sid = f"S{m.group(1)}"
        ent = by_id.get(sid)
        if not ent:
            return m.group(0)
        url = (ent.get("url") or "").strip()
        doi = (ent.get("doi") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"
        if url:
            return f"[[{sid}]]({url})"
        return f"**[{sid}]**"

    return _SOURCE_TAG_RE.sub(_repl, text)


def linkify_source_anchors_html(
    html_text: str,
    registry: List[Dict[str, Any]] | None,
) -> str:
    if not html_text or not registry:
        return html_text
    by_id = {str(e.get("id")): e for e in registry if e.get("id")}

    def _repl(m: re.Match[str]) -> str:
        sid = f"S{m.group(1)}"
        ent = by_id.get(sid)
        if not ent:
            return m.group(0)
        title = (ent.get("title") or sid).replace('"', "&quot;")
        url = (ent.get("url") or "").strip()
        doi = (ent.get("doi") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"
        if url:
            safe = html.escape(url, quote=True)
            return (
                f'<a href="{safe}" class="source-anchor" target="_blank" '
                f'rel="noopener noreferrer" title="{title}">[{sid}]</a>'
            )
        return f'<span class="source-anchor" title="{title}">[{sid}]</span>'

    return _SOURCE_TAG_RE.sub(_repl, html_text)
