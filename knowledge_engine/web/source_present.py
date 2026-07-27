"""HTML для списков источников в web UI."""

from __future__ import annotations

from knowledge_engine.web.linkify import (
    arxiv_link_html,
    doi_link_html,
    extract_arxiv_id_from_text,
    extract_doi_from_text,
)


def _esc(s: str) -> str:
    from html import escape

    return escape(s or "")


def _is_consensus_paper_url(url: str) -> bool:
    u = (url or "").lower()
    return "consensus.app/papers/" in u or "consensus.app/paper/" in u


def _secondary_refs_html(primary_url: str, blob: str) -> str:
    """DOI / arXiv — не дублировать ту же consensus.app URL под заголовком."""
    parts: list[str] = []
    doi = extract_doi_from_text(blob)
    if doi:
        parts.append(doi_link_html(doi))
    arxiv_id = extract_arxiv_id_from_text(blob)
    if arxiv_id:
        parts.append(arxiv_link_html(arxiv_id))
    if not parts and primary_url and _is_consensus_paper_url(primary_url):
        parts.append(
            "<span class='muted source-hint'>страница Consensus "
            "(откройте заголовок; DOI не извлечён из текста)</span>"
        )
    return " ".join(parts)


def _no_link_reason(title: str, url: str, doc_id: str) -> str:
    from knowledge_engine.src.retrieval.consensus_capture import (
        is_generic_consensus_url,
    )

    if not (url or "").strip():
        if doc_id.startswith("scholar_") and len(title) > 100:
            return "текст из Consensus API без URL статьи"
        return "нет внешнего URL"
    if is_generic_consensus_url(url):
        return "только consensus.app/home/search, не карточка paper"
    if len(title) > 140 and "?" not in title[:80]:
        return "фрагмент ответа, не название публикации"
    return "URL не прошёл фильтр (generic consensus)"


def document_source_li(d: dict) -> str:
    from knowledge_engine.src.retrieval.consensus_capture import (
        is_generic_consensus_url,
    )

    url = (d.get("source_url") or "").strip()
    title = (d.get("title") or d.get("doc_id") or "document").strip()
    doc_id = _esc(str(d.get("doc_id") or ""))
    pdf = "PDF" if d.get("is_pdf") else "HTML"
    blob = (d.get("raw_markdown") or "")[:8000]
    if not blob and isinstance(d.get("snippet"), str):
        blob = d.get("snippet") or ""

    primary = (
        url if url.startswith("http") and not is_generic_consensus_url(url) else ""
    )

    if primary:
        from html import escape

        safe_url = escape(primary, quote=True)
        secondary = _secondary_refs_html(primary, blob)
        sec_block = (
            f"<span class='source-secondary'>{secondary}</span>" if secondary else ""
        )
        return (
            f"<li><span class='badge'>{pdf}</span> "
            f"<a class='source-link' href='{safe_url}' target='_blank' rel='noopener'>"
            f"{_esc(title)}</a> "
            f"<span class='muted doc-id'>{doc_id}</span>{sec_block}</li>"
        )

    reason = _no_link_reason(title, url, str(d.get("doc_id") or ""))
    short_title = title if len(title) <= 200 else title[:197] + "…"
    return (
        f"<li><span class='badge badge-warn' title='{_esc(reason)}'>без ссылки</span> "
        f"<span class='source-title-plain'>{_esc(short_title)}</span> "
        f"<span class='muted doc-id'>{doc_id}</span> "
        f"<span class='muted source-hint'>{_esc(reason)}</span></li>"
    )


def scholarly_paper_li(item: dict) -> str:
    from knowledge_engine.src.retrieval.consensus_capture import (
        is_generic_consensus_url,
    )

    title = (item.get("title") or "paper").strip()
    url = (item.get("source_url") or item.get("url") or "").strip()
    snippet = (item.get("abstract") or item.get("tldr") or item.get("snippet") or "")[
        :400
    ]
    blob = f"{snippet}\n{url}"

    if url.startswith("http") and not is_generic_consensus_url(url):
        from html import escape

        safe_url = escape(url, quote=True)
        snip = f"<p class='muted snippet'>{_esc(snippet)}</p>" if snippet else ""
        secondary = _secondary_refs_html(url, blob)
        sec = f"<span class='source-secondary'>{secondary}</span>" if secondary else ""
        return (
            f"<li><a class='source-link' href='{safe_url}' target='_blank' rel='noopener'>"
            f"{_esc(title)}</a>{sec}{snip}</li>"
        )

    reason = _no_link_reason(title, url, "")
    body = f"<span class='badge badge-warn'>без ссылки</span> <span class='source-title-plain'>{_esc(title[:200])}</span>"
    if snippet:
        body += f"<p class='muted snippet'>{_esc(snippet)}</p>"
    body += f"<p class='muted source-hint'>{_esc(reason)}</p>"
    return f"<li>{body}</li>"
