"""Build TOC + HTML sections for v0.7 web UI."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from knowledge_engine.src.processors.source_anchors import strip_source_anchor_tags
from knowledge_engine.web.linkify import (
    linkify_references,
    markdown_document_html,
    paragraphs_html,
)
from knowledge_engine.web.source_present import document_source_li, scholarly_paper_li


def _esc(s: str) -> str:
    from html import escape

    return escape(s or "")


def _paper_list_item(title: str, url: str, snippet: str = "") -> str:
    return scholarly_paper_li(
        {
            "title": title,
            "source_url": url,
            "snippet": snippet,
        }
    )


def build_ui_view(result: Dict[str, Any]) -> dict[str, Any]:
    """Structured payload for SPA: toc + sections + sources."""
    registry = result.get("source_registry") or []
    if isinstance(registry, list) and not registry:
        registry = []
    toc: List[dict[str, str]] = []
    sections: List[dict[str, str]] = []

    def add_section(
        section_id: str, title: str, body_html: str, level: int = 2
    ) -> None:
        toc.append({"id": section_id, "title": title, "level": str(level)})
        sections.append({"id": section_id, "title": title, "html": body_html})

    query = str(result.get("user_query") or "")
    mode = str(result.get("retrieval_mode") or "fast").strip().lower()
    mode_label = "Consensus" if mode == "consensus" else "Fast (Reasoner)"
    add_section(
        "query",
        "Запрос",
        f"<p>{_esc(query)}</p><p class='muted'>Режим: <strong>{_esc(mode_label)}</strong></p>",
        level=1,
    )

    academic_q = (result.get("consensus_academic_query") or "").strip()
    if academic_q:
        body = f"<p>{_esc(academic_q)}</p>"
        preserved = result.get("consensus_preserved_terms") or []
        if preserved:
            body += (
                "<p class='muted'><strong>Термины (verbatim):</strong> "
                + _esc(", ".join(str(t) for t in preserved))
                + "</p>"
            )
        add_section(
            "consensus_query",
            "Запрос в Consensus (EN)",
            body,
        )

    vstatus = (result.get("validation_status") or "").strip()
    if vstatus and vstatus != "FAST_MODE":
        reason = _esc(str(result.get("validation_reason") or ""))
        add_section(
            "validation",
            "Валидация Consensus",
            f"<p><strong>{_esc(vstatus)}</strong></p><p class='muted'>{reason}</p>",
        )

    docs_meta = result.get("consensus_docs") or []
    scholarly = result.get("scholarly_papers") or []
    paper_rows: List[str] = []
    seen_titles: set[str] = set()
    for source in (scholarly, docs_meta):
        for item in source:
            if not isinstance(item, dict):
                continue
            url = (item.get("source_url") or item.get("url") or "").strip()
            title = (item.get("title") or "paper").strip()
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            snippet = (
                item.get("abstract") or item.get("tldr") or item.get("snippet") or ""
            )[:400]
            anchor = (item.get("source_anchor") or "").strip()
            if anchor:
                title = f"[{anchor}] {title}"
            paper_rows.append(_paper_list_item(title, url, snippet))
    if paper_rows:
        add_section(
            "scholarly_papers",
            "Публикации (Consensus)",
            "<ul class='source-list'>" + "".join(paper_rows) + "</ul>",
        )

    spec = result.get("query_spec") or {}
    if isinstance(spec, dict) and spec.get("cs_formal_query"):
        add_section(
            "formal",
            "Формальная CS-формулировка",
            paragraphs_html(str(spec["cs_formal_query"]), registry),
        )

    docs = result.get("documents") or []
    if docs:
        rows: List[str] = ["<ul class='source-list'>"]
        for d in docs:
            if not isinstance(d, dict):
                continue
            rows.append(document_source_li(d))
        rows.append("</ul>")
        add_section("sources", "Источники", "\n".join(rows))

    cg = result.get("concept_graph") or {}
    if isinstance(cg, dict):
        placeholder = cg.get("task_summary") == "Нет чанков для анализа"
        placeholder |= cg.get("task_summary") == "Нет источников для анализа"
        parts: List[str] = []
        if cg.get("research_synthesis"):
            parts.append(
                "<h3>Синтез</h3>"
                + paragraphs_html(str(cg["research_synthesis"]), registry)
            )
        if cg.get("task_summary") and not placeholder:
            parts.append(
                "<h3>Сводка</h3>" + paragraphs_html(str(cg["task_summary"]), registry)
            )
        for label, key in (
            ("Подводные камни", "engineering_pitfalls"),
            ("Теория ↔ практика", "theory_practice_bridges"),
            ("Инварианты", "invariants"),
        ):
            items = cg.get(key) or []
            if items:
                parts.append(f"<h3>{label}</h3><ul>")
                for it in items:
                    parts.append(f"<li>{linkify_references(str(it), registry)}</li>")
                parts.append("</ul>")
        contrasts = cg.get("cross_source_contrasts") or []
        if contrasts:
            parts.append("<h3>Сравнение подходов</h3>")
            for c in contrasts:
                if not isinstance(c, dict):
                    continue
                parts.append(
                    "<div class='contrast-card'>"
                    f"<strong>{_esc(str(c.get('topic', '')))}</strong>"
                    f"<p>A: {linkify_references(str(c.get('approach_a', '')), registry)}</p>"
                    f"<p>B: {linkify_references(str(c.get('approach_b', '')), registry)}</p>"
                    f"<p class='muted'>{linkify_references(str(c.get('principal_difference', '')), registry)}</p>"
                    "</div>"
                )
        if parts:
            add_section("l2a", "L2a — Синтез и концепты", "\n".join(parts))

    gap = result.get("profile_gap_map") or {}
    if isinstance(gap, dict):
        parts: List[str] = []
        if gap.get("context_synthesis"):
            parts.append(paragraphs_html(str(gap["context_synthesis"]), registry))
        for label, key in (
            ("Столкновения допущений", "assumption_clashes"),
            ("Контекстные флаги", "context_flags"),
        ):
            items = gap.get(key) or []
            if items:
                parts.append(f"<h3>{label}</h3><ul>")
                for it in items:
                    parts.append(f"<li>{linkify_references(str(it), registry)}</li>")
                parts.append("</ul>")
        if parts:
            add_section("l2b", "L2b — Условия и контекст", "\n".join(parts))

    matrix = result.get("tradeoff_matrix") or []
    if matrix:
        cards: List[str] = []
        for row in matrix:
            if not isinstance(row, dict):
                continue
            title = f"{row.get('pattern_name', '')} ({row.get('column', '')})"
            body_parts = []
            for key, label in (
                ("fundamental_idea", "Идея"),
                ("mechanics_detail", "Механика"),
                ("applicability", "Применимость"),
            ):
                if row.get(key):
                    body_parts.append(
                        f"<p><strong>{label}</strong>: "
                        f"{linkify_references(str(row[key]), registry)}</p>"
                    )
            for key, label in (("pros", "Плюсы"), ("cons_and_risks", "Риски")):
                items = row.get(key) or []
                if items:
                    body_parts.append(f"<p><strong>{label}</strong></p><ul>")
                    for it in items:
                        body_parts.append(
                            f"<li>{linkify_references(str(it), registry)}</li>"
                        )
                    body_parts.append("</ul>")
            align = row.get("aligning_sources") or []
            if align:
                body_parts.append(
                    "<details class='ke-details ke-details-closed-default'>"
                    "<summary>Источники</summary><ul>"
                )
                for it in align:
                    body_parts.append(
                        f"<li>{linkify_references(str(it), registry)}</li>"
                    )
                body_parts.append("</ul></details>")
            cards.append(
                f"<article class='matrix-card'>"
                f"<h3>{_esc(title)}</h3>" + "".join(body_parts) + "</article>"
            )
        add_section("l2c", "L2c — Архитектурные варианты", "\n".join(cards))

    step = (result.get("current_step") or "").strip()
    final_answer = (result.get("user_final_answer") or "").strip()
    if step == "reasoner" and not final_answer:
        add_section(
            "reasoner_pending",
            "Ответ (Gemini Reasoner)",
            "<p class='muted'>Генерация финального ответа…</p>",
            level=1,
        )

    if final_answer:
        add_section(
            "final_answer",
            "Ответ (Gemini Reasoner)",
            markdown_document_html(final_answer, registry),
            level=1,
        )

    nuggets = result.get("fact_nuggets") or []
    if nuggets:
        clean_nuggets = [strip_source_anchor_tags(str(n)) for n in nuggets]
        clean_nuggets = [n for n in clean_nuggets if n]
        add_section(
            "facts",
            "Факты в Light RAG",
            "<ul>" + "".join(f"<li>{_esc(n)}</li>" for n in clean_nuggets) + "</ul>",
        )

    meta = {
        "step": result.get("current_step"),
        "pipeline": result.get("pipeline_version") or "",
        "depth": result.get("search_depth"),
        "docs": len(docs),
        "papers": len(result.get("scholarly_papers") or []),
        "chunks": len(result.get("structured_chunks") or []),
    }

    return {
        "toc": toc,
        "sections": sections,
        "meta": meta,
        "raw_json": json.dumps(result, ensure_ascii=False, default=str),
    }
