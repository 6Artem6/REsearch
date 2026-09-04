"""Pinned context: схемы статей → Main Tutor Chat (только текст, без пикселей)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from knowledge_engine.services.article_diagram_store import list_diagrams_for_article
from knowledge_engine.src.node_deep_dive.schemas import DiagramAsset, NodeDataInput

PINNED_DIAGRAMS_TAG = "[PINNED_DIAGRAMS_CONTEXT]"
_MAX_DIAGRAMS = 6
_MAX_SUMMARY = 400
_MAX_MERMAID_BODY = 900
_MAX_BLOCK_CHARS = 4200

_FENCE_RE = re.compile(r"^```(?:mermaid)?\s*([\s\S]*?)```\s*$", re.I)


def normalize_source_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    from knowledge_engine.src.fetcher.academic import extract_doi

    doi = extract_doi(u)
    if doi:
        return f"https://doi.org/{doi.lower()}"
    u = u.split("#", 1)[0].strip()
    if u.endswith("/"):
        u = u[:-1]
    return u.lower()


def canonical_article_id(source_id: str = "", url: str = "") -> str:
    """
    Стабильный article_id для ingest и lookup.
    source_id + URL → src:{sid}:{url_md8}; только URL → url:{sha256}.
    """
    sid = (source_id or "").strip()
    norm = normalize_source_url(url)
    if sid and norm:
        url_tag = hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]
        return f"src:{sid}:{url_tag}"
    if sid:
        return f"src:{sid}"
    if not norm:
        return ""
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:40]
    return f"url:{digest}"


def resolve_article_ids_for_node(
    node: NodeDataInput,
    curriculum_id: str,
) -> list[str]:
    """article_id для всех источников ноды (primary + mapped + source_ref)."""
    ids: list[str] = []
    seen: set[str] = set()

    def add(sid: str = "", url: str = "") -> None:
        aid = canonical_article_id(sid, url)
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)

    ref = node.source_ref
    if ref is not None:
        add(ref.source_id or "", ref.url or "")

    cid = (curriculum_id or "").strip()
    by_id: dict[str, dict[str, Any]] = {}
    if cid:
        from knowledge_engine.services.skill_tree_store import get_curriculum_graph

        raw = get_curriculum_graph(cid) or {}
        entries = list(raw.get("curriculum_sources_registry") or [])
        for e in entries:
            key = str(e.get("source_id") or "").strip()
            if key:
                by_id[key] = e

    primary = (node.primary_source_id or "").strip()
    if primary:
        # source_id в одиночку, БЕЗ url, схлопывает canonical_article_id в
        # голый "src:{sid}" — а sid ("src_1", "src_2"…) это короткий
        # per-curriculum слот, переиспользуемый почти во ВСЕХ курсах. Без
        # реального URL из registry это коллизирует с диаграммами/
        # figure_registry чужой ноды в другом curriculum, у которой
        # primary_source_id тоже "src_1" (подтверждено на
        # python_internals_and_memory/gil_internals — тянуло диаграммы
        # совершенно другого курса про RAG/vector DB). Резолвим URL так же,
        # как и для mapped_source_ids ниже.
        ent = by_id.get(primary) or {}
        add(primary, str(ent.get("url") or ""))

    mapped = [str(x).strip() for x in (node.mapped_source_ids or []) if str(x).strip()]
    for mid in mapped:
        ent = by_id.get(mid) or {}
        add(mid, str(ent.get("url") or ""))

    return ids


def source_urls_for_node(
    node: NodeDataInput,
    curriculum_id: str,
) -> list[tuple[str, str]]:
    """(url, source_id) для всех привязанных источников ноды."""
    out: list[tuple[str, str]] = []
    seen_url: set[str] = set()

    def add(url: str, sid: str = "") -> None:
        u = (url or "").strip()
        if not u.startswith("http"):
            return
        key = normalize_source_url(u)
        if not key or key in seen_url:
            return
        seen_url.add(key)
        out.append((u, (sid or "").strip()))

    ref = node.source_ref
    if ref is not None:
        add(ref.url or "", ref.source_id or "")

    cid = (curriculum_id or "").strip()
    nid = (node.node_id or "").strip()
    if cid and nid:
        from knowledge_engine.services.skill_tree_store import get_curriculum_graph

        raw = get_curriculum_graph(cid) or {}
        for n in raw.get("nodes") or []:
            if str(n.get("node_id") or "") != nid:
                continue
            for u in n.get("resource_urls") or []:
                add(str(u), str(n.get("primary_source_id") or ""))
            break

    cid = (curriculum_id or "").strip()
    mapped = [str(x).strip() for x in (node.mapped_source_ids or []) if str(x).strip()]
    if cid and mapped:
        from knowledge_engine.services.skill_tree_store import get_curriculum_graph

        raw = get_curriculum_graph(cid) or {}
        entries = list(raw.get("curriculum_sources_registry") or [])
        by_id: dict[str, dict[str, Any]] = {}
        for e in entries:
            key = str(e.get("source_id") or "").strip()
            if key:
                by_id[key] = e
        for mid in mapped:
            ent = by_id.get(mid) or {}
            add(str(ent.get("url") or ""), mid)

    return out


def collect_diagram_rows_for_node(
    node: NodeDataInput,
    curriculum_id: str,
    *,
    max_rows: int = 12,
    extra_urls: list[str] | None = None,
) -> list[Any]:
    """Строки article_diagrams для ноды (pHash dedupe, URL-алиасы source_id)."""
    from knowledge_engine.services.article_diagram_store import (
        list_diagrams_for_article,
        list_diagrams_for_normalized_url,
    )

    rows: list[Any] = []
    seen_phash: set[str] = set()
    seen_row_id: set[str] = set()

    def absorb(batch: list[Any]) -> None:
        for row in batch:
            if len(rows) >= max_rows:
                return
            rid = str(getattr(row, "id", "") or "")
            if rid and rid in seen_row_id:
                continue
            ph = (row.image_phash or "").strip()
            if ph and ph in seen_phash:
                continue
            if not _is_diagram_row(row.mermaid_code, row.summary):
                continue
            if ph:
                seen_phash.add(ph)
            if rid:
                seen_row_id.add(rid)
            rows.append(row)

    for aid in resolve_article_ids_for_node(node, curriculum_id):
        absorb(list_diagrams_for_article(aid))

    for url, sid in source_urls_for_node(node, curriculum_id):
        absorb(list_diagrams_for_normalized_url(url))
        aid = canonical_article_id(sid, url)
        if aid:
            absorb(list_diagrams_for_article(aid))
        absorb(list_diagrams_for_article(canonical_article_id("", url)))

    for raw in extra_urls or []:
        url = str(raw or "").strip()
        if not url.startswith("http"):
            continue
        absorb(list_diagrams_for_normalized_url(url))
        absorb(list_diagrams_for_article(canonical_article_id("", url)))

    return rows[:max_rows]


def build_diagram_assets_for_node(
    node: NodeDataInput,
    curriculum_id: str,
    *,
    max_diagrams: int = 12,
    extra_urls: list[str] | None = None,
) -> list[DiagramAsset]:
    from knowledge_engine.services.mermaid_validate import normalize_stored_mermaid

    assets: list[DiagramAsset] = []
    for row in collect_diagram_rows_for_node(
        node,
        curriculum_id,
        max_rows=max_diagrams,
        extra_urls=extra_urls,
    ):
        mermaid = normalize_stored_mermaid(_mermaid_body(row.mermaid_code))
        if not mermaid:
            continue
        title = (row.caption or row.summary or "").strip()[:200]
        assets.append(
            DiagramAsset(
                id="diagram-1",
                title=title,
                mermaid=mermaid[:8000],
            )
        )
    return assets


def _mermaid_body(code: str) -> str:
    raw = (code or "").strip()
    if not raw:
        return ""
    m = _FENCE_RE.match(raw)
    if m:
        raw = m.group(1).strip()
    return raw[:_MAX_MERMAID_BODY]


def _is_diagram_row(mermaid_code: str, summary: str) -> bool:
    """Эквивалент is_diagram=True: в БД только прошедшие VLM, с непустым Mermaid."""
    return bool(_mermaid_body(mermaid_code))


def format_pinned_diagrams_context(
    article_ids: list[str],
    *,
    article_id: str | None = None,
) -> str:
    """
    Компактный текстовый блок для pinned_context (без image bytes / multimodal).
    """
    aids: list[str] = []
    seen_a: set[str] = set()
    if article_id:
        a = (article_id or "").strip()
        if a:
            aids.append(a)
            seen_a.add(a)
    for a in article_ids or []:
        t = (a or "").strip()
        if t and t not in seen_a:
            seen_a.add(t)
            aids.append(t)
    if not aids:
        return ""

    rows: list[Any] = []
    seen_phash: set[str] = set()
    for aid in aids:
        for row in list_diagrams_for_article(aid):
            ph = (row.image_phash or "").strip()
            if ph and ph in seen_phash:
                continue
            if not _is_diagram_row(row.mermaid_code, row.summary):
                continue
            if ph:
                seen_phash.add(ph)
            rows.append(row)

    if not rows:
        return ""

    lines = [
        PINNED_DIAGRAMS_TAG,
        "Схемы из статей (Mermaid + summary). Приоритет для поля diagram в JSON.",
        "Используй для связки объяснения с архитектурой источника и code_snippets.",
    ]
    total = len("\n".join(lines))
    count = 0
    for row in rows:
        if count >= _MAX_DIAGRAMS:
            break
        title = (row.caption or "").strip()[:200]
        summary = (row.summary or "").strip()[:_MAX_SUMMARY]
        mermaid = _mermaid_body(row.mermaid_code)
        if not mermaid:
            continue
        chunk_lines = [
            f"--- diagram {count + 1} | article={row.article_id} | phash={row.image_phash[:12]}",
        ]
        if title:
            chunk_lines.append(f"title: {title}")
        if summary:
            chunk_lines.append(f"summary: {summary}")
        chunk_lines.append("mermaid:")
        chunk_lines.append(mermaid)
        chunk = "\n".join(chunk_lines)
        if total + len(chunk) > _MAX_BLOCK_CHARS:
            break
        lines.append(chunk)
        total += len(chunk)
        count += 1

    if count == 0:
        return ""
    return "\n\n".join(lines).strip()


def format_article_mermaids_for_source(
    *,
    url: str = "",
    source_id: str = "",
    max_diagrams: int = 4,
    max_mermaid_chars: int = 700,
) -> str:
    """Mermaid из article_diagrams для одного URL/source_id (Lecture RAG chunk)."""
    from knowledge_engine.services.article_diagram_store import (
        list_diagrams_for_article,
    )

    aid = canonical_article_id(source_id, url)
    if not aid:
        return ""
    rows = list_diagrams_for_article(aid)
    if not rows:
        return ""
    lines = ["Схемы из article_diagrams (Mermaid):"]
    n = 0
    for row in rows:
        if n >= max_diagrams:
            break
        mermaid = _mermaid_body(row.mermaid_code)
        if not mermaid:
            continue
        cap = (row.caption or row.summary or "")[:200]
        lines.append(f"--- mermaid {n + 1} | {cap}")
        lines.append(mermaid[:max_mermaid_chars])
        n += 1
    if n == 0:
        return ""
    return "\n".join(lines)


def format_figure_registry_for_source(
    *,
    url: str = "",
    source_id: str = "",
    article_id: str = "",
    max_entries: int = 12,
) -> str:
    """Полный реестр схем (anchor + VLM) для тьютора / Q&A."""
    from knowledge_engine.services.article_ingestion.figure_registry_service import (
        load_figure_registry,
    )

    aid = (article_id or "").strip() or canonical_article_id(source_id, url)
    if not aid:
        return ""
    reg = load_figure_registry(aid)
    if not reg.entries:
        return ""
    lines = ["FigureRegistry (anchors + VLM):"]
    n = 0
    for ent in reg.entries.values():
        if n >= max_entries:
            break
        labels = ", ".join(ent.labels[:4]) if ent.labels else ent.internal_id
        lines.append(f"--- {ent.internal_id} | {labels} | page={ent.page_no}")
        if ent.caption:
            lines.append(f"caption: {ent.caption[:300]}")
        if ent.vlm_summary:
            lines.append(f"vlm: {ent.vlm_summary[:1200]}")
        if ent.mermaid_code:
            lines.append(f"mermaid: {ent.mermaid_code[:700]}")
        n += 1
    if n == 0:
        return ""
    return "\n".join(lines)


def build_pinned_diagrams_for_node(
    node: NodeDataInput,
    curriculum_id: str,
) -> str:
    aids = resolve_article_ids_for_node(node, curriculum_id)
    return format_pinned_diagrams_context(aids)


def build_figure_registry_for_node(
    node: NodeDataInput,
    curriculum_id: str,
    *,
    max_entries: int = 12,
) -> str:
    """Реестр фигур (anchor + VLM) для тьютора / Q&A."""
    aids = resolve_article_ids_for_node(node, curriculum_id)
    if not aids:
        return ""
    parts: list[str] = []
    for aid in aids:
        block = format_figure_registry_for_source(
            article_id=aid, max_entries=max_entries
        )
        if block.strip():
            parts.append(block)
    return "\n\n".join(parts).strip()


def build_lecture_pinned_diagrams_block(
    node: NodeDataInput,
    curriculum_id: str,
) -> str:
    """
    Блок для user payload лекции: caption + context (summary) + Mermaid.
    """
    aids = resolve_article_ids_for_node(node, curriculum_id)
    if not aids:
        return ""

    rows: list[Any] = []
    seen_phash: set[str] = set()
    for aid in aids:
        for row in list_diagrams_for_article(aid):
            ph = (row.image_phash or "").strip()
            if ph and ph in seen_phash:
                continue
            if not _is_diagram_row(row.mermaid_code, row.summary):
                continue
            if ph:
                seen_phash.add(ph)
            rows.append(row)

    if not rows:
        return ""

    lines = [
        "### ИМЕЮЩИЕСЯ СХЕМЫ В ИНТЕРФЕЙСЕ (ОБЯЗАТЕЛЬНО ДЛЯ ССЫЛОК В ТЕКСТЕ):",
        PINNED_DIAGRAMS_TAG,
    ]
    total = len("\n".join(lines))
    count = 0
    diagram_titles: list[str] = []
    for row in rows:
        if count >= _MAX_DIAGRAMS:
            break
        cap = (row.caption or "").strip()[:200]
        ctx = (row.summary or row.caption or "").strip()[:_MAX_SUMMARY]
        mermaid = _mermaid_body(row.mermaid_code)
        if not mermaid:
            continue
        line = (
            f"- [Diagram {count + 1}]: \"{cap or '(без подписи)'}\" "
            f"| Описание: {ctx or '(нет описания)'}"
        )
        diagram_titles.append(f"[Diagram {count + 1}]")
        chunk_lines = [line]
        chunk_lines.append("Mermaid structure:")
        chunk_lines.append(mermaid[:_MAX_MERMAID_BODY])
        chunk = "\n".join(chunk_lines)
        if total + len(chunk) > _MAX_BLOCK_CHARS:
            break
        lines.append(chunk)
        total += len(chunk)
        count += 1

    if count == 0:
        return ""

    refs = ", ".join(diagram_titles)
    lines.append(
        f"ИНСТРУКЦИЯ: Ссылайся только на {refs} и только если эти номера совпадают с "
        "блоком DIAGRAM_CATALOG (панель Materials). Без выдуманных подписей «[Diagram N: …]». "
        "Формат: [Diagram N] или [diagram:diagram-N] + разбор mermaid ниже."
    )
    return "\n\n".join(lines).strip()
