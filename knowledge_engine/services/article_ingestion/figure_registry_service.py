"""Persistent FigureRegistry (SQLite) + VLM sync."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field

from sqlalchemy import delete, select

from knowledge_engine.db.session import db_session, init_db
from knowledge_engine.models.figure_registry import FigureRegistryRow
from knowledge_engine.services.article_diagram_store import list_diagrams_for_article
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    WindowDiagramCheck,
)
from knowledge_engine.services.article_ingestion.figure_anchor_mapper import (
    FigureAnchor,
    build_figure_anchors,
)
from knowledge_engine.services.article_ingestion.spatial_diagram_dispatch import (
    ingest_target_diagrams,
)
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle
from knowledge_engine.ui.run_log import trace


@dataclass
class RegistryEntry:
    internal_id: str
    labels: list[str] = field(default_factory=list)
    caption: str = ""
    page_no: int = 0
    anchor_p_ids: list[str] = field(default_factory=list)
    extract_source: str = ""
    vlm_summary: str = ""
    mermaid_code: str = ""
    image_phash: str = ""


@dataclass
class FigureRegistry:
    article_id: str
    entries: dict[str, RegistryEntry] = field(default_factory=dict)

    def get(self, fig_id: str) -> RegistryEntry | None:
        fid = fig_id.strip().upper()
        if not fid.startswith("FIG"):
            fid = f"FIG_{fid}"
        return self.entries.get(fid) or self.entries.get(fig_id)

    def lookup_by_label(self, label: str) -> RegistryEntry | None:
        low = (label or "").strip().lower()
        for ent in self.entries.values():
            for lab in ent.labels:
                if lab.lower() == low:
                    return ent
        m = __import__("re").match(r"fig(?:ure)?\.?\s*(\d+)", low, __import__("re").I)
        if m:
            key = f"FIG_{int(m.group(1))}"
            return self.entries.get(key)
        return None


def ensure_figure_registry_schema() -> None:
    init_db()


def _row_to_entry(row: FigureRegistryRow) -> RegistryEntry:
    try:
        labels = json.loads(row.labels_json or "[]")
    except json.JSONDecodeError:
        labels = []
    try:
        pids = json.loads(row.anchor_p_ids_json or "[]")
    except json.JSONDecodeError:
        pids = []
    return RegistryEntry(
        internal_id=row.internal_fig_id,
        labels=list(labels) if isinstance(labels, list) else [],
        caption=row.caption or "",
        page_no=int(row.page_no or 0),
        anchor_p_ids=list(pids) if isinstance(pids, list) else [],
        extract_source=row.extract_source or "",
        vlm_summary=row.vlm_summary or "",
        mermaid_code=row.mermaid_code or "",
        image_phash=row.image_phash or "",
    )


def load_figure_registry(article_id: str) -> FigureRegistry:
    aid = (article_id or "").strip()
    ensure_figure_registry_schema()
    entries: dict[str, RegistryEntry] = {}
    with db_session() as session:
        stmt = (
            select(FigureRegistryRow)
            .where(FigureRegistryRow.article_id == aid)
            .order_by(FigureRegistryRow.internal_fig_id)
        )
        for row in session.scalars(stmt).all():
            ent = _row_to_entry(row)
            entries[ent.internal_id] = ent
    return FigureRegistry(article_id=aid, entries=entries)


def persist_figure_registry(
    article_id: str,
    annotated: AnnotatedArticle,
    anchors: dict[str, FigureAnchor] | None = None,
) -> FigureRegistry:
    aid = (article_id or "").strip()
    anchors = anchors or build_figure_anchors(annotated)
    sources = annotated.fig_extract_source or {}
    ensure_figure_registry_schema()

    with db_session() as session:
        session.execute(
            delete(FigureRegistryRow).where(FigureRegistryRow.article_id == aid)
        )
        for fid, anchor in anchors.items():
            if fid not in (annotated.fig_bytes or {}) and fid not in (
                annotated.fig_map or {}
            ):
                continue
            src = sources.get(fid, "unknown")
            if str(src).startswith("invalid:"):
                continue
            row = FigureRegistryRow(
                id=str(uuid.uuid4()),
                article_id=aid,
                internal_fig_id=fid,
                labels_json=json.dumps(anchor.labels, ensure_ascii=False),
                caption=(anchor.caption or "")[:2000],
                page_no=int(anchor.page_no or 0),
                anchor_p_ids_json=json.dumps(anchor.anchor_p_ids, ensure_ascii=False),
                extract_source=str(src)[:64],
            )
            session.add(row)

    reg = load_figure_registry(aid)
    trace(
        f"FIG_REGISTRY ✓ | article={aid[:40]} entries={len(reg.entries)} " f"(pre-VLM)"
    )
    return reg


def delete_figure_registry_for_urls(urls: list[str]) -> int:
    """Удалить figure_registry-строки для URL — тот же матчинг article_id,
    что и `article_diagram_store.delete_diagrams_for_urls` (обе таблицы
    ключуются одинаковым `canonical_article_id(source_id, url)`). Без этого
    VLM-разбор фигур не перезапускается при повторном clear+re-ingest ноды —
    просто переиспользуется старая запись."""
    from knowledge_engine.services.article_diagram_context import (
        canonical_article_id,
        normalize_source_url,
    )

    ensure_figure_registry_schema()
    removed = 0
    with db_session() as session:
        for raw in urls:
            u = (raw or "").strip()
            if not u.startswith("http"):
                continue
            norm = normalize_source_url(u)
            if not norm:
                continue
            aid_no_source = canonical_article_id("", u)
            tag = hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]
            suffix = f":{tag}"
            stmt = select(FigureRegistryRow).where(
                (FigureRegistryRow.article_id == aid_no_source)
                | (FigureRegistryRow.article_id.endswith(suffix))
            )
            rows = list(session.scalars(stmt).all())
            for row in rows:
                session.delete(row)
                removed += 1
    return removed


def run_vlm_on_registry(
    article_id: str,
    annotated: AnnotatedArticle,
    registry: FigureRegistry,
    *,
    source_id: str = "",
    page_url: str = "",
) -> int:
    """VLM для всех записей реестра с байтами (до Map-фазы)."""
    checks: list[WindowDiagramCheck] = []
    for ent in registry.entries.values():
        if ent.internal_id not in (annotated.fig_bytes or {}):
            continue
        reason = ent.caption or ", ".join(ent.labels[:3]) or ent.internal_id
        checks.append(
            WindowDiagramCheck(
                figure_id=ent.internal_id,
                referenced_paragraphs=list(ent.anchor_p_ids),
                reason=reason[:500],
            )
        )
    if not checks:
        trace("FIG_REGISTRY vlm ⊘ | no renderable figures")
        return 0
    n = ingest_target_diagrams(
        article_id,
        annotated,
        checks,
        source_id=source_id,
        page_url=page_url or annotated.page_url,
    )
    sync_registry_from_article_diagrams(article_id, registry)
    trace(f"FIG_REGISTRY vlm ✓ | saved={n}")
    return n


def sync_registry_from_article_diagrams(
    article_id: str,
    registry: FigureRegistry,
) -> None:
    """Подтянуть summary/mermaid из article_diagrams в figure_registry."""
    import re

    aid = (article_id or "").strip()
    diagrams = list_diagrams_for_article(aid)
    if not diagrams:
        return
    by_fig: dict[str, tuple[str, str, str]] = {}
    for d in diagrams:
        cap = d.caption or ""
        m = re.match(r"^(FIG(?:_SEQ)?_\d+)", cap.strip(), re.I)
        if not m:
            for ent in registry.entries.values():
                if ent.internal_id in cap:
                    m = re.match(r"^(FIG(?:_SEQ)?_\d+)", ent.internal_id, re.I)
                    break
        if not m:
            continue
        fid = m.group(1).upper()
        by_fig[fid] = (d.summary or "", d.mermaid_code or "", d.image_phash or "")

    ensure_figure_registry_schema()
    with db_session() as session:
        for fid, (summary, mermaid, phash) in by_fig.items():
            stmt = (
                select(FigureRegistryRow)
                .where(FigureRegistryRow.article_id == aid)
                .where(FigureRegistryRow.internal_fig_id == fid)
                .limit(1)
            )
            row = session.scalar(stmt)
            if row is None:
                continue
            row.vlm_summary = summary
            row.mermaid_code = mermaid
            if phash:
                row.image_phash = phash
            ent = registry.entries.get(fid)
            if ent:
                ent.vlm_summary = summary
                ent.mermaid_code = mermaid
                ent.image_phash = phash
