"""CRUD для article_diagrams (pHash cache)."""

from __future__ import annotations

import hashlib
import uuid

from sqlalchemy import func, select

from knowledge_engine.db.session import db_session, init_db
from knowledge_engine.models.article_diagrams import ArticleDiagram


def ensure_article_diagrams_schema() -> None:
    init_db()


def phash_exists(image_phash: str) -> bool:
    ph = (image_phash or "").strip()
    if not ph:
        return False
    ensure_article_diagrams_schema()
    with db_session() as session:
        stmt = select(ArticleDiagram).where(ArticleDiagram.image_phash == ph).limit(1)
        return session.scalar(stmt) is not None


def has_diagrams_for_article(article_id: str) -> bool:
    aid = (article_id or "").strip()
    if not aid:
        return False
    ensure_article_diagrams_schema()
    with db_session() as session:
        stmt = (
            select(func.count())
            .select_from(ArticleDiagram)
            .where(ArticleDiagram.article_id == aid)
        )
        return int(session.scalar(stmt) or 0) > 0


def save_diagram(
    article_id: str,
    image_phash: str,
    caption: str,
    mermaid_code: str,
    summary: str,
) -> str:
    from knowledge_engine.services.mermaid_validate import normalize_stored_mermaid

    ensure_article_diagrams_schema()
    mermaid = normalize_stored_mermaid(mermaid_code or "")
    row = ArticleDiagram(
        id=str(uuid.uuid4()),
        article_id=(article_id or "").strip(),
        image_phash=(image_phash or "").strip(),
        caption=(caption or "")[:2000],
        mermaid_code=mermaid,
        summary=(summary or ""),
    )
    with db_session() as session:
        session.add(row)
        session.flush()
        return str(row.id)


def update_diagram_by_phash(
    article_id: str,
    image_phash: str,
    *,
    mermaid_code: str,
    caption: str | None = None,
    summary: str | None = None,
) -> bool:
    """Обновить mermaid (и опционально caption/summary) для существующей строки."""
    from knowledge_engine.services.mermaid_validate import normalize_stored_mermaid

    aid = (article_id or "").strip()
    ph = (image_phash or "").strip()
    if not aid or not ph:
        return False
    ensure_article_diagrams_schema()
    from sqlalchemy import update

    mermaid = normalize_stored_mermaid(mermaid_code or "")
    with db_session() as session:
        stmt = (
            update(ArticleDiagram)
            .where(ArticleDiagram.article_id == aid)
            .where(ArticleDiagram.image_phash == ph)
            .values(
                mermaid_code=mermaid,
                **({"caption": (caption or "")[:2000]} if caption is not None else {}),
                **({"summary": (summary or "")} if summary is not None else {}),
            )
        )
        result = session.execute(stmt)
        return int(result.rowcount or 0) > 0


def list_diagrams_for_article(article_id: str) -> list[ArticleDiagram]:
    aid = (article_id or "").strip()
    if not aid:
        return []
    ensure_article_diagrams_schema()
    with db_session() as session:
        stmt = (
            select(ArticleDiagram)
            .where(ArticleDiagram.article_id == aid)
            .order_by(ArticleDiagram.created_at)
        )
        return list(session.scalars(stmt).all())


def list_diagrams_for_normalized_url(url: str) -> list[ArticleDiagram]:
    """Диаграммы для URL: src:*:md5tag и url:sha256 (canonical без source_id)."""
    from knowledge_engine.services.article_diagram_context import (
        canonical_article_id,
        normalize_source_url,
    )

    norm = normalize_source_url(url)
    if not norm:
        return []
    ensure_article_diagrams_schema()
    seen_id: set[str] = set()
    merged: list[ArticleDiagram] = []

    def take(rows: list[ArticleDiagram]) -> None:
        for row in rows:
            rid = str(row.id or "")
            if rid and rid in seen_id:
                continue
            if rid:
                seen_id.add(rid)
            merged.append(row)

    take(list_diagrams_for_article(canonical_article_id("", url)))
    tag = hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]
    suffix = f":{tag}"
    with db_session() as session:
        stmt = (
            select(ArticleDiagram)
            .where(ArticleDiagram.article_id.endswith(suffix))
            .order_by(ArticleDiagram.created_at)
        )
        take(list(session.scalars(stmt).all()))
    return merged
