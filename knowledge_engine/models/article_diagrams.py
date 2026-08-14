"""Диаграммы из статей (Mermaid + summary)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_engine.db.base import Base


class ArticleDiagram(Base):
    __tablename__ = "article_diagrams"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    article_id: Mapped[str] = mapped_column(String(128), index=True)
    image_phash: Mapped[str] = mapped_column(String(64), index=True)
    caption: Mapped[str] = mapped_column(String(2000), default="")
    mermaid_code: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
