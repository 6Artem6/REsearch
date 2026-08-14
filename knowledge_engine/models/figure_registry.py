"""Реестр фигур до/после VLM (anchor mapping + обогащение)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_engine.db.base import Base


class FigureRegistryRow(Base):
    __tablename__ = "figure_registry"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    article_id: Mapped[str] = mapped_column(String(128), index=True)
    internal_fig_id: Mapped[str] = mapped_column(String(32), index=True)
    labels_json: Mapped[str] = mapped_column(Text, default="[]")
    caption: Mapped[str] = mapped_column(String(2000), default="")
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    anchor_p_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    extract_source: Mapped[str] = mapped_column(String(64), default="")
    vlm_summary: Mapped[str] = mapped_column(Text, default="")
    mermaid_code: Mapped[str] = mapped_column(Text, default="")
    image_phash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
