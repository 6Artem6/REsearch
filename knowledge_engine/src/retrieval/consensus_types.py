"""Результат одного сообщения в Consensus."""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper


class ConsensusMessageResult(BaseModel):
    raw_text: str = ""
    papers: list[ScholarPaper] = Field(default_factory=list)
