"""Light RAG — динамический профиль и факты (LanceDB). Без hardcode в коде."""

from __future__ import annotations

import json
import re
import uuid
from typing import List

import lancedb
import numpy as np
from langchain_ollama import OllamaEmbeddings

from knowledge_engine.config import (
    EMBED_MODEL,
    LANCE_DB_PATH,
    LIGHT_RAG_MIN_COSINE_SIM,
    LIGHT_RAG_PROFILE_LIMIT,
    OLLAMA_BASE_URL,
    USER_PROFILE_PATH,
)
from knowledge_engine.src.locks import run_under_uma_lock
from knowledge_engine.src.processors.source_anchors import strip_source_anchor_tags_list
from knowledge_engine.ui.run_log import trace

LIGHT_RAG_TABLE = "light_rag_facts"
PROFILE_DOC_PREFIX = "profile_segment"
FACT_DOC_ID = "fact_nugget"
_PROFILE_SECTION_RE = re.compile(r"^##\s+", re.MULTILINE)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _split_profile_markdown(md: str) -> List[str]:
    text = (md or "").strip()
    if not text:
        return []
    parts = _PROFILE_SECTION_RE.split(text)
    headers = _PROFILE_SECTION_RE.findall(text)
    blocks: List[str] = []
    if not headers:
        return [text] if len(text) > 30 else []
    preamble = parts[0].strip()
    if len(preamble) > 30:
        blocks.append(preamble)
    for i, body in enumerate(parts[1:], start=0):
        title = headers[i].strip() if i < len(headers) else ""
        chunk = f"## {title}\n{body.strip()}".strip()
        if len(chunk) > 30:
            blocks.append(chunk)
    return blocks


class LightRAG:
    def __init__(self, table_name: str = LIGHT_RAG_TABLE) -> None:
        LANCE_DB_PATH.mkdir(parents=True, exist_ok=True)
        self._table_name = table_name
        self._db = lancedb.connect(str(LANCE_DB_PATH))
        self._embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    def _embed_sync(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text[:8000])

    async def _embed(self, text: str) -> List[float]:
        return await run_under_uma_lock(self._embed_sync, text)

    def _delete_profile_segments_sync(self) -> None:
        if self._table_name not in self._db.table_names():
            return
        table = self._db.open_table(self._table_name)
        try:
            table.delete(f"doc_id LIKE '{PROFILE_DOC_PREFIX}%'")
        except Exception:
            pass

    def _ingest_profile_rows_sync(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._delete_profile_segments_sync()
        if self._table_name not in self._db.table_names():
            self._db.create_table(self._table_name, data=rows)
        else:
            self._db.open_table(self._table_name).add(rows)

    async def sync_profile_from_markdown(self, profile_md: str) -> int:
        """
        Индексирует user_profile.md в векторную базу (сегменты ##).
        Не возвращает текст в промпты — только для get_relevant_profile_context.
        """
        source = (profile_md or "").strip()
        if not source and USER_PROFILE_PATH.is_file():
            source = USER_PROFILE_PATH.read_text(encoding="utf-8").strip()
        segments = _split_profile_markdown(source)
        if not segments:
            return 0
        rows: list[dict] = []
        for i, seg in enumerate(segments):
            vector = await self._embed(seg)
            rows.append(
                {
                    "chunk_id": f"{PROFILE_DOC_PREFIX}_{i}",
                    "doc_id": f"{PROFILE_DOC_PREFIX}_{i}",
                    "text": seg[:12_000],
                    "vector": vector,
                    "meta_json": json.dumps({"kind": "profile"}, ensure_ascii=False),
                }
            )
        await run_under_uma_lock(self._ingest_profile_rows_sync, rows)
        trace(f"Light RAG ✓ profile segments indexed | n={len(rows)}")
        return len(rows)

    def _ingest_rows_sync(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self._table_name not in self._db.table_names():
            self._db.create_table(self._table_name, data=rows)
        else:
            self._db.open_table(self._table_name).add(rows)

    async def ingest_facts(self, facts: list[str]) -> int:
        facts = strip_source_anchor_tags_list(facts)
        accepted = 0
        batch: list[dict] = []
        for fact in facts:
            clean = (fact or "").strip()
            if len(clean) < 12:
                continue
            vector = await self._embed(clean)
            batch.append(
                {
                    "chunk_id": uuid.uuid4().hex[:16],
                    "doc_id": FACT_DOC_ID,
                    "text": clean[:12_000],
                    "vector": vector,
                    "meta_json": json.dumps({"kind": "fact"}, ensure_ascii=False),
                }
            )
            accepted += 1
        if batch:
            await run_under_uma_lock(self._ingest_rows_sync, batch)
            trace(f"Light RAG ✓ ingest_facts | n={accepted}")
        return accepted

    def _relevant_texts_sync(
        self,
        vector: List[float],
        limit: int,
        *,
        kinds: frozenset[str],
    ) -> List[tuple[float, str]]:
        if self._table_name not in self._db.table_names():
            return []
        table = self._db.open_table(self._table_name)
        if table.count_rows() == 0:
            return []
        try:
            hits = table.search(vector).limit(max(limit * 4, 12)).to_list()
        except Exception:
            return []
        scored: List[tuple[float, str]] = []
        for row in hits:
            doc_id = str(row.get("doc_id") or "")
            meta_raw = row.get("meta_json") or "{}"
            try:
                kind = json.loads(meta_raw).get("kind", "")
            except Exception:
                kind = "profile" if doc_id.startswith(PROFILE_DOC_PREFIX) else "fact"
            if kind not in kinds:
                continue
            text = str(row.get("text") or "").strip()
            existing = row.get("vector")
            if not text or existing is None:
                continue
            sim = _cosine_similarity(vector, existing)
            if sim >= LIGHT_RAG_MIN_COSINE_SIM:
                scored.append((sim, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[tuple[float, str]] = []
        seen: set[str] = set()
        for sim, text in scored:
            if text in seen:
                continue
            seen.add(text)
            out.append((sim, text))
            if len(out) >= limit:
                break
        return out

    async def get_relevant_profile_context(self, query: str) -> str:
        """
        Селективный контекст для Gemini Lite/Reasoner (не для Consensus).
        Пустая строка, если нет релевантных сегментов профиля/фактов.
        """
        q = (query or "").strip()
        if not q:
            return ""
        vector = await self._embed(q)
        hits = await run_under_uma_lock(
            self._relevant_texts_sync,
            vector,
            LIGHT_RAG_PROFILE_LIMIT,
            kinds=frozenset({"profile", "fact"}),
        )
        if not hits:
            trace("Light RAG ⊘ no relevant profile context for query")
            return ""
        lines = [text for _, text in hits]
        trace(f"Light RAG ✓ profile context | segments={len(lines)}")
        return "\n\n---\n\n".join(lines)

    async def get_relevant_facts_context(self, query: str, limit: int = 5) -> str:
        q = (query or "").strip()
        if not q:
            return ""
        vector = await self._embed(q)
        hits = await run_under_uma_lock(
            self._relevant_texts_sync,
            vector,
            limit,
            kinds=frozenset({"fact"}),
        )
        if not hits:
            return ""
        return "\n".join(f"- {t}" for _, t in hits)
