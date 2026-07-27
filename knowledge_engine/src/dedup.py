"""Knowledge Engine v0.7 — LanceDB dedup (cosine) and density-delta search termination."""

from __future__ import annotations

import json
import uuid
from typing import List, Optional

import lancedb
import numpy as np
from langchain_ollama import OllamaEmbeddings

from knowledge_engine.config import EMBED_MODEL, LANCE_DB_PATH, OLLAMA_BASE_URL
from knowledge_engine.src.locks import run_under_uma_lock
from knowledge_engine.src.state import StructuredChunk

V07_CHUNKS_TABLE = "v07_chunks"
COSINE_DEDUP_THRESHOLD = 0.88
DENSITY_DELTA_MIN = 0.15
MIN_CHUNKS_FOR_DENSITY = 10
MAX_SEARCH_DEPTH = 2


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class ChunkDedupStore:
    """LanceDB chunk index; all embed/search/write under ``staged_uma_lock``."""

    def __init__(self, table_name: str = V07_CHUNKS_TABLE) -> None:
        LANCE_DB_PATH.mkdir(parents=True, exist_ok=True)
        self._table_name = table_name
        self._db = lancedb.connect(str(LANCE_DB_PATH))
        self._embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    def _embed_sync(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text[:8000])

    async def embed_text(self, text: str) -> List[float]:
        return await run_under_uma_lock(self._embed_sync, text)

    def _max_similarity_sync(self, vector: List[float]) -> float:
        if self._table_name not in self._db.table_names():
            return 0.0
        table = self._db.open_table(self._table_name)
        if table.count_rows() == 0:
            return 0.0
        try:
            hits = table.search(vector).limit(3).to_list()
        except Exception:
            return 0.0
        best = 0.0
        for row in hits:
            existing = row.get("vector")
            if existing is None:
                continue
            sim = _cosine_similarity(vector, existing)
            if sim > best:
                best = sim
        return best

    def _ingest_sync(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        vector: List[float],
        meta: dict,
    ) -> None:
        row = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text[:12_000],
            "vector": vector,
            "meta_json": json.dumps(meta, ensure_ascii=False),
        }
        if self._table_name not in self._db.table_names():
            self._db.create_table(self._table_name, data=[row])
        else:
            self._db.open_table(self._table_name).add([row])

    async def is_near_duplicate(self, text: str) -> bool:
        """True if cosine similarity > threshold vs any stored chunk."""
        vector = await self.embed_text(text)
        max_sim = await run_under_uma_lock(self._max_similarity_sync, vector)
        return max_sim > COSINE_DEDUP_THRESHOLD

    async def ingest_chunk_text(
        self,
        doc_id: str,
        text: str,
        meta: Optional[dict] = None,
    ) -> Optional[StructuredChunk]:
        """
        Ingest if not duplicate. Returns chunk or None if discarded.
        Embedding + LanceDB write happen under UMA lock.
        """
        clean = text.strip()
        if len(clean) < 40:
            return None
        if await self.is_near_duplicate(clean):
            return None
        vector = await self.embed_text(clean)
        chunk_id = uuid.uuid4().hex[:16]
        meta_payload = meta or {}
        await run_under_uma_lock(
            self._ingest_sync,
            chunk_id,
            doc_id,
            clean,
            vector,
            meta_payload,
        )
        return StructuredChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=clean,
            concepts=list(meta_payload.get("concepts") or []),
            code_snippets=list(meta_payload.get("code_snippets") or []),
            p99_relevance_score=float(meta_payload.get("p99_relevance_score") or 0.0),
        )


def compute_density_delta(unique_ingested: int, total_scraped: int) -> float:
    if total_scraped <= 0:
        return 0.0
    return unique_ingested / total_scraped


def should_terminate_search(
    density_delta: float,
    search_depth: int,
    total_scraped_chunks: int,
    *,
    min_chunks: int = MIN_CHUNKS_FOR_DENSITY,
    delta_min: float = DENSITY_DELTA_MIN,
    max_depth: int = MAX_SEARCH_DEPTH,
) -> bool:
    """
    Terminate discovery when density is low (after min chunks) or depth cap hit.
    Cosine threshold is ingestion-only, not search stop signal.
    """
    if search_depth >= max_depth:
        return True
    if total_scraped_chunks >= min_chunks and density_delta < delta_min:
        return True
    return False


async def ingest_document_chunks(
    store: ChunkDedupStore,
    doc_id: str,
    raw_markdown: str,
    chunk_size: int = 1200,
) -> tuple[List[StructuredChunk], int]:
    """
    Split markdown into chunks, dedup-ingest each.
    Returns (accepted chunks, scraped_chunk_count).
    """
    paragraphs = [p.strip() for p in raw_markdown.split("\n\n") if p.strip()]
    scraped = 0
    accepted: List[StructuredChunk] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
            continue
        if buffer:
            scraped += 1
            chunk = await store.ingest_chunk_text(doc_id, buffer)
            if chunk is not None:
                accepted.append(chunk)
            buffer = para
    if buffer:
        scraped += 1
        chunk = await store.ingest_chunk_text(doc_id, buffer)
        if chunk is not None:
            accepted.append(chunk)
    return accepted, scraped
