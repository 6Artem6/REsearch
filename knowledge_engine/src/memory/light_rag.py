"""Light RAG — динамический профиль и факты (LanceDB). Без hardcode в коде."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import List

import lancedb
import numpy as np

from knowledge_engine.config import (
    LANCE_DB_PATH,
    LIGHT_RAG_MIN_COSINE_SIM,
    LIGHT_RAG_PROFILE_LIMIT,
    USER_PROFILE_PATH,
)
from knowledge_engine.db.embed_model_guard import (
    drop_if_embed_space_mismatch,
    row_matches_embed_model,
    stamp_embed_model,
)
from knowledge_engine.services.search.bge_m3_embed import BgeM3Embeddings
from knowledge_engine.src.locks import run_under_uma_lock
from knowledge_engine.src.processors.source_anchors import strip_source_anchor_tags_list
from knowledge_engine.ui.run_log import trace

LIGHT_RAG_TABLE = "light_rag_facts"
PROFILE_DOC_PREFIX = "profile_segment"
FACT_DOC_ID = "fact_nugget"
_PROFILE_SYNC_HASH_FILE = LANCE_DB_PATH / ".light_rag_profile_sha256"
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
        self._embeddings = BgeM3Embeddings()

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
        drop_if_embed_space_mismatch(self._db, self._table_name)
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
                stamp_embed_model(
                    {
                        "chunk_id": f"{PROFILE_DOC_PREFIX}_{i}",
                        "doc_id": f"{PROFILE_DOC_PREFIX}_{i}",
                        "text": seg[:12_000],
                        "vector": vector,
                        "meta_json": json.dumps(
                            {"kind": "profile"}, ensure_ascii=False
                        ),
                    }
                )
            )
        await run_under_uma_lock(self._ingest_profile_rows_sync, rows)
        trace(f"Light RAG ✓ profile segments indexed | n={len(rows)}")
        return len(rows)

    def _ingest_rows_sync(self, rows: list[dict]) -> None:
        if not rows:
            return
        drop_if_embed_space_mismatch(self._db, self._table_name)
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
                stamp_embed_model(
                    {
                        "chunk_id": uuid.uuid4().hex[:16],
                        "doc_id": FACT_DOC_ID,
                        "text": clean[:12_000],
                        "vector": vector,
                        "meta_json": json.dumps({"kind": "fact"}, ensure_ascii=False),
                    }
                )
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
            if not row_matches_embed_model(row):
                continue
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

    def _search_rows_sync(
        self,
        vector: List[float],
        limit: int,
        *,
        kinds: frozenset[str],
        min_cosine: float | None = None,
    ) -> List[tuple[float, str, dict]]:
        """Семантический поиск: (cosine_sim, text, meta)."""
        floor = min_cosine if min_cosine is not None else LIGHT_RAG_MIN_COSINE_SIM
        if self._table_name not in self._db.table_names():
            return []
        table = self._db.open_table(self._table_name)
        if table.count_rows() == 0:
            return []
        try:
            hits = table.search(vector).limit(max(limit * 4, 12)).to_list()
        except Exception:
            return []
        scored: List[tuple[float, str, dict]] = []
        for row in hits:
            if not row_matches_embed_model(row):
                continue
            doc_id = str(row.get("doc_id") or "")
            meta_raw = row.get("meta_json") or "{}"
            try:
                meta = json.loads(meta_raw)
                if not isinstance(meta, dict):
                    meta = {}
            except Exception:
                meta = {}
            kind = meta.get("kind", "")
            if not kind:
                kind = "profile" if doc_id.startswith(PROFILE_DOC_PREFIX) else "fact"
            if kind not in kinds:
                continue
            text = str(row.get("text") or "").strip()
            existing = row.get("vector")
            if not text or existing is None:
                continue
            sim = _cosine_similarity(vector, existing)
            if sim >= floor:
                scored.append((sim, text, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[tuple[float, str, dict]] = []
        seen: set[str] = set()
        for sim, text, meta in scored:
            if text in seen:
                continue
            seen.add(text)
            out.append((sim, text, meta))
            if len(out) >= limit:
                break
        return out

    async def vector_search(
        self,
        query: str,
        limit: int = 5,
        *,
        kinds: frozenset[str] | None = None,
        min_cosine: float | None = None,
    ) -> List[tuple[float, str, dict]]:
        """Направленный поиск для RAG Gateway (Модуль 3)."""
        q = (query or "").strip()
        if not q:
            return []
        use_kinds = kinds or frozenset({"profile", "fact"})
        vector = await self._embed(q)
        return await run_under_uma_lock(
            self._search_rows_sync,
            vector,
            limit,
            kinds=use_kinds,
            min_cosine=min_cosine,
        )

    async def save_user_fact(
        self,
        fact_text: str,
        category: str,
        node_id: str,
    ) -> int:
        """Индексация личного факта/пробела для будущих сессий (Модуль 3 write)."""
        clean = (fact_text or "").strip()
        if len(clean) < 12:
            return 0
        meta = {
            "kind": "fact",
            "category": (category or "").strip()[:200],
            "node_id": (node_id or "").strip()[:80],
        }
        vector = await self._embed(clean)
        row = stamp_embed_model(
            {
                "chunk_id": uuid.uuid4().hex[:16],
                "doc_id": FACT_DOC_ID,
                "text": clean[:12_000],
                "vector": vector,
                "meta_json": json.dumps(meta, ensure_ascii=False),
            }
        )
        await run_under_uma_lock(self._ingest_rows_sync, [row])
        trace(
            f"Light RAG ✓ save_user_fact | node={meta['node_id']} "
            f"category={meta['category']}"
        )
        return 1

    def count_profile_segments_sync(self) -> int:
        if self._table_name not in self._db.table_names():
            return 0
        try:
            table = self._db.open_table(self._table_name)
            doc_ids = table.to_arrow().column("doc_id").to_pylist()
            return sum(1 for d in doc_ids if str(d).startswith(PROFILE_DOC_PREFIX))
        except Exception:
            return 0

    async def count_profile_segments(self) -> int:
        return await run_under_uma_lock(self.count_profile_segments_sync)

    def count_indexed_rows_sync(self) -> int:
        if self._table_name not in self._db.table_names():
            return 0
        try:
            return int(self._db.open_table(self._table_name).count_rows())
        except Exception:
            return 0

    async def count_indexed_rows(self) -> int:
        return await run_under_uma_lock(self.count_indexed_rows_sync)


def count_light_rag_rows_sync(table_name: str = LIGHT_RAG_TABLE) -> int:
    """Row count without constructing the bi-encoder (API memory-status)."""
    if not LANCE_DB_PATH.is_dir():
        return 0
    try:
        db = lancedb.connect(str(LANCE_DB_PATH))
        if table_name not in db.table_names():
            return 0
        return int(db.open_table(table_name).count_rows())
    except Exception:
        return 0


def _user_profile_content_hash() -> str:
    path = USER_PROFILE_PATH
    if not path.is_file():
        return ""
    raw = path.read_bytes()
    if len(raw.strip()) < 40:
        return ""
    return hashlib.sha256(raw).hexdigest()


def _read_stored_profile_hash() -> str:
    try:
        if _PROFILE_SYNC_HASH_FILE.is_file():
            return _PROFILE_SYNC_HASH_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


async def sync_profile_from_markdown_if_needed() -> bool:
    """
    Индексирует user_profile.md в LanceDB, если файл новый/изменён
    или сегменты профиля ещё не в таблице.
    """
    current_hash = _user_profile_content_hash()
    if not current_hash:
        trace("[PERSONAL_RAG] user_profile.md missing or too short — skip profile sync")
        return False

    rag = LightRAG()
    stored_hash = _read_stored_profile_hash()
    profile_segments = await rag.count_profile_segments()
    total_rows = await rag.count_indexed_rows()

    need_sync = stored_hash != current_hash or profile_segments == 0
    if not need_sync:
        trace(
            f"[PERSONAL_RAG] profile index up-to-date | "
            f"segments={profile_segments} total_rows={total_rows}"
        )
        return False

    reason = "hash_changed" if stored_hash != current_hash else "no_profile_segments"
    trace(
        f"[PERSONAL_RAG] profile sync needed | reason={reason} "
        f"segments={profile_segments} total_rows={total_rows}"
    )
    n_seg = await rag.sync_profile_from_markdown("")
    if n_seg <= 0:
        trace(
            "[PERSONAL_RAG] profile sync produced no segments — "
            "hash not updated (check ## sections and min chunk length)"
        )
        return False
    try:
        LANCE_DB_PATH.mkdir(parents=True, exist_ok=True)
        _PROFILE_SYNC_HASH_FILE.write_text(current_hash, encoding="utf-8")
    except Exception as exc:
        trace(f"[PERSONAL_RAG] profile hash write skip | {exc}")
    trace(f"[PERSONAL_RAG] Syncing profile facts: OK | segments={n_seg}")
    return True
