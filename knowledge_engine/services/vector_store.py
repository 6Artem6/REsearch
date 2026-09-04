"""LanceDB: DocumentSummary (legacy) + KnowledgeNode graph (v0.3) + rag_chunks."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Callable, List, Optional, TypeVar

import lancedb
import numpy as np

import knowledge_engine.config as _config
from knowledge_engine.config import EMBED_MODEL, LANCE_DB_PATH
from knowledge_engine.db.embed_model_guard import (
    drop_if_embed_space_mismatch,
    row_matches_embed_model,
    stamp_embed_model,
)
from knowledge_engine.db.knowledge_atoms_schema import (
    COL_CONTEXT_QUOTE as KA_COL_CONTEXT_QUOTE,
)
from knowledge_engine.db.knowledge_atoms_schema import COL_DOC_ID as KA_COL_DOC_ID
from knowledge_engine.db.knowledge_atoms_schema import COL_ID as KA_COL_ID
from knowledge_engine.db.knowledge_atoms_schema import COL_SCOPE as KA_COL_SCOPE
from knowledge_engine.db.knowledge_atoms_schema import (
    COL_SOURCE_CHUNK_IDS as KA_COL_SOURCE_CHUNK_IDS,
)
from knowledge_engine.db.knowledge_atoms_schema import COL_STATEMENT as KA_COL_STATEMENT
from knowledge_engine.db.knowledge_atoms_schema import COL_URL as KA_COL_URL
from knowledge_engine.db.knowledge_atoms_schema import COL_VECTOR as KA_COL_VECTOR
from knowledge_engine.db.knowledge_atoms_schema import (
    KNOWLEDGE_ATOMS_TABLE,
)
from knowledge_engine.db.rag_chunks_schema import (
    COL_CHUNK_ID,
    COL_CHUNK_INDEX,
    COL_CHUNK_TEXT,
    COL_CHUNK_VECTOR,
    COL_CHUNKS_IN_DOC,
    COL_DOC_ID,
    COL_DOC_META_VECTOR,
    COL_DOC_SUMMARY_TEXT,
    COL_TITLE,
    COL_TRUST_SCORE,
    COL_URL,
    COL_WINDOW_SUMMARY,
    RAG_CHUNKS_TABLE,
)
from knowledge_engine.db.repositories.postgres_vector_repository import (
    generate_vector_id,
)
from knowledge_engine.schemas import DocumentSummary, KnowledgeNode
from knowledge_engine.services.lance_db_maintenance import (
    is_lance_format_error,
    reset_lance_directory,
)
from knowledge_engine.services.search.bge_m3_embed import BgeM3Embeddings
from knowledge_engine.ui.run_log import trace

TABLE_NAME = "document_summaries"

_qdrant_store_singleton: Any | None = None
_postgres_store_adapter: Any | None = None
_postgres_store_loop_id: int | None = None


def _get_qdrant_backend() -> Any | None:
    """Lazy singleton; only constructed when QDRANT_URL is configured."""
    global _qdrant_store_singleton
    if not (_config.QDRANT_URL or "").strip():
        return None
    if _qdrant_store_singleton is None:
        from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore

        _qdrant_store_singleton = QdrantVectorStore()
    return _qdrant_store_singleton if _qdrant_store_singleton.enabled else None


async def _get_postgres_backend() -> Any | None:
    """Lazy singleton adapter, НО keyed по текущему event loop.

    ИСПРАВЛЕНО (Phase 3): комментарий здесь раньше утверждал, что проблемы
    per-job event loop нет — это оказалось НЕВЕРНО и найдено живым
    воспроизведением: воркер зовёт node_deep_dive job'ы через
    asyncio.run() (см. services/work_handlers.py) — КАЖДЫЙ job получает
    новый event loop. SQLAlchemy AsyncEngine/asyncpg-пул, созданный в loop A,
    ломается при использовании из loop B с
    `InterfaceError: cannot perform operation: another operation is in
    progress` на второй же job. Детектируем смену loop'а и пересоздаём
    движок — старый просто отбрасывается (его loop уже закрыт, корректно
    dispose() из НОВОГО loop'а невозможно; соединения умирают вместе со
    старым loop'ом на транспортном уровне)."""
    global _postgres_store_adapter, _postgres_store_loop_id
    import asyncio

    current_loop_id = id(asyncio.get_running_loop())
    if (
        _postgres_store_adapter is not None
        and _postgres_store_loop_id != current_loop_id
    ):
        _postgres_store_adapter = None

    if _postgres_store_adapter is None:
        from knowledge_engine.db.repositories.postgres_vector_repository import (
            PostgresVectorRepository,
        )
        from knowledge_engine.services.postgres_vector_store_adapter import (
            PostgresVectorStoreAdapter,
        )

        repo = await PostgresVectorRepository.create(_config.POSTGRES_DSN)
        _postgres_store_adapter = PostgresVectorStoreAdapter(repo)
        _postgres_store_loop_id = current_loop_id
    return _postgres_store_adapter


async def _get_active_vector_store() -> Any | None:
    """Единая точка выбора бэкенда (см. config.VECTOR_STORE_BACKEND, Phase 2).
    Заменяет прежний _get_qdrant_store() как единственную фабрику, которую
    зовут все ~9 read/write путей ниже — им не нужно знать, какой бэкенд
    активен, оба соответствуют одному интерфейсу (см. postgres_vector_store_adapter.py).
    """
    if _config.VECTOR_STORE_BACKEND == "qdrant":
        return _get_qdrant_backend()
    return await _get_postgres_backend()


NODES_TABLE = "knowledge_nodes"

T = TypeVar("T")


def _summary_document(summary: DocumentSummary) -> str:
    """FTS-текст паспорта: сначала executive_summary, затем сжатые takeaways."""
    parts = [
        summary.title,
        summary.url,
        (summary.executive_summary or "").strip(),
        " ".join(summary.cs_concepts),
        " ".join(summary.key_takeaways),
        " ".join(summary.failure_modes),
        " ".join(summary.diagram_descriptions),
    ]
    return "\n".join(p for p in parts if (p or "").strip())


class VectorStore:
    def __init__(self) -> None:
        LANCE_DB_PATH.mkdir(parents=True, exist_ok=True)
        self._embeddings = BgeM3Embeddings()
        self._db = lancedb.connect(str(LANCE_DB_PATH))
        self._verify_lance_readable()

    def _connect(self) -> None:
        self._db = lancedb.connect(str(LANCE_DB_PATH))

    def _verify_lance_readable(self) -> None:
        for name in self._db.table_names():
            try:
                self._db.open_table(name).count_rows()
            except Exception as exc:
                if is_lance_format_error(exc):
                    reset_lance_directory(f"verify table {name}: {exc}")
                    self._connect()
                    return
                raise

    def _with_lance_recovery(self, fn: Callable[..., T], *args, **kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_lance_format_error(exc):
                raise
            reset_lance_directory(str(exc))
            self._connect()
            return fn(*args, **kwargs)

    def _table(self):
        if TABLE_NAME not in self._db.table_names():
            raise RuntimeError("LanceDB table empty — сначала save_summary")
        return self._db.open_table(TABLE_NAME)

    async def save_summary(
        self,
        summary: DocumentSummary,
        *,
        skip_rag_ingest: bool = False,
    ) -> bool:
        """Returns True only if the document_summaries Qdrant write actually
        landed (store configured + upsert confirmed) — callers that report a
        per-item success status must check this instead of assuming success."""
        document = _summary_document(summary)
        trace(f"EMBED ▶ {EMBED_MODEL} | Qdrant save {summary.url[:60]}")
        from knowledge_engine.ui.logger import set_phase, set_status

        set_phase(f"embed {EMBED_MODEL}")
        set_status(f"[Qdrant] embed → save {summary.title[:50]}…")
        vector = self._embeddings.embed_query(document)
        row = stamp_embed_model(
            {
                "title": summary.title,
                "url": summary.url,
                "executive_summary": (summary.executive_summary or "").strip(),
                "cs_concepts": json.dumps(summary.cs_concepts, ensure_ascii=False),
                "key_takeaways": json.dumps(summary.key_takeaways, ensure_ascii=False),
                "failure_modes": json.dumps(summary.failure_modes, ensure_ascii=False),
                "diagram_descriptions": json.dumps(
                    summary.diagram_descriptions, ensure_ascii=False
                ),
                "document": document,
            }
        )
        store = await _get_active_vector_store()
        qdrant_ok = False
        if store is None:
            trace(f"QDRANT save skip ⚠ | QDRANT_URL not set | {summary.url[:60]}")
        else:
            await store.ensure_collection(TABLE_NAME, len(vector))
            qdrant_ok = await store.upsert_documents(TABLE_NAME, [row], [vector])
            if qdrant_ok:
                trace(f"QDRANT save ✓ | {summary.url[:60]}")
            else:
                trace(
                    f"QDRANT save ✗ | upsert failed, see PERF trace above | {summary.url[:60]}"
                )
        if skip_rag_ingest:
            return qdrant_ok
        from knowledge_engine.ingestion.ingest import ingest_document_summary

        await ingest_document_summary(summary, body_text=document, store=self)
        return qdrant_ok

    def _rag_chunks_table(self, *, create: bool = False):
        if RAG_CHUNKS_TABLE not in self._db.table_names():
            if not create:
                return None
            return None
        return self._db.open_table(RAG_CHUNKS_TABLE)

    async def _add_rag_chunk_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        doc_id: str | None = None,
    ) -> None:
        """doc_id больше не используется внутри (раньше — для delete-перед-upsert,
        см. комментарий ниже); оставлен в сигнатуре ради вызывающих call site'ов,
        не тронутых в этом Phase."""
        rows = [stamp_embed_model(r) for r in rows]
        store = await _get_active_vector_store()
        if store is None:
            trace(
                f"QDRANT rag_chunks save skip ⚠ | QDRANT_URL not set | rows={len(rows)}"
            )
            return

        vectors = [list(r.get(COL_CHUNK_VECTOR) or []) for r in rows]
        docs = [{k: v for k, v in r.items() if k != COL_CHUNK_VECTOR} for r in rows]
        if vectors and vectors[0]:
            await store.ensure_collection(RAG_CHUNKS_TABLE, len(vectors[0]))
        # Раньше здесь стоял delete_by_field(doc_id) ПЕРЕД upsert (неатомарная
        # пара — см. фикс в upsert_knowledge_atoms выше). chunk_id теперь
        # детерминирован (f"{doc_id}_chunk_{i}" / f"{doc_id}_map_{i}"),
        # upsert сам перезаписывает существующие строки. Компромисс: если
        # документ переключается с naive-чанкинга на MAP-чанкинг (или
        # наоборот) и новых чанков МЕНЬШЕ, чем было, старые лишние строки не
        # удаляются автоматически — та же orphan-очистка, сознательно вне
        # этого Phase (см. отчёт).
        ok = await store.upsert_documents(RAG_CHUNKS_TABLE, docs, vectors)
        if ok:
            trace(f"QDRANT rag_chunks save ✓ | rows={len(rows)}")

    @staticmethod
    def _resolve_chunk_trust_score(
        url: str,
        *,
        source_type: str | None = None,
    ) -> float:
        from knowledge_engine.src.services.openalex_evaluator import (
            resolve_source_trust_score,
        )

        try:
            score = resolve_source_trust_score(url, source_type=source_type)
            trace(f"RAG_CHUNKS trust_score={score:.3f} | {(url or '')[:70]}")
            return float(score)
        except Exception as exc:
            trace(f"RAG_CHUNKS trust_score fallback=1.0 | {exc}")
            return 1.0

    @staticmethod
    def doc_id_for_url(url: str) -> str:
        key = (url or "").strip().lower().rstrip("/")
        if not key:
            return hashlib.sha256(b"local").hexdigest()[:24]
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    async def upsert_rag_chunks_from_summary(
        self,
        summary: DocumentSummary,
        *,
        body_text: str | None = None,
    ) -> int:
        from knowledge_engine.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE
        from knowledge_engine.services.rag_chunk_splitter import split_sliding_window

        url = (summary.url or "").strip()
        title = (summary.title or url or "source")[:400]
        doc_id = self.doc_id_for_url(url or title)
        if any(
            "_map_" in str(row.get(COL_CHUNK_ID) or "")
            for row in self.fetch_rag_chunks_by_doc_id(doc_id)
        ):
            trace(f"RAG_CHUNKS skip naive ingest | MAP windows exist | {url[:70]}")
            return 0
        doc_summary_text = _summary_document(summary)[:8000]
        chunk_source = (body_text or doc_summary_text).strip()
        pieces = split_sliding_window(
            chunk_source,
            chunk_size=RAG_CHUNK_SIZE,
            overlap=RAG_CHUNK_OVERLAP,
        )
        if not pieces:
            return 0

        doc_meta_vector = self._embeddings.embed_query(doc_summary_text[:8000])
        chunk_vectors = [self._embeddings.embed_query(p[:8000]) for p in pieces]
        trust = self._resolve_chunk_trust_score(url)
        n_chunks = len(pieces)
        rows: list[dict[str, Any]] = []
        for i, (text, vec) in enumerate(zip(pieces, chunk_vectors), 1):
            rows.append(
                {
                    COL_CHUNK_ID: f"{doc_id}_chunk_{i}",
                    COL_DOC_ID: doc_id,
                    COL_URL: url,
                    COL_TITLE: title,
                    COL_CHUNK_TEXT: text,
                    COL_CHUNK_VECTOR: vec,
                    COL_DOC_SUMMARY_TEXT: doc_summary_text,
                    COL_DOC_META_VECTOR: doc_meta_vector,
                    COL_CHUNK_INDEX: i,
                    COL_CHUNKS_IN_DOC: n_chunks,
                    COL_TRUST_SCORE: trust,
                }
            )

        await self._add_rag_chunk_rows(rows, doc_id=doc_id)
        return len(rows)

    async def upsert_rag_academic_map_windows(
        self,
        url: str,
        title: str,
        map_window_texts: list[str],
        summary: DocumentSummary,
        *,
        window_summaries: list[str | None] | None = None,
    ) -> int:
        """Persist each MAP window body as rag_chunks row (+ shared doc meta from REDUCE)."""
        url = (url or "").strip()
        pieces = [t.strip() for t in map_window_texts if (t or "").strip()]
        if not url.startswith("http") or not pieces:
            return 0
        title = (title or url)[:400]
        doc_id = self.doc_id_for_url(url)
        doc_summary_text = _summary_document(summary)[:8000]
        doc_meta_vector = self._embeddings.embed_query(doc_summary_text[:8000])
        chunk_vectors = [self._embeddings.embed_query(p[:8000]) for p in pieces]
        trust = self._resolve_chunk_trust_score(url)
        n_chunks = len(pieces)
        summaries = list(window_summaries or [])
        rows: list[dict[str, Any]] = []
        for i, (text, vec) in enumerate(zip(pieces, chunk_vectors), 1):
            win_sum = ""
            if i - 1 < len(summaries) and summaries[i - 1] is not None:
                win_sum = str(summaries[i - 1] or "").strip()[:8000]
            rows.append(
                {
                    COL_CHUNK_ID: f"{doc_id}_map_{i}",
                    COL_DOC_ID: doc_id,
                    COL_URL: url,
                    COL_TITLE: title,
                    COL_CHUNK_TEXT: text[:8000],
                    COL_CHUNK_VECTOR: vec,
                    COL_DOC_SUMMARY_TEXT: doc_summary_text,
                    COL_DOC_META_VECTOR: doc_meta_vector,
                    COL_CHUNK_INDEX: i,
                    COL_CHUNKS_IN_DOC: n_chunks,
                    COL_TRUST_SCORE: trust,
                    COL_WINDOW_SUMMARY: win_sum,
                }
            )
        await self._add_rag_chunk_rows(rows, doc_id=doc_id)
        return len(rows)

    async def upsert_knowledge_atoms(
        self,
        url: str,
        atoms: list[Any],
        *,
        doc_id: str | None = None,
    ) -> int:
        """Replace knowledge_atoms rows for a document (statement vectors for fact RAG)."""
        from knowledge_engine.schemas.extraction import KnowledgeAtom

        url = (url or "").strip()
        did = (doc_id or "").strip() or (self.doc_id_for_url(url) if url else "")
        if not did:
            return 0
        normalized: list[KnowledgeAtom] = []
        for item in atoms or []:
            if isinstance(item, KnowledgeAtom):
                normalized.append(item)
            elif isinstance(item, dict):
                try:
                    normalized.append(KnowledgeAtom.model_validate(item))
                except Exception:
                    continue
        store = await _get_active_vector_store()
        if store is None:
            trace(
                f"QDRANT knowledge_atoms save skip ⚠ | QDRANT_URL not set | doc_id={did[:12]}…"
            )
            return 0

        if not normalized:
            # Still clear stale atoms for this doc when REDUCE produced none.
            await store.delete_by_field(KNOWLEDGE_ATOMS_TABLE, KA_COL_DOC_ID, did)
            return 0

        rows: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        for atom in normalized:
            stmt = (atom.statement or "").strip()
            if not stmt:
                continue
            vec = self._embeddings.embed_query(stmt[:8000])
            rows.append(
                stamp_embed_model(
                    {
                        # Детерминированный UUIDv5 от (doc_id, statement) — раньше
                        # тут был uuid4(), из-за чего повторный ingest того же
                        # документа плодил дубли атомов вместо upsert-перезаписи
                        # (см. аудит Phase 1, PostgresVectorRepository.generate_vector_id).
                        KA_COL_ID: str(generate_vector_id(did, "atom", stmt)),
                        KA_COL_DOC_ID: did,
                        KA_COL_URL: url,
                        KA_COL_STATEMENT: stmt[:2000],
                        KA_COL_SCOPE: atom.scope.value,
                        KA_COL_SOURCE_CHUNK_IDS: list(atom.source_chunk_ids or []),
                        KA_COL_CONTEXT_QUOTE: (atom.context_quote or "")[:800],
                        # Kept in payload (unlike rag_chunks/document_summaries):
                        # search_knowledge_atoms' MMR needs per-candidate vectors,
                        # not just similarity-to-query.
                        KA_COL_VECTOR: vec,
                    }
                )
            )
            vectors.append(vec)
        if not rows:
            return 0

        await store.ensure_collection(KNOWLEDGE_ATOMS_TABLE, len(vectors[0]))
        # Раньше здесь стоял delete_by_field ПЕРЕД upsert — два отдельных
        # сетевых вызова без транзакции: падение между ними оставляло doc_id
        # вообще без атомов (см. аудит Phase 1). id теперь детерминирован
        # (см. фикс выше), поэтому upsert сам корректно перезаписывает
        # существующие строки — delete больше не нужен для ЭТОГО случая.
        # Компромисс: если набор атомов для doc_id УМЕНЬШИЛСЯ между прогонами
        # (новый REDUCE даёт меньше statement'ов, чем раньше), старые лишние
        # строки не удаляются автоматически — orphan-очистка сознательно не
        # входит в этот Phase (см. отчёт), не как часть write-пути.
        ok = await store.upsert_documents(KNOWLEDGE_ATOMS_TABLE, rows, vectors)
        n = len(rows) if ok else 0
        if ok:
            trace(
                f"QDRANT knowledge_atoms upsert ✓ | doc_id={did[:12]}… "
                f"atoms={n} | {(url or '')[:55]}"
            )
        return n

    @staticmethod
    def _row_chunk_ids(row: dict[str, Any]) -> list[str]:
        raw = row.get(KA_COL_SOURCE_CHUNK_IDS)
        if raw is None:
            return []
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
            return [s]
        if isinstance(raw, (list, tuple, set)):
            return [str(x).strip() for x in raw if str(x).strip()]
        return []

    @staticmethod
    def _mmr_select_rows(
        rows: list[dict[str, Any]],
        qv: np.ndarray,
        *,
        limit: int,
        lambda_mult: float,
    ) -> list[dict[str, Any]]:
        """Maximal Marginal Relevance over scored knowledge_atoms rows."""
        if not rows or limit <= 0:
            return []
        lam = float(lambda_mult)
        if lam >= 0.999 or len(rows) <= 1:
            return rows[:limit]

        def _norm_vec(row: dict[str, Any]) -> np.ndarray | None:
            vec = row.get(KA_COL_VECTOR)
            if vec is None:
                return None
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
            n = float(np.linalg.norm(arr))
            if n <= 0:
                return None
            return arr / n

        candidates = list(rows)
        selected: list[dict[str, Any]] = []
        selected_vecs: list[np.ndarray] = []
        while candidates and len(selected) < limit:
            best_i = 0
            best_score = float("-inf")
            for i, row in enumerate(candidates):
                rel = float(row.get("_score") or 0.0)
                if not selected_vecs:
                    mmr = rel
                else:
                    cv = _norm_vec(row)
                    if cv is None:
                        mmr = lam * rel
                    else:
                        max_sim = max(float(np.dot(cv, sv)) for sv in selected_vecs)
                        mmr = lam * rel - (1.0 - lam) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_i = i
            chosen = candidates.pop(best_i)
            selected.append(chosen)
            cv = _norm_vec(chosen)
            if cv is not None:
                selected_vecs.append(cv)
        return selected

    @staticmethod
    def _stochastic_sample_rows(
        rows: list[dict[str, Any]],
        *,
        limit: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Sample ``limit`` rows from a larger pool with score-weighted weights."""
        if not rows or limit <= 0:
            return []
        if len(rows) <= limit:
            return list(rows)
        scores = np.asarray(
            [max(1e-6, float(r.get("_score") or 0.0)) for r in rows],
            dtype=np.float64,
        )
        # Softmax-ish: emphasize higher scores but keep mass on the long tail.
        logits = scores / max(float(np.mean(scores)), 1e-6)
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        weights = weights / float(np.sum(weights))
        idx = rng.choice(len(rows), size=limit, replace=False, p=weights)
        # Preserve relative relevance order among the sample.
        picked = [rows[int(i)] for i in sorted(idx, key=lambda j: -scores[int(j)])]
        return picked

    async def search_knowledge_atoms(
        self,
        query: str,
        *,
        limit: int = 8,
        allowed_doc_ids: list[str] | None = None,
        min_score: float = 0.0,
        exclude_ids: list[str] | None = None,
        exclude_chunk_ids: list[str] | None = None,
        lambda_mult: float = 1.0,
        query_noise: float = 0.0,
        stochastic_sample: bool = False,
        pool_mult: int = 3,
        rng_seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Vector search over ``knowledge_atoms.statement`` (no parent-chunk expand).

        Returns rows with ``_score`` (cosine similarity) plus schema fields.

        Deep-analysis diversity knobs:
        - ``exclude_ids`` / ``exclude_chunk_ids``: hard filters (DB where + post-filter)
        - ``lambda_mult`` < 1: MMR toward diversity
        - ``query_noise``: stochastic perturbation of the query embedding
        - ``stochastic_sample``: score-weighted sample from an over-fetched pool
        """
        q = (query or "").strip()
        if not q:
            return []
        store = await _get_active_vector_store()
        if store is None:
            return []
        allow = [d for d in (allowed_doc_ids or []) if (d or "").strip()]
        if allowed_doc_ids is not None and not allow:
            return []

        excl_ids = [str(x).strip() for x in (exclude_ids or []) if str(x).strip()]
        excl_chunks = {
            str(x).strip() for x in (exclude_chunk_ids or []) if str(x).strip()
        }

        qv = np.asarray(self._embeddings.embed_query(q[:8000]), dtype=np.float64)
        qn = float(np.linalg.norm(qv))
        if qn > 0:
            qv = qv / qn

        noise = max(0.0, float(query_noise))
        if noise > 0:
            rng = np.random.default_rng(rng_seed)
            qv = qv + noise * rng.normal(size=qv.shape)
            qn2 = float(np.linalg.norm(qv))
            if qn2 > 0:
                qv = qv / qn2

        want = max(1, int(limit))
        fetch_mult = max(1, int(pool_mult))
        # Over-fetch for exclude / MMR / stochastic refill.
        if excl_ids or excl_chunks or float(lambda_mult) < 0.999 or stochastic_sample:
            fetch_mult = max(fetch_mult, 4)
        fetch_n = max(want * fetch_mult, want + len(excl_ids) + 8)

        results = await store.search_similar(
            KNOWLEDGE_ATOMS_TABLE,
            qv.tolist(),
            limit=fetch_n,
            doc_id_filter=allow or None,
            exclude_ids_filter=excl_ids or None,
        )

        floor = max(0.0, float(min_score))
        excl_id_set = set(excl_ids)
        out: list[dict[str, Any]] = []
        for row in results:
            if not row_matches_embed_model(row):
                continue
            rid = str(row.get(KA_COL_ID) or "").strip()
            if rid and rid in excl_id_set:
                continue
            chunk_ids = self._row_chunk_ids(row)
            if excl_chunks and chunk_ids and any(c in excl_chunks for c in chunk_ids):
                continue
            # Also exclude by atom id used as a synthetic chunk key when chunks empty.
            if excl_chunks and rid and rid in excl_chunks:
                continue
            # Qdrant collections are COSINE-distance; `_score` IS cosine similarity.
            score = float(row.get("_score") or 0.0)
            if score < floor:
                continue
            did = str(row.get(KA_COL_DOC_ID) or "").strip()
            if allow and did not in set(allow):
                continue
            item = dict(row)
            item["_score"] = score
            out.append(item)
        out.sort(key=lambda r: float(r.get("_score") or 0.0), reverse=True)

        # Pool for MMR / stochastic: keep top-M above floor (never lower threshold).
        pool_n = max(
            want * 2, min(len(out), want * 5 if stochastic_sample else want * 3)
        )
        pool = out[:pool_n]
        selected = self._mmr_select_rows(
            pool,
            qv,
            limit=max(want * 2, want) if stochastic_sample else want,
            lambda_mult=float(lambda_mult),
        )
        if stochastic_sample and len(selected) > want:
            rng = np.random.default_rng(rng_seed)
            selected = self._stochastic_sample_rows(selected, limit=want, rng=rng)
        else:
            selected = selected[:want]
        return selected

    async def count_knowledge_atoms(self, doc_id: str) -> int:
        """Number of knowledge_atoms rows for ``doc_id`` (0 if Qdrant unset/empty)."""
        did = (doc_id or "").strip()
        store = await _get_active_vector_store()
        if not did or store is None:
            return 0
        return await store.count_by_field(KNOWLEDGE_ATOMS_TABLE, KA_COL_DOC_ID, did)

    async def knowledge_atom_doc_ids(self) -> set[str]:
        """Set of doc_ids that already have at least one knowledge atom."""
        store = await _get_active_vector_store()
        if store is None:
            return set()
        return await store.distinct_field_values(KNOWLEDGE_ATOMS_TABLE, KA_COL_DOC_ID)

    def list_rag_documents(self) -> list[dict[str, Any]]:
        """
        Aggregate ``rag_chunks`` by doc_id.

        Each item: doc_id, url, title, chunk_count, missing_window_summary_count.
        """
        table = self._rag_chunks_table(create=False)
        if table is None or table.count_rows() == 0:
            return []
        try:
            rows = table.to_arrow().to_pylist()
        except Exception:
            return []
        by_doc: dict[str, dict[str, Any]] = {}
        for row in rows:
            did = str(row.get(COL_DOC_ID) or "").strip()
            if not did:
                continue
            meta = by_doc.get(did)
            if meta is None:
                meta = {
                    "doc_id": did,
                    "url": str(row.get(COL_URL) or "").strip(),
                    "title": str(row.get(COL_TITLE) or "").strip(),
                    "chunk_count": 0,
                    "missing_window_summary_count": 0,
                }
                by_doc[did] = meta
            meta["chunk_count"] = int(meta["chunk_count"]) + 1
            if not meta.get("url"):
                meta["url"] = str(row.get(COL_URL) or "").strip()
            if not meta.get("title"):
                meta["title"] = str(row.get(COL_TITLE) or "").strip()
            ws = row.get(COL_WINDOW_SUMMARY)
            if ws is None or not str(ws).strip():
                meta["missing_window_summary_count"] = (
                    int(meta["missing_window_summary_count"]) + 1
                )
        return sorted(by_doc.values(), key=lambda m: str(m.get("doc_id") or ""))

    @staticmethod
    def passport_is_filled(summary: DocumentSummary | None) -> bool:
        """True, если паспорт содержит Reduce-прозу или (legacy) takeaways."""
        if summary is None:
            return False
        if (summary.executive_summary or "").strip():
            return True
        takes = [
            str(t).strip() for t in (summary.key_takeaways or []) if str(t).strip()
        ]
        if not takes:
            return False
        return sum(len(t) for t in takes) >= 40

    def fetch_latest_summary_for_url(self, url: str) -> DocumentSummary | None:
        """Latest document_summaries row for URL (last matching row wins)."""
        key = (url or "").strip().rstrip("/").lower()
        if not key.startswith("http") or TABLE_NAME not in self._db.table_names():
            return None
        try:
            rows = self._table().to_arrow().to_pylist()
        except Exception:
            return None
        found: DocumentSummary | None = None
        for row in rows:
            u = (row.get("url") or "").strip().rstrip("/").lower()
            if u == key:
                found = self._row_to_summary(row)
        return found

    async def update_rag_window_summaries(
        self,
        doc_id: str,
        summaries_by_chunk_id: dict[str, str],
    ) -> int:
        """
        Rewrite ``rag_chunks`` for ``doc_id``, filling ``window_summary`` by chunk_id.

        Preserves existing chunk_text / vectors (no re-embed).
        """
        did = (doc_id or "").strip()
        if not did or not summaries_by_chunk_id:
            return 0
        rows = self.fetch_rag_chunks_by_doc_id(did)
        if not rows:
            return 0
        updated = 0
        out: list[dict[str, Any]] = []
        for row in rows:
            r = dict(row)
            cid = str(r.get(COL_CHUNK_ID) or "").strip()
            if cid and cid in summaries_by_chunk_id:
                r[COL_WINDOW_SUMMARY] = str(summaries_by_chunk_id[cid] or "").strip()[
                    :8000
                ]
                updated += 1
            elif COL_WINDOW_SUMMARY not in r:
                r[COL_WINDOW_SUMMARY] = ""
            for key in (COL_CHUNK_VECTOR, COL_DOC_META_VECTOR):
                val = r.get(key)
                if val is not None and hasattr(val, "tolist"):
                    r[key] = list(val)
            out.append(r)
        if updated == 0:
            return 0
        await self._add_rag_chunk_rows(out, doc_id=did)
        trace(
            f"RAG_CHUNKS window_summary ✓ | doc_id={did[:12]}… "
            f"updated={updated}/{len(out)}"
        )
        return updated

    async def upsert_rag_exa_highlights_fallback(
        self,
        url: str,
        title: str,
        body_text: str,
        *,
        source_type: str | None = None,
        detail_instruction: str | None = None,
    ) -> int:
        from knowledge_engine.db.rag_chunks_schema import (
            COL_DETAIL_INSTRUCTION,
            COL_SOURCE_TYPE,
            EXA_HIGHLIGHTS_FALLBACK_DETAIL_INSTRUCTION,
            EXA_HIGHLIGHTS_FALLBACK_SOURCE_TYPE,
        )

        url = (url or "").strip()
        text = (body_text or "").strip()
        if not url.startswith("http") or len(text) < 40:
            return 0
        title = (title or url)[:400]
        doc_id = self.doc_id_for_url(url)
        st = (source_type or EXA_HIGHLIGHTS_FALLBACK_SOURCE_TYPE).strip()[:64]
        detail = (
            detail_instruction or EXA_HIGHLIGHTS_FALLBACK_DETAIL_INSTRUCTION
        ).strip()[:500]
        doc_summary_text = f"{title}\n{url}\n{detail}\n{text[:6000]}"
        doc_meta_vector = self._embeddings.embed_query(doc_summary_text[:8000])
        chunk_vector = self._embeddings.embed_query(text[:8000])
        trust = self._resolve_chunk_trust_score(url, source_type=st)
        row = {
            COL_CHUNK_ID: f"{doc_id}_chunk_exa_hl_1",
            COL_DOC_ID: doc_id,
            COL_URL: url,
            COL_TITLE: title,
            COL_CHUNK_TEXT: text[:8000],
            COL_CHUNK_VECTOR: chunk_vector,
            COL_DOC_SUMMARY_TEXT: doc_summary_text[:8000],
            COL_DOC_META_VECTOR: doc_meta_vector,
            COL_CHUNK_INDEX: 1,
            COL_CHUNKS_IN_DOC: 1,
            COL_SOURCE_TYPE: st,
            COL_DETAIL_INSTRUCTION: detail,
            COL_TRUST_SCORE: trust,
        }
        await self._add_rag_chunk_rows([row], doc_id=doc_id)
        return 1

    def delete_rag_chunks_for_urls(self, urls: list[str]) -> int:
        removed = 0
        for raw in urls:
            u = (raw or "").strip()
            if not u.startswith("http"):
                continue
            did = self.doc_id_for_url(u)
            table = self._rag_chunks_table(create=False)
            if table is None:
                continue
            try:
                before = len(
                    table.search()
                    .where(f"{COL_DOC_ID} = '{self._sql_literal(did)}'")
                    .limit(500)
                    .to_list()
                )
                if before:
                    table.delete(f"{COL_DOC_ID} = '{self._sql_literal(did)}'")
                    removed += before
            except Exception:
                continue
        return removed

    def _knowledge_atoms_table(self):
        if KNOWLEDGE_ATOMS_TABLE not in self._db.table_names():
            return None
        return self._db.open_table(KNOWLEDGE_ATOMS_TABLE)

    def delete_knowledge_atoms_for_urls(self, urls: list[str]) -> int:
        """Local LanceDB ``knowledge_atoms`` rows by doc_id, mirroring
        ``delete_rag_chunks_for_urls``. Live writes for this table moved to
        Qdrant-only (see ``upsert_knowledge_atoms``), but pre-migration rows
        were left behind in LanceDB with no cleanup path — clear_node_data /
        library GC previously deleted rag_chunks/summaries here but never
        touched this table, leaking stale atoms for cleared/orphaned nodes."""
        removed = 0
        for raw in urls:
            u = (raw or "").strip()
            if not u.startswith("http"):
                continue
            did = self.doc_id_for_url(u)
            table = self._knowledge_atoms_table()
            if table is None:
                continue
            try:
                before = len(
                    table.search()
                    .where(f"{KA_COL_DOC_ID} = '{self._sql_literal(did)}'")
                    .limit(500)
                    .to_list()
                )
                if before:
                    table.delete(f"{KA_COL_DOC_ID} = '{self._sql_literal(did)}'")
                    removed += before
            except Exception:
                continue
        return removed

    def delete_summaries_for_urls(self, urls: list[str]) -> int:
        want = {
            (u or "").strip().rstrip("/").lower()
            for u in urls
            if (u or "").startswith("http")
        }
        if not want or TABLE_NAME not in self._db.table_names():
            return 0
        table = self._table()
        removed = 0
        try:
            rows = table.to_arrow().to_pylist()
        except Exception:
            return 0
        for row in rows:
            url = (row.get("url") or "").strip()
            key = url.rstrip("/").lower()
            if key not in want:
                continue
            try:
                table.delete(f"url = '{self._sql_literal(url)}'")
                removed += 1
            except Exception:
                pass
        return removed

    @staticmethod
    def _sql_literal(value: str) -> str:
        return (value or "").replace("'", "''")

    @classmethod
    def _where_doc_ids_in(cls, doc_ids: list[str]) -> str:
        safe = [cls._sql_literal(d) for d in doc_ids if (d or "").strip()]
        if not safe:
            return ""
        if len(safe) == 1:
            return f"{COL_DOC_ID} = '{safe[0]}'"
        inner = ", ".join(f"'{d}'" for d in safe[:256])
        return f"{COL_DOC_ID} IN ({inner})"

    def count_rag_chunks_in_scope(self, allowed_doc_ids: list[str] | None) -> int:
        table = self._rag_chunks_table(create=False)
        if table is None:
            return 0
        allow = [d for d in (allowed_doc_ids or []) if (d or "").strip()]
        if not allow:
            try:
                return int(table.count_rows())
            except Exception:
                return 0
        try:
            where = self._where_doc_ids_in(allow)
            if not where:
                return 0
            return len(table.search().where(where).limit(10_000).to_list())
        except Exception:
            return 0

    async def search_rag_chunk_rows(
        self,
        query: str,
        *,
        limit: int = 64,
        doc_gate_threshold: float = 0.0,
        allowed_doc_ids: list[str] | None = None,
        prefilter: bool = True,
        relevance_penalty: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Vector search on fine chunks via Qdrant; doc_id allow-list applied server-side."""
        store = await _get_active_vector_store()
        if store is None:
            return []

        allow = [d for d in (allowed_doc_ids or []) if (d or "").strip()]
        if allowed_doc_ids is not None and not allow:
            return []
        _ = prefilter  # Qdrant filtered ANN is always applied server-side, pre-search

        qv = np.asarray(
            self._embeddings.embed_query((query or "")[:8000]), dtype=np.float64
        )
        qn = float(np.linalg.norm(qv))
        if qn > 0:
            qv = qv / qn

        results = await store.search_similar(
            RAG_CHUNKS_TABLE,
            qv.tolist(),
            limit=limit,
            doc_id_filter=allow or None,
        )
        results = [r for r in results if row_matches_embed_model(r)]

        # --- Early exit: score + hard cutoff BEFORE any CE/MMR / Lite context ---
        penalty = max(0.0, min(1.0, float(relevance_penalty)))
        from knowledge_engine.src.services.openalex_evaluator import (
            coerce_trust_score,
            final_retrieval_score,
            passes_trust_hard_cutoff,
        )

        scored: list[dict[str, Any]] = []
        for row in results:
            doc_vec = row.get(COL_DOC_META_VECTOR)
            # Qdrant collections are COSINE-distance; `_score` IS cosine similarity
            # (equivalent to the manual normalized-dot-product LanceDB used).
            cos_chunk = float(row.get("_score") or 0.0)
            if doc_gate_threshold > 0 and doc_vec is not None:
                dv = np.asarray(doc_vec, dtype=np.float64)
                dn = float(np.linalg.norm(dv))
                if dn > 0:
                    cos_doc = float(np.dot(qv, dv / dn))
                    if cos_doc < doc_gate_threshold:
                        continue
            if penalty < 1.0:
                cos_chunk *= penalty
            trust = coerce_trust_score(row.get(COL_TRUST_SCORE), default=1.0)
            item = dict(row)
            item["_cosine_raw"] = cos_chunk
            item["_trust_score"] = trust
            if doc_vec is not None:
                dv = np.asarray(doc_vec, dtype=np.float64)
                dn = float(np.linalg.norm(dv))
                item["_cosine_doc"] = float(np.dot(qv, dv / dn)) if dn > 0 else 0.0
            if penalty < 1.0:
                item["_scope_penalty"] = penalty
            scored.append(item)

        out: list[dict[str, Any]] = []
        dropped_hard = 0
        for item in scored:
            cos_chunk = float(item.get("_cosine_raw") or 0.0)
            trust = float(item.get("_trust_score") or 1.0)
            if not passes_trust_hard_cutoff(cos_chunk, trust):
                dropped_hard += 1
                continue
            item["_cosine_chunk"] = final_retrieval_score(cos_chunk, trust)
            out.append(item)
        out.sort(key=lambda r: float(r.get("_cosine_chunk") or 0.0), reverse=True)
        if dropped_hard:
            trace(
                f"RAG_CHUNKS hard_cutoff early_exit ⊘ | dropped={dropped_hard} "
                f"kept={len(out)} (before CE/MMR)"
            )
        return out

    def fetch_rag_chunks_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        """All fine chunks for a parent document, ordered by chunk_index."""
        did = (doc_id or "").strip()
        if not did:
            return []
        table = self._rag_chunks_table(create=False)
        if table is None or table.count_rows() == 0:
            return []
        try:
            rows = (
                table.search()
                .where(f"{COL_DOC_ID} = '{self._sql_literal(did)}'")
                .limit(500)
                .to_list()
            )
        except Exception:
            rows = []
            for row in table.to_arrow().to_pylist():
                if str(row.get(COL_DOC_ID) or "") == did:
                    rows.append(row)
        rows.sort(key=lambda r: int(r.get(COL_CHUNK_INDEX) or 0))
        return rows

    async def hybrid_search(self, query: str, limit: int = 3) -> List[DocumentSummary]:
        """Vector search via Qdrant (document_summaries). No keyword/FTS fusion —
        Qdrant search_similar is vector-only, unlike LanceDB's BM25+vector hybrid."""
        store = await _get_active_vector_store()
        if store is None:
            return []
        query_vector = self._embeddings.embed_query(query)
        results = await store.search_similar(TABLE_NAME, query_vector, limit=limit)
        return [self._row_to_summary(row) for row in results]

    def _row_to_summary(self, row: dict) -> DocumentSummary:
        return DocumentSummary(
            title=row.get("title") or "",
            url=row.get("url") or "",
            executive_summary=str(row.get("executive_summary") or "").strip(),
            cs_concepts=json.loads(row.get("cs_concepts") or "[]"),
            key_takeaways=json.loads(row.get("key_takeaways") or "[]"),
            failure_modes=json.loads(row.get("failure_modes") or "[]"),
            diagram_descriptions=json.loads(row.get("diagram_descriptions") or "[]"),
        )

    def hybrid_search_with_vectors(
        self, query: str, limit: int = 3
    ) -> list[tuple[DocumentSummary, list[float]]]:
        """Hybrid search with LanceDB document vectors (doc-level embedding)."""
        if TABLE_NAME not in self._db.table_names():
            return []

        table = self._table()
        if table.count_rows() == 0:
            return []

        query_vector = self._embeddings.embed_query(query)
        try:
            results = (
                table.search(query, query_type="hybrid")
                .vector(query_vector)
                .limit(limit)
                .to_list()
            )
        except Exception:
            results = table.search(query_vector).limit(limit).to_list()

        out: list[tuple[DocumentSummary, list[float]]] = []
        for row in results:
            if not row_matches_embed_model(row):
                continue
            vec = row.get("vector")
            if vec is None:
                continue
            out.append((self._row_to_summary(row), list(vec)))
        return out

    async def fetch_summaries_by_urls(
        self,
        urls: list[str],
        limit: int = 8,
    ) -> list[DocumentSummary]:
        """Конспекты для URL из маршрута / registry — точечный lookup через Qdrant."""
        store = await _get_active_vector_store()
        if store is None:
            return []
        summaries: list[DocumentSummary] = []
        seen: set[str] = set()
        for raw in urls:
            if len(summaries) >= limit:
                break
            u = (raw or "").strip()
            if not u.startswith("http") or u in seen:
                continue
            seen.add(u)
            payload = await store.fetch_by_url(TABLE_NAME, u)
            if payload is not None:
                summaries.append(self._row_to_summary(payload))
        return summaries

    def fetch_summaries_by_urls_with_vectors(
        self,
        urls: list[str],
        limit: int = 8,
    ) -> list[tuple[DocumentSummary, list[float]]]:
        want: set[str] = set()
        for raw in urls:
            u = (raw or "").strip()
            if u.startswith("http"):
                want.add(u.rstrip("/").lower())
        if not want or TABLE_NAME not in self._db.table_names():
            return []
        table = self._table()
        if table.count_rows() == 0:
            return []
        out: list[tuple[DocumentSummary, list[float]]] = []
        seen: set[str] = set()
        try:
            rows = table.to_arrow().to_pylist()
        except Exception:
            rows = []
        for row in rows:
            url = (row.get("url") or "").strip()
            key = url.rstrip("/").lower()
            if key not in want or key in seen:
                continue
            vec = row.get("vector")
            if vec is None:
                continue
            seen.add(key)
            out.append((self._row_to_summary(row), list(vec)))
            if len(out) >= limit:
                break
        return out

    def _nodes_table(self):
        if NODES_TABLE not in self._db.table_names():
            return None
        return self._db.open_table(NODES_TABLE)

    def save_knowledge_node(
        self,
        level: str,
        content: str,
        parent_id: Optional[str] = None,
        source_url: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or str(uuid.uuid4())
        document = "\n".join(filter(None, [level, content, source_url or ""]))
        trace(f"KNODE save {level} | {nid[:8]}…")
        vector = self._embeddings.embed_query(document)
        row = stamp_embed_model(
            {
                "id": nid,
                "level": level,
                "parent_id": parent_id or "",
                "content": content,
                "source_url": source_url or "",
                "document": document,
                "vector": vector,
            }
        )
        drop_if_embed_space_mismatch(self._db, NODES_TABLE)
        if NODES_TABLE not in self._db.table_names():
            self._db.create_table(NODES_TABLE, data=[row])
            try:
                self._db.open_table(NODES_TABLE).create_fts_index("document")
            except Exception:
                pass
        else:
            self._with_lance_recovery(lambda: self._nodes_table().add([row]))
        return nid

    def get_knowledge_node(self, node_id: str) -> Optional[KnowledgeNode]:
        table = self._nodes_table()
        if table is None or table.count_rows() == 0:
            return None
        try:
            rows = table.search().where(f"id = '{node_id}'").limit(1).to_list()
        except Exception:
            rows = []
            for row in table.to_arrow().to_pylist():
                if row.get("id") == node_id:
                    rows.append(row)
                    break
        if not rows:
            return None
        r = rows[0]
        return KnowledgeNode(
            id=r["id"],
            level=r.get("level") or "",
            parent_id=(r.get("parent_id") or None) or None,
            content=r.get("content") or "",
            source_url=(r.get("source_url") or None) or None,
        )

    def get_hierarchical_context(self, node_id: str) -> str:
        """L2 → L1 → L0 для передачи в stateless Gemini."""
        node = self.get_knowledge_node(node_id)
        if node is None:
            return ""

        lines: list[str] = []
        current: Optional[KnowledgeNode] = node
        chain: list[KnowledgeNode] = []
        while current is not None:
            chain.append(current)
            if not current.parent_id:
                break
            current = self.get_knowledge_node(current.parent_id)

        for n in reversed(chain):
            prefix = n.level
            url = f" ({n.source_url})" if n.source_url else ""
            lines.append(f"[{prefix}]{url}\n{n.content}")

        return "\n\n".join(lines)

    def hybrid_search_nodes(self, query: str, limit: int = 5) -> List[KnowledgeNode]:
        table = self._nodes_table()
        if table is None or table.count_rows() == 0:
            return []
        query_vector = self._embeddings.embed_query(query)
        try:
            results = (
                table.search(query, query_type="hybrid")
                .vector(query_vector)
                .limit(limit)
                .to_list()
            )
        except Exception:
            results = table.search(query_vector).limit(limit).to_list()

        nodes: list[KnowledgeNode] = []
        for row in results:
            nodes.append(
                KnowledgeNode(
                    id=row.get("id") or "",
                    level=row.get("level") or "",
                    parent_id=(row.get("parent_id") or None) or None,
                    content=row.get("content") or "",
                    source_url=(row.get("source_url") or None) or None,
                )
            )
        return nodes

    def hybrid_search_nodes_with_vectors(
        self, query: str, limit: int = 5
    ) -> list[tuple[KnowledgeNode, list[float]]]:
        table = self._nodes_table()
        if table is None or table.count_rows() == 0:
            return []
        query_vector = self._embeddings.embed_query(query)
        try:
            results = (
                table.search(query, query_type="hybrid")
                .vector(query_vector)
                .limit(limit)
                .to_list()
            )
        except Exception:
            results = table.search(query_vector).limit(limit).to_list()

        out: list[tuple[KnowledgeNode, list[float]]] = []
        for row in results:
            if not row_matches_embed_model(row):
                continue
            vec = row.get("vector")
            if vec is None:
                continue
            out.append(
                (
                    KnowledgeNode(
                        id=row.get("id") or "",
                        level=row.get("level") or "",
                        parent_id=(row.get("parent_id") or None) or None,
                        content=row.get("content") or "",
                        source_url=(row.get("source_url") or None) or None,
                    ),
                    list(vec),
                )
            )
        return out
