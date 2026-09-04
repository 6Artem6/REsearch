"""Semantic Intent Classification for tutor control chips via BGE-M3 embeddings.

Phrase catalog is ``intent_definitions.INTENT_REFERENCE_PHRASES`` (SSOT).
Vectors are persisted in LanceDB (``intent_vectors``) and loaded into a RAM
NumPy matrix so cold start does not re-embed the catalog on every process boot
when the phrase IDs are unchanged. Startup ``sync_and_validate_intents`` checks
registry integrity, then diffs LanceDB rows against the expected IDs.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from knowledge_engine.config import (
    EMBED_MODEL,
    LANCE_DB_PATH,
    VECTOR_INTENT_ENABLED,
    VECTOR_INTENT_THRESHOLD,
)
from knowledge_engine.db.intent_vectors_schema import (
    COL_EMBED_MODEL,
    COL_ID,
    COL_INTENT,
    COL_PHRASE,
    COL_VECTOR,
    INTENT_VECTORS_TABLE,
)
from knowledge_engine.src.node_deep_dive.intent_definitions import (
    INTENT_REFERENCE_PHRASES,
    validate_intent_catalog,
)
from knowledge_engine.ui.run_log import trace

EmbedFn = Callable[[str], list[float] | np.ndarray]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def phrase_vector_id(
    intent: str,
    phrase: str,
    *,
    embed_model: str = EMBED_MODEL,
) -> str:
    """Stable id: hash(embed_model + intent + phrase)."""
    key = f"{embed_model}\n{intent}\n{phrase}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def iter_reference_entries(
    phrases: dict[str, tuple[str, ...]] | None = None,
    *,
    embed_model: str = EMBED_MODEL,
) -> list[tuple[str, str, str]]:
    """Return ``[(id, intent, phrase), ...]`` in stable catalog order."""
    src = phrases or INTENT_REFERENCE_PHRASES
    out: list[tuple[str, str, str]] = []
    for intent, plist in src.items():
        for phrase in plist:
            p = (phrase or "").strip()
            if not p:
                continue
            out.append(
                (phrase_vector_id(intent, p, embed_model=embed_model), intent, p)
            )
    return out


class VectorIntentRouter:
    """Cosine-similarity intent router over LanceDB-backed reference embeddings."""

    def __init__(
        self,
        *,
        threshold: float | None = None,
        embed_fn: EmbedFn | None = None,
        reference_phrases: dict[str, tuple[str, ...]] | None = None,
        enabled: bool | None = None,
        persist: bool = True,
        db_path: Path | str | None = None,
        embed_model: str | None = None,
        auto_sync: bool = True,
        timeout_sec: float | None = None,
    ) -> None:
        self.threshold = float(
            VECTOR_INTENT_THRESHOLD if threshold is None else threshold
        )
        self.enabled = VECTOR_INTENT_ENABLED if enabled is None else bool(enabled)
        self._embed_fn = embed_fn
        self._phrases = reference_phrases or INTENT_REFERENCE_PHRASES
        self._persist = bool(persist)
        self._db_path = Path(db_path) if db_path is not None else LANCE_DB_PATH
        self._embed_model = (embed_model or EMBED_MODEL).strip() or EMBED_MODEL
        self._lock = threading.Lock()
        self._ready = False
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None  # (N, D) L2-normalized
        self._db = None
        self._degraded = False
        # Injected embed (tests) skips the timeout wrapper unless the caller
        # sets timeout_sec. Production BGE-M3 / LanceDB path defaults to 2s.
        if timeout_sec is not None:
            self.timeout_sec = float(timeout_sec)
        else:
            self.timeout_sec = 0.0 if embed_fn is not None else 2.0
        if auto_sync and self.enabled:
            try:
                self.sync_and_validate_intents()
            except Exception as exc:
                trace(
                    f"[VECTOR_ROUTER] startup sync deferred | "
                    f"{type(exc).__name__}: {exc}"
                )

    def _default_embed(self, text: str) -> list[float]:
        from knowledge_engine.services.search.bge_m3_embed import embed_query_bge_m3

        return embed_query_bge_m3(text)

    def _embed(self, text: str) -> np.ndarray:
        fn = self._embed_fn or self._default_embed
        vec = np.asarray(fn(text), dtype=np.float64).reshape(-1)
        return _l2_normalize(vec)

    def _connect_db(self):
        if self._db is not None:
            return self._db
        from knowledge_engine.db.lancedb_pool import get_lancedb_connection

        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = get_lancedb_connection(self._db_path)
        return self._db

    def _expected_entries(self) -> list[tuple[str, str, str]]:
        return iter_reference_entries(self._phrases, embed_model=self._embed_model)

    def _read_table_rows(self) -> list[dict[str, Any]]:
        db = self._connect_db()
        names = set(db.table_names())
        if INTENT_VECTORS_TABLE not in names:
            return []
        table = db.open_table(INTENT_VECTORS_TABLE)
        try:
            rows = table.to_arrow().to_pylist()
            out: list[dict[str, Any]] = []
            for row in rows:
                model = row.get(COL_EMBED_MODEL)
                if model not in (None, "", self._embed_model):
                    continue
                out.append(row)
            return out
        except Exception as exc:
            trace(
                f"[VECTOR_ROUTER] read {INTENT_VECTORS_TABLE} failed | "
                f"{type(exc).__name__}: {exc}"
            )
            return []

    def _replace_table(self, rows: list[dict[str, Any]]) -> None:
        db = self._connect_db()
        names = set(db.table_names())
        if INTENT_VECTORS_TABLE in names:
            db.drop_table(INTENT_VECTORS_TABLE)
        if not rows:
            return
        db.create_table(INTENT_VECTORS_TABLE, data=rows)

    def _load_matrix_from_rows(
        self,
        rows_by_id: dict[str, dict[str, Any]],
        expected: list[tuple[str, str, str]],
    ) -> None:
        labels: list[str] = []
        vecs: list[np.ndarray] = []
        for rid, intent, _phrase in expected:
            row = rows_by_id.get(rid)
            if row is None:
                continue
            raw_vec = row.get(COL_VECTOR)
            if raw_vec is None:
                continue
            v = np.asarray(raw_vec, dtype=np.float64).reshape(-1)
            labels.append(intent)
            vecs.append(_l2_normalize(v))
        if not vecs:
            self._labels = []
            self._matrix = np.zeros((0, 1), dtype=np.float64)
        else:
            self._labels = labels
            self._matrix = np.vstack(vecs)
        self._ready = True

    def sync_and_validate_intents(self) -> dict[str, Any]:
        """
        Compare SSOT ``INTENT_REFERENCE_PHRASES`` with LanceDB ``intent_vectors``.

        Re-embed only missing/stale phrases. When catalog is intact, perform
        **zero** embedding API calls for reference vectors and load RAM matrix from DB.
        """
        catalog = validate_intent_catalog()
        with self._lock:
            expected = self._expected_entries()
            expected_ids = {eid for eid, _, _ in expected}
            stats: dict[str, Any] = {
                "expected": len(expected),
                "embedded": 0,
                "removed": 0,
                "loaded_from_db": 0,
                "persist": self._persist,
                "embed_model": self._embed_model,
                "catalog_valid": bool(catalog.get("ok")),
                "catalog_intents": catalog.get("intents"),
                "catalog_phrases": catalog.get("phrases"),
            }

            if not self._persist:
                # In-memory only (unit tests): embed all references once.
                labels: list[str] = []
                rows: list[np.ndarray] = []
                for _eid, intent, phrase in expected:
                    labels.append(intent)
                    rows.append(self._embed(phrase))
                    stats["embedded"] += 1
                self._labels = labels
                self._matrix = (
                    np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float64)
                )
                self._ready = True
                trace(
                    f"[VECTOR_ROUTER] In-memory matrix ready | rows={len(labels)} "
                    f"embedded={stats['embedded']} (persist=false)"
                )
                return stats

            existing = self._read_table_rows()
            by_id = {str(r.get(COL_ID) or ""): r for r in existing if r.get(COL_ID)}
            existing_ids = set(by_id.keys())
            missing_ids = expected_ids - existing_ids
            stale_ids = existing_ids - expected_ids

            need_rebuild = (
                not existing
                or bool(missing_ids)
                or bool(stale_ids)
                or len(existing_ids) != len(expected_ids)
            )

            if not need_rebuild:
                # Integrity OK — no Ollama for references
                self._load_matrix_from_rows(by_id, expected)
                stats["loaded_from_db"] = len(self._labels)
                trace(
                    f"[VECTOR_ROUTER] Catalog intact | loaded={stats['loaded_from_db']} "
                    f"from LanceDB (0 Ollama reference embeds) "
                    f"model={self._embed_model}"
                )
                return stats

            # Keep valid rows; embed only missing
            new_rows: list[dict[str, Any]] = []
            for eid, intent, phrase in expected:
                if eid in by_id and eid not in missing_ids:
                    row = dict(by_id[eid])
                    row[COL_EMBED_MODEL] = self._embed_model
                    new_rows.append(row)
                    continue
                vec = self._embed(phrase)
                stats["embedded"] += 1
                new_rows.append(
                    {
                        COL_ID: eid,
                        COL_INTENT: intent,
                        COL_PHRASE: phrase,
                        COL_VECTOR: vec.astype(np.float32).tolist(),
                        COL_EMBED_MODEL: self._embed_model,
                    }
                )

            stats["removed"] = len(stale_ids)
            self._replace_table(new_rows)
            by_id = {str(r[COL_ID]): r for r in new_rows}
            self._load_matrix_from_rows(by_id, expected)
            stats["loaded_from_db"] = len(self._labels)
            trace(
                f"[VECTOR_ROUTER] Synced intent_vectors | expected={stats['expected']} "
                f"embedded={stats['embedded']} removed={stats['removed']} "
                f"loaded={stats['loaded_from_db']} model={self._embed_model}"
            )
            return stats

    def _ensure_matrix(self) -> None:
        if self._ready and self._matrix is not None:
            return
        self.sync_and_validate_intents()

    def _classify_vector_core(
        self,
        raw: str,
        *,
        allowed_intents: frozenset[str] | set[str] | None = None,
        threshold: float | None = None,
    ) -> tuple[str, float]:
        self._ensure_matrix()
        if self._matrix is None or self._matrix.size == 0:
            raise RuntimeError("intent matrix empty")
        q = self._embed(raw)
        scores = self._matrix @ q
        allow = {str(x) for x in allowed_intents} if allowed_intents else None
        if allow:
            masked = np.array(
                [
                    float(scores[i]) if self._labels[i] in allow else -1.0
                    for i in range(len(self._labels))
                ],
                dtype=np.float64,
            )
            scores = masked
        idx = int(np.argmax(scores))
        best = float(scores[idx])
        intent = self._labels[idx]
        cut = float(self.threshold if threshold is None else threshold)
        if allow and intent not in allow:
            trace(
                f"[VECTOR_ROUTER] No slot match | best='{intent}' "
                f"score={best:.3f} allow={sorted(allow)}"
            )
            return "", best
        if best < cut:
            trace(
                f"[VECTOR_ROUTER] No match | best='{intent}' score={best:.3f} "
                f"< threshold={cut:.3f}"
            )
            return "", best
        trace(f"[VECTOR_ROUTER] Matched intent '{intent}' with score={best:.3f}")
        return intent, best

    def classify(
        self,
        user_text: str,
        *,
        allowed_intents: frozenset[str] | set[str] | None = None,
        threshold: float | None = None,
    ) -> tuple[str, float]:
        """
        Return ``(intent, score)``.

        Empty intent when disabled, empty text, embed failure, or
        ``score < threshold`` (free-text answer → gap evaluator).
        ``allowed_intents`` ограничивает argmax интентами активного FSM-слота.
        LanceDB / embed faults / timeouts fall back to exact ``INTENT_RULES``.
        """
        self._degraded = False
        raw = (user_text or "").strip()
        if not raw:
            return "", 0.0
        if not self.enabled:
            return "", 0.0
        try:
            timeout = float(self.timeout_sec or 0.0)
            if timeout > 0:
                from concurrent.futures import ThreadPoolExecutor
                from concurrent.futures import TimeoutError as FuturesTimeout

                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        self._classify_vector_core,
                        raw,
                        allowed_intents=allowed_intents,
                        threshold=threshold,
                    )
                    try:
                        return fut.result(timeout=timeout)
                    except FuturesTimeout as exc:
                        raise TimeoutError("vector intent classify timeout") from exc
            return self._classify_vector_core(
                raw,
                allowed_intents=allowed_intents,
                threshold=threshold,
            )
        except Exception as exc:
            from knowledge_engine.src.resilience_manager import (
                classify_intent_from_rules,
            )

            trace(
                f"[VECTOR_ROUTER] degrade to INTENT_RULES | "
                f"{type(exc).__name__}: {exc}"
            )
            fb = classify_intent_from_rules(raw)
            if allowed_intents and fb not in allowed_intents:
                fb = ""
            self._degraded = True
            return (fb, 1.0) if fb else ("", 0.0)

    def reset(self) -> None:
        """Drop RAM matrix (tests). Does not wipe LanceDB."""
        with self._lock:
            self._ready = False
            self._labels = []
            self._matrix = None


_router_lock = threading.Lock()
_default_router: VectorIntentRouter | None = None


def get_vector_intent_router() -> VectorIntentRouter:
    global _default_router
    if _default_router is None:
        with _router_lock:
            if _default_router is None:
                _default_router = VectorIntentRouter(auto_sync=False)
    return _default_router


def warm_vector_intent_router() -> dict[str, Any]:
    """Startup hook: sync LanceDB catalog + load RAM matrix (idempotent)."""
    router = get_vector_intent_router()
    if not router.enabled:
        trace("[VECTOR_ROUTER] warm skip | VECTOR_INTENT_ENABLED=false")
        return {"enabled": False}
    try:
        stats = router.sync_and_validate_intents()
        stats["enabled"] = True
        return stats
    except Exception as exc:
        trace(f"[VECTOR_ROUTER] warm failed | {type(exc).__name__}: {exc}")
        return {"enabled": True, "error": str(exc)}


def set_vector_intent_router_for_tests(router: VectorIntentRouter | None) -> None:
    """Replace process-wide singleton (unit tests)."""
    global _default_router
    with _router_lock:
        _default_router = router
