"""Vector lexicon: classify deep_analysis theses as edge-case / trade-off content.

Reference phrases live in LanceDB (``edge_case_vectors``) and load into a RAM
matrix at app startup — no hard-coded word-stem regex for digest ranking.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from knowledge_engine.config import (
    EDGE_CASE_VECTOR_ENABLED,
    EDGE_CASE_VECTOR_THRESHOLD,
    EMBED_MODEL,
    LANCE_DB_PATH,
)
from knowledge_engine.db.edge_case_vectors_schema import (
    COL_EMBED_MODEL,
    COL_ID,
    COL_LABEL,
    COL_PHRASE,
    COL_VECTOR,
    EDGE_CASE_VECTORS_TABLE,
)
from knowledge_engine.ui.run_log import trace

EmbedFn = Callable[[str], list[float] | np.ndarray]

# Seed catalog — edit phrases here; startup sync re-embeds only diffs.
EDGE_CASE_REFERENCE_PHRASES: dict[str, tuple[str, ...]] = {
    "edge_case": (
        "edge case under partial failure",
        "крайний случай при сбое одного воркера",
        "timeout cascade when one subagent hangs",
        "таймаут и каскадная деградация при зависании",
        "race condition and deadlock in async gather",
        "гонка данных и взаимная блокировка в event loop",
        "saturation and backpressure at fan-out",
        "насыщение очереди и backpressure при fan-out",
        "vulnerability under malformed contracts",
        "уязвимость при нарушении контракта данных",
    ),
    "bottleneck": (
        "latency bottleneck in parallel aggregation",
        "узкое место по задержке при агрегации",
        "token overhead and repeated model calls",
        "расход токенов на служебный трафик",
        "rate limit exhaustion under load",
        "исчерпание лимитов провайдера под нагрузкой",
    ),
    "trade_off": (
        "architectural trade-off cancel versus wait",
        "архитектурный компромисс между отменой и ожиданием",
        "latency versus consistency trade-off",
        "компромисс задержки и консистентности",
        "isolation versus completeness of context",
        "изоляция контекста против полноты информации",
    ),
}


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def phrase_vector_id(
    label: str,
    phrase: str,
    *,
    embed_model: str = EMBED_MODEL,
) -> str:
    key = f"{embed_model}\n{label}\n{phrase}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def iter_reference_entries(
    phrases: dict[str, tuple[str, ...]] | None = None,
    *,
    embed_model: str = EMBED_MODEL,
) -> list[tuple[str, str, str]]:
    src = phrases or EDGE_CASE_REFERENCE_PHRASES
    out: list[tuple[str, str, str]] = []
    for label, plist in src.items():
        for phrase in plist:
            p = (phrase or "").strip()
            if not p:
                continue
            out.append((phrase_vector_id(label, p, embed_model=embed_model), label, p))
    return out


class VectorEdgeCaseLexicon:
    """Cosine match of thesis text against seed edge/trade-off phrases."""

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
    ) -> None:
        self.threshold = float(
            EDGE_CASE_VECTOR_THRESHOLD if threshold is None else threshold
        )
        self.enabled = EDGE_CASE_VECTOR_ENABLED if enabled is None else bool(enabled)
        self._embed_fn = embed_fn
        self._phrases = reference_phrases or EDGE_CASE_REFERENCE_PHRASES
        self._persist = bool(persist)
        self._db_path = Path(db_path) if db_path is not None else LANCE_DB_PATH
        self._embed_model = (embed_model or EMBED_MODEL).strip() or EMBED_MODEL
        self._lock = threading.Lock()
        self._ready = False
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None
        self._db = None
        if auto_sync and self.enabled:
            try:
                self.sync_and_validate()
            except Exception as exc:
                trace(
                    f"[EDGE_CASE_LEXICON] startup sync deferred | "
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
        import lancedb

        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))
        return self._db

    def _expected_entries(self) -> list[tuple[str, str, str]]:
        return iter_reference_entries(self._phrases, embed_model=self._embed_model)

    def _read_table_rows(self) -> list[dict[str, Any]]:
        db = self._connect_db()
        names = set(db.table_names())
        if EDGE_CASE_VECTORS_TABLE not in names:
            return []
        table = db.open_table(EDGE_CASE_VECTORS_TABLE)
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
            trace(f"[EDGE_CASE_LEXICON] read failed | {type(exc).__name__}: {exc}")
            return []

    def _replace_table(self, rows: list[dict[str, Any]]) -> None:
        db = self._connect_db()
        names = set(db.table_names())
        if EDGE_CASE_VECTORS_TABLE in names:
            db.drop_table(EDGE_CASE_VECTORS_TABLE)
        if not rows:
            return
        db.create_table(EDGE_CASE_VECTORS_TABLE, data=rows)

    def _load_matrix_from_rows(
        self,
        rows_by_id: dict[str, dict[str, Any]],
        expected: list[tuple[str, str, str]],
    ) -> None:
        labels: list[str] = []
        vecs: list[np.ndarray] = []
        for rid, label, _phrase in expected:
            row = rows_by_id.get(rid)
            if row is None:
                continue
            raw_vec = row.get(COL_VECTOR)
            if raw_vec is None:
                continue
            v = np.asarray(raw_vec, dtype=np.float64).reshape(-1)
            labels.append(label)
            vecs.append(_l2_normalize(v))
        if not vecs:
            self._labels = []
            self._matrix = np.zeros((0, 1), dtype=np.float64)
        else:
            self._labels = labels
            self._matrix = np.vstack(vecs)
        self._ready = True

    def sync_and_validate(self) -> dict[str, Any]:
        """Sync seed phrases ↔ LanceDB; embed only missing/stale rows."""
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
            }

            if not self._persist:
                labels: list[str] = []
                rows: list[np.ndarray] = []
                for _eid, label, phrase in expected:
                    labels.append(label)
                    rows.append(self._embed(phrase))
                    stats["embedded"] += 1
                self._labels = labels
                self._matrix = (
                    np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float64)
                )
                self._ready = True
                trace(
                    f"[EDGE_CASE_LEXICON] In-memory ready | rows={len(labels)} "
                    f"embedded={stats['embedded']}"
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
                self._load_matrix_from_rows(by_id, expected)
                stats["loaded_from_db"] = len(self._labels)
                trace(
                    f"[EDGE_CASE_LEXICON] Catalog intact | "
                    f"loaded={stats['loaded_from_db']} model={self._embed_model}"
                )
                return stats

            new_rows: list[dict[str, Any]] = []
            for eid, label, phrase in expected:
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
                        COL_LABEL: label,
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
                f"[EDGE_CASE_LEXICON] Synced | expected={stats['expected']} "
                f"embedded={stats['embedded']} removed={stats['removed']} "
                f"loaded={stats['loaded_from_db']}"
            )
            return stats

    def _ensure_matrix(self) -> None:
        if self._ready and self._matrix is not None:
            return
        self.sync_and_validate()

    def classify(self, text: str) -> tuple[str, float]:
        """Return ``(label, score)`` or ``(\"\", score)`` below threshold."""
        raw = (text or "").strip()
        if not raw or not self.enabled:
            return "", 0.0
        try:
            self._ensure_matrix()
        except Exception as exc:
            trace(
                f"[EDGE_CASE_LEXICON] matrix init failed | "
                f"{type(exc).__name__}: {exc}"
            )
            return "", 0.0
        if self._matrix is None or self._matrix.size == 0:
            return "", 0.0
        try:
            q = self._embed(raw[:2000])
        except Exception as exc:
            trace(
                f"[EDGE_CASE_LEXICON] embed query failed | "
                f"{type(exc).__name__}: {exc}"
            )
            return "", 0.0
        scores = self._matrix @ q
        idx = int(np.argmax(scores))
        best = float(scores[idx])
        label = self._labels[idx]
        if best < self.threshold:
            return "", best
        return label, best

    def is_edge_related(self, text: str) -> bool:
        """True when thesis is semantically near edge/bottleneck/trade-off."""
        label, _score = self.classify(text)
        return bool(label)

    def reset(self) -> None:
        with self._lock:
            self._ready = False
            self._labels = []
            self._matrix = None


_default_lexicon: VectorEdgeCaseLexicon | None = None
_default_lock = threading.Lock()


def get_edge_case_lexicon() -> VectorEdgeCaseLexicon:
    global _default_lexicon
    with _default_lock:
        if _default_lexicon is None:
            _default_lexicon = VectorEdgeCaseLexicon(auto_sync=False)
        return _default_lexicon


def warm_edge_case_lexicon() -> dict[str, Any]:
    """App startup: sync LanceDB catalog and load RAM matrix."""
    router = get_edge_case_lexicon()
    try:
        return router.sync_and_validate()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def set_edge_case_lexicon_for_tests(lexicon: VectorEdgeCaseLexicon | None) -> None:
    global _default_lexicon
    with _default_lock:
        _default_lexicon = lexicon


def is_edge_related_thesis(text: str) -> bool:
    """Host helper used by digests — vector match, never stem regex."""
    try:
        return get_edge_case_lexicon().is_edge_related(text)
    except Exception as exc:
        trace(f"[EDGE_CASE_LEXICON] classify skip | {exc}")
        return False
