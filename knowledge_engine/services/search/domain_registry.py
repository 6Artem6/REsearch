"""LanceDB domain_registry: upsert KEEP hosts and cosine lookup for Pass 1."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from knowledge_engine.config import (
    DOMAIN_REGISTRY_COSINE_MIN,
    DOMAIN_REGISTRY_SEARCH_LIMIT,
    LANCE_DB_PATH,
)
from knowledge_engine.db.domain_registry_schema import (
    COL_CLASSIFICATION,
    COL_DOMAIN,
    COL_EMBED_MODEL,
    COL_GENERAL_SUMMARY,
    COL_UPDATED_AT,
    COL_VECTOR,
    DOMAIN_REGISTRY_TABLE,
)
from knowledge_engine.db.embed_model_guard import (
    drop_if_embed_space_mismatch,
    expected_embed_model,
    row_matches_embed_model,
)
from knowledge_engine.db.lancedb_pool import get_lancedb_connection
from knowledge_engine.schemas.llm_contracts.exa_search import (
    AUTHORITY_KEEP_CLASSES,
    PASS1_INCLUDE_CLASSES,
    DomainAuthorityItem,
)
from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3
from knowledge_engine.services.search.exa_domains import clean_domain_for_exa
from knowledge_engine.ui.run_log import trace

_lock = threading.Lock()


def _table_names(db: Any) -> set[str]:
    try:
        names = db.list_tables()
        if hasattr(names, "tables"):
            return set(names.tables)
        return set(names)
    except Exception:
        return set(db.table_names())


def _l2_normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(vec), dtype=np.float64)
    if arr.size == 0:
        return arr
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return arr
    return arr / n


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va = _l2_normalize(a)
    vb = _l2_normalize(b)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    return float(np.dot(va, vb))


def _sql_literal(value: str) -> str:
    return "'" + (value or "").replace("'", "''") + "'"


class DomainRegistry:
    """Persist KEEP domain gist vectors; retrieve OFFICIAL_DOCS by topic cosine."""

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        cosine_min: float | None = None,
        search_limit: int | None = None,
        embed_model: str | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else LANCE_DB_PATH
        self.cosine_min = float(
            DOMAIN_REGISTRY_COSINE_MIN if cosine_min is None else cosine_min
        )
        self.search_limit = int(
            DOMAIN_REGISTRY_SEARCH_LIMIT if search_limit is None else search_limit
        )
        self._embed_model = (embed_model or expected_embed_model()).strip() or (
            expected_embed_model()
        )
        self._space_checked = False

    def _db(self) -> Any:
        db = get_lancedb_connection(self._db_path)
        if not self._space_checked:
            drop_if_embed_space_mismatch(db, DOMAIN_REGISTRY_TABLE)
            self._space_checked = True
        return db

    def _table(self) -> Any | None:
        db = self._db()
        names = _table_names(db)
        if DOMAIN_REGISTRY_TABLE not in names:
            return None
        return db.open_table(DOMAIN_REGISTRY_TABLE)

    def upsert_keep_items(self, items: Sequence[DomainAuthorityItem]) -> int:
        """Embed + upsert KEEP rows. REJECT classes are skipped."""
        keep = [
            it
            for it in items
            if it.classification in AUTHORITY_KEEP_CLASSES
            and clean_domain_for_exa(it.domain)
            and (it.general_summary or "").strip()
        ]
        if not keep:
            return 0
        texts = [(it.general_summary or "").strip() for it in keep]
        try:
            vectors = embed_texts_bge_m3(texts)
        except Exception as exc:
            trace(f"DOMAIN_REGISTRY embed ⊘ | {exc}")
            return 0
        if len(vectors) != len(keep):
            trace("DOMAIN_REGISTRY embed ⊘ | vector count mismatch")
            return 0
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        for item, vec in zip(keep, vectors, strict=True):
            host = clean_domain_for_exa(item.domain)
            rows.append(
                {
                    COL_DOMAIN: host,
                    COL_CLASSIFICATION: item.classification,
                    COL_GENERAL_SUMMARY: (item.general_summary or "").strip()[:400],
                    COL_VECTOR: [float(x) for x in vec],
                    COL_UPDATED_AT: now,
                    COL_EMBED_MODEL: self._embed_model,
                }
            )
        with _lock:
            db = self._db()
            names = _table_names(db)
            if DOMAIN_REGISTRY_TABLE not in names:
                db.create_table(DOMAIN_REGISTRY_TABLE, data=rows)
            else:
                tbl = db.open_table(DOMAIN_REGISTRY_TABLE)
                for row in rows:
                    tbl.delete(f"{COL_DOMAIN} = {_sql_literal(row[COL_DOMAIN])}")
                tbl.add(rows)
        trace(f"DOMAIN_REGISTRY upsert ✓ | n={len(rows)} model={self._embed_model}")
        return len(rows)

    def search_official_docs(self, topic_text: str) -> list[str]:
        """OFFICIAL_DOCS hosts with cosine(topic, general_summary) >= threshold."""
        q = (topic_text or "").strip()
        if not q:
            return []
        tbl = self._table()
        if tbl is None:
            return []
        try:
            qvec = embed_texts_bge_m3([q])[0]
        except Exception as exc:
            trace(f"DOMAIN_REGISTRY search embed ⊘ | {exc}")
            return []
        try:
            hits = (
                tbl.search(qvec)
                .metric("cosine")
                .limit(max(self.search_limit, 8))
                .to_list()
            )
        except Exception:
            try:
                hits = list(tbl.to_pandas().to_dict(orient="records"))
            except Exception as exc:
                trace(f"DOMAIN_REGISTRY search ⊘ | {exc}")
                return []
        out: list[str] = []
        seen: set[str] = set()
        for row in hits:
            if not row_matches_embed_model(row, expected=self._embed_model):
                continue
            host = clean_domain_for_exa(str(row.get(COL_DOMAIN) or ""))
            cls = str(row.get(COL_CLASSIFICATION) or "")
            if not host or host in seen or cls not in PASS1_INCLUDE_CLASSES:
                continue
            stored = row.get(COL_VECTOR)
            if stored is None:
                continue
            sim = _cosine(qvec, stored)
            if sim < self.cosine_min:
                continue
            seen.add(host)
            out.append(host)
            trace(f"DOMAIN_REGISTRY hit ✓ | {host} | cos={sim:.3f}")
            if len(out) >= self.search_limit:
                break
        if not out:
            trace(f"DOMAIN_REGISTRY miss | min_cos={self.cosine_min}")
        return out


_default: DomainRegistry | None = None


def get_domain_registry() -> DomainRegistry:
    global _default
    if _default is None:
        _default = DomainRegistry()
    return _default


def set_domain_registry_for_tests(registry: DomainRegistry | None) -> None:
    global _default
    _default = registry


def reset_domain_registry_for_tests() -> None:
    set_domain_registry_for_tests(None)
