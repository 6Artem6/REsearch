"""LanceDB embed_model metadata: one Bi-Encoder space (BAAI/bge-m3)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.ui.run_log import trace

COL_EMBED_MODEL = "embed_model"
CANONICAL_BI_ENCODER = "BAAI/bge-m3"


def expected_embed_model() -> str:
    from knowledge_engine.config import EMBED_MODEL

    return (EMBED_MODEL or CANONICAL_BI_ENCODER).strip() or CANONICAL_BI_ENCODER


def stamp_embed_model(
    row: dict[str, Any], *, model: str | None = None
) -> dict[str, Any]:
    out = dict(row)
    out[COL_EMBED_MODEL] = (model or expected_embed_model()).strip()
    return out


def row_matches_embed_model(
    row: dict[str, Any] | None,
    *,
    expected: str | None = None,
) -> bool:
    if not row:
        return False
    want = (expected or expected_embed_model()).strip()
    stored = str(row.get(COL_EMBED_MODEL) or "").strip()
    return bool(stored) and stored == want


def _table_names(db: Any) -> set[str]:
    try:
        names = db.list_tables()
        if hasattr(names, "tables"):
            return set(names.tables)
        return set(names)
    except Exception:
        try:
            return set(db.table_names())
        except Exception:
            return set()


def drop_if_embed_space_mismatch(db: Any, table_name: str) -> bool:
    """Drop ``table_name`` when stored embed_model ≠ current Bi-Encoder.

    nomic-embed-text (768-d) and BAAI/bge-m3 (1024-d) must not share a table.
    Missing ``embed_model`` is treated as a foreign (legacy) space.
    """
    if not table_name or table_name not in _table_names(db):
        return False
    expected = expected_embed_model()
    try:
        tbl = db.open_table(table_name)
        rows = tbl.head(1).to_pandas().to_dict(orient="records")
    except Exception:
        try:
            rows = db.open_table(table_name).to_arrow().to_pylist()[:1]
        except Exception as exc:
            trace(f"EMBED_GUARD read ⊘ | {table_name} | {exc}")
            return False
    if not rows:
        return False
    stored = str(rows[0].get(COL_EMBED_MODEL) or "").strip()
    if stored == expected:
        return False
    try:
        db.drop_table(table_name)
    except Exception as exc:
        trace(f"EMBED_GUARD drop ⊘ | {table_name} | {exc}")
        return False
    trace(
        f"EMBED_GUARD drop ✓ | {table_name} stored={stored or '∅'} "
        f"expected={expected}"
    )
    return True


def drop_incompatible_embed_tables(
    db: Any, table_names: tuple[str, ...] | list[str]
) -> int:
    n = 0
    for name in table_names:
        if drop_if_embed_space_mismatch(db, name):
            n += 1
    return n
