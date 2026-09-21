"""Migrate knowledge_atoms.payload['scope'] from the 3-category taxonomy
(PRINCIPLE/MECHANIC/INSTANCE) to the 5-category taxonomy (CONCEPT/MECHANIC/
PRACTICE/EDGE_CASE/ANTI_PATTERN) in real Postgres, using the dry-run
classification results already produced and verified.

Source of truth: the resumable JSON cache written by
``audit_and_reclassify_atoms_5_scopes.py``
(knowledge_engine/.runs/reclassification_5_scopes_cache.json) — a complete
id -> new_scope mapping for all 703 real atoms. The human-readable
``reclassification_5_scopes_result.log`` only carries aggregate counts and a
handful of examples, not a full per-atom mapping, so it cannot drive this
migration directly.

Safety choices:
- All UPDATEs run inside ONE transaction (SQLAlchemy ``engine.begin()``):
  any single failure rolls back everything, leaving Postgres exactly as it
  was pre-migration ("в единой транзакции ... с откатом при ошибке").
- Each row is updated by a direct, id-targeted
  ``UPDATE ... SET payload = jsonb_set(...) WHERE payload->>'id' = :id``,
  NOT by re-running ``VectorStore.upsert_knowledge_atoms`` (the path
  ``backfill_core_relevance_score.py`` uses). Upsert recomputes the row id
  via ``generate_vector_id(doc_id, statement)`` — the exact mechanism that
  produced 185 duplicate orphaned rows earlier in this project (a
  pre-existing id-mismatch for 6 source-code articles). A targeted
  UPDATE-by-existing-id touches only rows that already exist under that id
  and carries none of that risk.
- Idempotent: jsonb_set on an already-migrated row is a no-op rewrite, so a
  re-run after a partial failure is safe.
- Every row's matched rowcount is checked; a 0-rowcount UPDATE (id not
  found — should never happen since ids come straight from a fresh SELECT)
  aborts the whole transaction rather than silently under-migrating.

Usage:
  python knowledge_engine/scripts/apply_reclassification_5_scopes.py --dry-run
  python knowledge_engine/scripts/apply_reclassification_5_scopes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import create_engine, text

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from knowledge_engine.src.config import settings

_CACHE_PATH = (
    Path(REPO_ROOT)
    / "knowledge_engine"
    / ".runs"
    / "reclassification_5_scopes_cache.json"
)
_VALID_NEW_SCOPES = {"CONCEPT", "MECHANIC", "PRACTICE", "EDGE_CASE", "ANTI_PATTERN"}


def load_target_mapping() -> dict[str, str]:
    """id -> new_scope, from the dry-run script's resumable cache."""
    if not _CACHE_PATH.exists():
        print(f"Cache not found: {_CACHE_PATH}", file=sys.stderr)
        raise SystemExit(1)
    raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for atom_id, rec in raw.items():
        new_scope = str(rec.get("new_scope") or "").strip().upper()
        if new_scope not in _VALID_NEW_SCOPES:
            print(f"  SKIP {atom_id}: invalid cached new_scope={new_scope!r}")
            continue
        mapping[atom_id] = new_scope
    return mapping


def load_current_scopes(engine) -> dict[str, str]:
    """id -> current payload['scope'], read-only, for pre-flight verification."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT payload->>'id' AS id, payload->>'scope' AS scope FROM knowledge_atoms"
            )
        ).fetchall()
    return {r.id: (r.scope or "").strip().upper() for r in rows if r.id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and pre-flight checks, but do not write to Postgres.",
    )
    args = parser.parse_args()

    mapping = load_target_mapping()
    print(f"Loaded {len(mapping)} target (id -> new_scope) pairs from cache.")

    engine = create_engine(
        settings.POSTGRES_SQLALCHEMY_SYNC_DSN, connect_args={"connect_timeout": 5}
    )
    current = load_current_scopes(engine)
    print(f"Postgres currently has {len(current)} knowledge_atoms rows.")

    missing_in_db = [aid for aid in mapping if aid not in current]
    missing_in_cache = [aid for aid in current if aid not in mapping]
    if missing_in_db:
        print(
            f"ABORT: {len(missing_in_db)} cached atom id(s) not found in Postgres "
            f"(e.g. {missing_in_db[:5]}) — refusing to run a partial migration.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if missing_in_cache:
        print(
            f"WARNING: {len(missing_in_cache)} DB row(s) have no cached "
            f"reclassification (e.g. {missing_in_cache[:5]}) — they will be left "
            "untouched (still on the old 3-category scope value)."
        )

    old_dist = Counter(current[aid] for aid in mapping)
    new_dist = Counter(mapping.values())
    print("Baseline (3-scope) distribution over the 703 atoms being migrated:")
    for scope, n in sorted(old_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {scope}: {n}")
    print("Target (5-scope) distribution:")
    for scope in ("CONCEPT", "MECHANIC", "PRACTICE", "EDGE_CASE", "ANTI_PATTERN"):
        print(f"  {scope}: {new_dist.get(scope, 0)}")

    if args.dry_run:
        print("\n--dry-run: no Postgres writes performed.")
        return

    updated = 0
    zero_match_ids: list[str] = []
    with engine.begin() as conn:  # single transaction — rollback on any exception
        for atom_id, new_scope in mapping.items():
            result = conn.execute(
                text(
                    """
                    UPDATE knowledge_atoms
                    SET payload = jsonb_set(payload, '{scope}', to_jsonb(CAST(:new_scope AS text)))
                    WHERE payload->>'id' = :atom_id
                    """
                ),
                {"new_scope": new_scope, "atom_id": atom_id},
            )
            if result.rowcount == 0:
                zero_match_ids.append(atom_id)
            elif result.rowcount == 1:
                updated += 1
            else:
                # A duplicate-id row would match >1 — abort rather than
                # silently mutate more rows than intended.
                raise RuntimeError(
                    f"UPDATE matched {result.rowcount} rows for id={atom_id} "
                    "(expected exactly 1) — aborting transaction."
                )
        if zero_match_ids:
            raise RuntimeError(
                f"{len(zero_match_ids)} id(s) matched 0 rows during UPDATE "
                f"(e.g. {zero_match_ids[:5]}) — aborting transaction, 0 rows committed."
            )
    # engine.begin() context commits here iff no exception was raised.

    print(f"\nCOMMITTED: {updated} rows updated, 0 errors/mismatches.")

    # Post-write verification: read back and confirm every migrated id now
    # carries exactly its target scope.
    after = load_current_scopes(engine)
    verified = sum(1 for aid, scope in mapping.items() if after.get(aid) == scope)
    lines = [
        "=== Migration checklist ===",
        f"Rows updated: {updated} / {len(mapping)}",
        f"Post-write verification (re-read from Postgres): {verified} / {len(mapping)} match target scope",
        f"Errors/mismatches: {0 if verified == len(mapping) else len(mapping) - verified}",
    ]
    print("\n".join(lines))
    if verified != len(mapping):
        print(
            "MISMATCH DETECTED after commit — investigate immediately.", file=sys.stderr
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
