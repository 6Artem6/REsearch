"""Migrate old-taxonomy ``[SCOPE: PRINCIPLE]`` / ``[SCOPE: INSTANCE]`` TEXT
tags — baked as literal substrings into free-text narrative fields by the
OLD ``SCOPE_TAGGING_PROMPT_RULES`` before the atom-level taxonomy migration
(see ``apply_reclassification_5_scopes.py``) — to the new taxonomy's
``[SCOPE: CONCEPT]`` / ``[SCOPE: PRACTICE]``.

Scope, confirmed by direct read-only Postgres inspection (not guessed):
``knowledge_atoms.payload['scope']`` is a structured field, already migrated
separately and carries 0 leftover old-tag text. Across every other table
(``v07_chunks``, ``light_rag_facts``, ``edge_case_vectors``,
``intent_vectors``, ``socratic_poles``, ``domain_registry``) 0 rows match.
Exactly two tables / three fields carry literal tag text baked into
narrative strings:
  - document_summaries.payload['document']       (51 rows)
  - document_summaries.payload['key_takeaways']   (51 rows, same row set)
  - rag_chunks.payload['doc_summary_text']        (106 rows)
There is no ``blog_chunks`` table in this schema — every table here follows
the same (id, embedding, payload jsonb, created_at, embed_model) shape.

Safety:
- All UPDATEs run inside ONE transaction (``engine.begin()``); any failure
  rolls back everything, matching the earlier scope migration's approach.
- Plain SQL ``REPLACE()`` on each field's JSONB scalar text value via
  ``jsonb_set`` — no regex needed, since inspection confirmed the tag
  format is byte-consistent (``[SCOPE: PRINCIPLE]`` / ``[SCOPE: INSTANCE]``,
  always with the exact same spacing) across every occurrence found.
- Each of the 3 (table, field) pairs gets its own UPDATE, scoped by a
  WHERE clause on that specific field — never touches a field that does
  not itself contain the old tag, so no accidental NULL-writes or
  unrelated-field churn.
- Post-write verification re-queries Postgres and asserts 0 remaining
  occurrences of both old tags, across the whole ``payload`` (not just the
  3 known fields) — a true from-scratch recount, not just re-checking the
  same fields we wrote.

Usage:
  python knowledge_engine/scripts/apply_chunk_scopes_migration.py --dry-run
  python knowledge_engine/scripts/apply_chunk_scopes_migration.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from knowledge_engine.src.config import settings

_OLD_PRINCIPLE = "[SCOPE: PRINCIPLE]"
_NEW_CONCEPT = "[SCOPE: CONCEPT]"
_OLD_INSTANCE = "[SCOPE: INSTANCE]"
_NEW_PRACTICE = "[SCOPE: PRACTICE]"

# (table, jsonb field) pairs confirmed by inspection to carry old tag text.
_TARGETS = [
    ("document_summaries", "document"),
    ("document_summaries", "key_takeaways"),
    ("rag_chunks", "doc_summary_text"),
]

_ALL_TABLES = [
    "document_summaries",
    "rag_chunks",
    "knowledge_atoms",
    "v07_chunks",
    "light_rag_facts",
    "edge_case_vectors",
    "intent_vectors",
    "socratic_poles",
    "domain_registry",
]


def count_old_tags(engine) -> dict[str, int]:
    """Read-only full-payload recount of old-tag occurrences per table."""
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for tbl in _ALL_TABLES:
            n = conn.execute(
                text(
                    f"SELECT count(*) FROM {tbl} "
                    f"WHERE payload::text LIKE '%[SCOPE: PRINCIPLE]%' "
                    f"OR payload::text LIKE '%[SCOPE: INSTANCE]%'"
                )
            ).scalar()
            counts[tbl] = int(n or 0)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count affected rows/fields, do not write to Postgres.",
    )
    args = parser.parse_args()

    engine = create_engine(
        settings.POSTGRES_SQLALCHEMY_SYNC_DSN, connect_args={"connect_timeout": 5}
    )

    before = count_old_tags(engine)
    print("=== Pre-migration scan (all tables, full payload) ===")
    for tbl, n in before.items():
        flag = "" if n == 0 else "  <-- has old tags"
        print(f"  {tbl}: {n} rows{flag}")

    total_before = sum(before.values())
    if total_before == 0:
        print("\nNothing to migrate — 0 rows contain old-taxonomy tags anywhere.")
        return

    if args.dry_run:
        print(
            f"\n--dry-run: would update rows in: {[t for t, f in _TARGETS]} "
            f"(fields: {[f for t, f in _TARGETS]}). No writes performed."
        )
        return

    updated_per_field: dict[str, int] = {}
    with engine.begin() as conn:  # single transaction — rollback on any exception
        for tbl, field in _TARGETS:
            result = conn.execute(
                text(
                    f"""
                    UPDATE {tbl}
                    SET payload = jsonb_set(
                        payload,
                        '{{{field}}}',
                        to_jsonb(
                            REPLACE(
                                REPLACE(payload->>'{field}', :old_p, :new_c),
                                :old_i, :new_pr
                            )
                        )
                    )
                    WHERE payload->>'{field}' LIKE '%' || :old_p || '%'
                       OR payload->>'{field}' LIKE '%' || :old_i || '%'
                    """
                ),
                {
                    "old_p": _OLD_PRINCIPLE,
                    "new_c": _NEW_CONCEPT,
                    "old_i": _OLD_INSTANCE,
                    "new_pr": _NEW_PRACTICE,
                },
            )
            updated_per_field[f"{tbl}.{field}"] = result.rowcount
    # engine.begin() commits here iff no exception was raised.

    print("\n=== COMMITTED ===")
    for key, n in updated_per_field.items():
        print(f"  {key}: {n} rows updated")

    after = count_old_tags(engine)
    remaining = sum(after.values())
    print("\n=== Post-migration verification (full re-scan, all tables) ===")
    for tbl, n in after.items():
        print(f"  {tbl}: {n} rows")
    print(f"\nRemaining old-tag occurrences: {remaining}")
    if remaining != 0:
        print("MISMATCH: expected 0 remaining occurrences.", file=sys.stderr)
        raise SystemExit(1)
    print(
        "Verification OK: 0 occurrences of [SCOPE: PRINCIPLE] / [SCOPE: INSTANCE] remain."
    )


if __name__ == "__main__":
    main()
