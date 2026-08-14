"""Sanitize Mermaid in all node sessions for a curriculum (backfill).

Pipeline matches serve/persist: normalize_stored_mermaid (regex + format)
before validation; optional Gemma repair for still-invalid diagrams.

Example:
  ./.venv/bin/python -m knowledge_engine.scripts.sanitize_curriculum_diagrams \\
      --curriculum-id agentic_systems_architecture
  ./.venv/bin/python -m knowledge_engine.scripts.sanitize_curriculum_diagrams \\
      --curriculum-id agentic_systems_architecture --with-gemma --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from knowledge_engine.services.mermaid_validate import (
    process_mermaid_for_ingest,
    strip_mermaid_fences,
    validate_mermaid_syntax,
)
from knowledge_engine.src.node_deep_dive.content_assets import (
    normalize_node_content_diagrams,
)
from knowledge_engine.src.node_deep_dive.session_store import get_session, save_session


def _list_session_keys(curriculum_id: str) -> list[tuple[str, str]]:
    import json

    from knowledge_engine.config import PACKAGE_ROOT

    store = PACKAGE_ROOT / ".runs" / "node_deep_dive_sessions.json"
    cid = (curriculum_id or "").strip()
    prefix = f"{cid}::"
    if not store.is_file():
        return []
    try:
        raw = json.loads(store.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    out: list[tuple[str, str]] = []
    for key in sorted(raw.keys()):
        if not str(key).startswith(prefix):
            continue
        parts = str(key).split("::", 1)
        if len(parts) != 2 or not parts[1].strip():
            continue
        out.append((parts[0], parts[1]))
    return out


def _repair_asset_mermaid(raw: str, *, allow_gemma: bool) -> tuple[str, str]:
    """Return (mermaid, reason)."""
    text = (raw or "").strip()
    if not text:
        return "", "empty"
    from knowledge_engine.services.mermaid_validate import normalize_stored_mermaid

    normed = normalize_stored_mermaid(text)
    if not normed:
        return text, "normalize_empty"
    inner = strip_mermaid_fences(normed)
    if validate_mermaid_syntax(inner):
        if normed.strip() != text:
            return normed.strip(), "normalized"
        return text, "already_valid"
    if not allow_gemma:
        return normed.strip() or text, "normalized_still_invalid"
    fixed = process_mermaid_for_ingest(normed or text, allow_gemma_repair=True)
    if not fixed:
        return normed.strip() or text, "gemma_failed"
    if validate_mermaid_syntax(strip_mermaid_fences(fixed)):
        return fixed.strip(), "gemma_repaired"
    return fixed.strip(), "gemma_still_invalid"


def sanitize_curriculum_sessions(
    curriculum_id: str,
    *,
    dry_run: bool = False,
    allow_gemma: bool = False,
) -> dict[str, int]:
    import json

    from knowledge_engine.config import PACKAGE_ROOT
    from knowledge_engine.src.node_deep_dive.schemas import (
        DiagramAsset,
        NodeContentBlock,
    )

    store = PACKAGE_ROOT / ".runs" / "node_deep_dive_sessions.json"
    stats = {
        "sessions": 0,
        "diagrams": 0,
        "changed": 0,
        "invalid_after": 0,
    }
    try:
        all_data = json.loads(store.read_text(encoding="utf-8"))
    except Exception:
        all_data = {}
    if not isinstance(all_data, dict):
        all_data = {}

    for cid, nid in _list_session_keys(curriculum_id):
        key = f"{cid}::{nid}"
        blob = all_data.get(key) or {}
        content_raw = blob.get("content") or {}
        try:
            content = NodeContentBlock.model_validate(content_raw)
        except Exception as exc:
            print(f"{key} | skip invalid content | {exc}")
            continue
        # Raw on-disk mermaid (before get_session normalize).
        raw_by_id = {
            str(d.id or "").strip(): (d.mermaid or "").strip()
            for d in (content.diagrams or [])
        }
        legacy_raw = (content.diagram or "").strip()

        content = normalize_node_content_diagrams(content)
        diagrams = list(content.diagrams or [])
        if not diagrams and not (content.diagram or "").strip():
            continue
        stats["sessions"] += 1
        changed = False
        new_diagrams: list[DiagramAsset] = []
        for d in diagrams:
            stats["diagrams"] += 1
            disk_before = raw_by_id.get((d.id or "").strip(), (d.mermaid or "").strip())
            after, reason = _repair_asset_mermaid(disk_before, allow_gemma=allow_gemma)
            ok = (
                validate_mermaid_syntax(strip_mermaid_fences(after)) if after else False
            )
            title = (d.title or "")[:60]
            print(f"{key} | {d.id} | {reason} | valid={ok} | {title!r}")
            if after and after.strip() != disk_before.strip():
                changed = True
                stats["changed"] += 1
                new_diagrams.append(d.model_copy(update={"mermaid": after}))
            else:
                new_diagrams.append(
                    d.model_copy(update={"mermaid": after or d.mermaid})
                )
            if after and not ok:
                stats["invalid_after"] += 1

        if not diagrams and legacy_raw:
            stats["diagrams"] += 1
            after, reason = _repair_asset_mermaid(legacy_raw, allow_gemma=allow_gemma)
            print(f"{key} | legacy.diagram | {reason}")
            if after and after.strip() != legacy_raw.strip():
                changed = True
                stats["changed"] += 1
                content = content.model_copy(update={"diagram": after})

        if changed:
            content = content.model_copy(update={"diagrams": new_diagrams})
            content = normalize_node_content_diagrams(content)
            if not dry_run:
                session = get_session(cid, nid)
                save_session(
                    cid,
                    nid,
                    session.node_status,
                    content,
                    session.history,
                    memory=session.memory,
                )
                print(f"  saved {key}")
            else:
                print(f"  dry-run skip save {key}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanitize Mermaid diagrams for all node sessions in a curriculum"
    )
    parser.add_argument(
        "--curriculum-id",
        required=True,
        help="e.g. agentic_systems_architecture",
    )
    parser.add_argument(
        "--with-gemma",
        action="store_true",
        help="Run Gemma repair when deterministic sanitize still fails validation",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = sanitize_curriculum_sessions(
        args.curriculum_id.strip(),
        dry_run=args.dry_run,
        allow_gemma=bool(args.with_gemma),
    )
    print(
        f"done | sessions={stats['sessions']} diagrams={stats['diagrams']} "
        f"changed={stats['changed']} invalid_after={stats['invalid_after']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
