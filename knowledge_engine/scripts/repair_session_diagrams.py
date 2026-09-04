"""Точечный ремонт Mermaid в session.content.diagrams (sanitize → validate → Gemma)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from knowledge_engine.services.mermaid_validate import (
    _needs_ui_repair,
    format_mermaid_for_storage,
    process_mermaid_for_ingest,
    strip_mermaid_fences,
    validate_mermaid_syntax,
)
from knowledge_engine.src.node_deep_dive.schemas import DiagramAsset
from knowledge_engine.src.node_deep_dive.session_store import get_session, save_session


def _repair_one(raw: str, *, allow_gemma: bool) -> tuple[str | None, str]:
    """Возвращает (новый mermaid или None, reason)."""
    text = (raw or "").strip()
    if not text:
        return None, "empty"

    inner = strip_mermaid_fences(raw)
    if validate_mermaid_syntax(inner) and not _needs_ui_repair(inner):
        normed = format_mermaid_for_storage(text).strip()
        ninner = strip_mermaid_fences(normed)
        if validate_mermaid_syntax(ninner) and normed != text:
            return normed, "normalized"
        return None, "already_valid"

    fixed = process_mermaid_for_ingest(text, allow_gemma_repair=allow_gemma)
    if not fixed:
        return None, "repair_failed"
    finner = strip_mermaid_fences(fixed)
    if not validate_mermaid_syntax(finner) or _needs_ui_repair(finner):
        return None, "still_invalid"
    if fixed.strip() == text:
        return None, "unchanged"
    return fixed.strip(), "repaired"


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair selected session diagrams")
    parser.add_argument("--curriculum-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--diagram-ids",
        nargs="+",
        required=True,
        help="e.g. diagram-1 diagram-3  or  1 3",
    )
    parser.add_argument("--no-gemma", action="store_true")
    parser.add_argument(
        "--force-gemma",
        action="store_true",
        help="Send to Gemma even if currently valid",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cid = args.curriculum_id.strip()
    nid = args.node_id.strip()

    def _norm(s: str) -> str:
        t = (s or "").strip()
        if t.isdigit():
            return f"diagram-{t}"
        return t

    targets = {_norm(d) for d in args.diagram_ids if (d or "").strip()}

    session = get_session(cid, nid)
    diagrams = list(session.content.diagrams or [])
    if not diagrams and (session.content.diagram or "").strip():
        diagrams = [
            DiagramAsset(id="diagram-1", title="", mermaid=session.content.diagram),
        ]

    changed = 0
    for d in diagrams:
        did = (d.id or "").strip()
        if did not in targets:
            continue
        raw = d.mermaid or ""
        before_ok = validate_mermaid_syntax(strip_mermaid_fences(raw))
        if before_ok and args.force_gemma and not args.no_gemma:
            # Force path: treat as invalid for repair_one by using ingest directly
            from knowledge_engine.services.mermaid_validate import (
                process_mermaid_for_ingest,
            )

            fixed = process_mermaid_for_ingest(raw, allow_gemma_repair=True)
            if fixed and validate_mermaid_syntax(strip_mermaid_fences(fixed)):
                new, reason = fixed.strip(), "force_repaired"
            else:
                new, reason = None, "force_failed"
            after_ok = (
                validate_mermaid_syntax(strip_mermaid_fences(new)) if new else before_ok
            )
        else:
            new, reason = _repair_one(
                raw,
                allow_gemma=not args.no_gemma,
            )
            after_ok = (
                validate_mermaid_syntax(strip_mermaid_fences(new)) if new else before_ok
            )
        print(f"{did}: before_valid={before_ok} -> {reason} after_valid={after_ok}")
        if new and not args.dry_run:
            d.mermaid = new
            changed += 1

    if changed and not args.dry_run:
        content = session.content.model_copy(update={"diagrams": diagrams})
        save_session(
            cid,
            nid,
            session.node_status,
            content,
            session.history,
            memory=session.memory,
        )
        print(f"saved session | updated={changed}")
    elif args.dry_run:
        print("dry-run: no save")
    else:
        print("no changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
