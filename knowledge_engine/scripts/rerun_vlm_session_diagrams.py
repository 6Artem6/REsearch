"""Перепрогон выбранных session.diagrams через VLM (parallel) + ingest mermaid pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from knowledge_engine.services.article_diagram_store import (
    list_diagrams_for_article,
    update_diagram_by_phash,
)
from knowledge_engine.services.article_ingestion.pipeline import _finalize_vlm_mermaid
from knowledge_engine.services.image_filter import ImageSanitizer, get_image_phash
from knowledge_engine.services.mermaid_validate import (
    format_mermaid_for_storage,
    strip_mermaid_fences,
    validate_mermaid_syntax,
)
from knowledge_engine.services.parsers.base import ExtractedImage
from knowledge_engine.services.vlm_batcher import run_vlm_images_parallel
from knowledge_engine.src.node_deep_dive.session_store import get_session, save_session


def _match_row(rows, title: str):
    t = (title or "").strip().lower()
    if not t:
        return None
    for row in rows:
        cap = (row.caption or "").strip().lower()
        if cap == t or t in cap or cap in t:
            return row
    return None


def _find_image_path(figures_dir: Path, phash_prefix: str) -> Path | None:
    pref = (phash_prefix or "").strip()[:16]
    if not pref:
        return None
    for p in sorted(figures_dir.glob("FIG*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        if get_image_phash(p.read_bytes()).startswith(pref):
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run VLM for session diagrams")
    parser.add_argument("--curriculum-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--article-id", required=True)
    parser.add_argument("--diagram-ids", nargs="+", required=True)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO / "knowledge_engine" / ".runs" / "acm_figures_qa",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cid = args.curriculum_id.strip()
    nid = args.node_id.strip()
    aid = args.article_id.strip()
    targets = {d.strip() for d in args.diagram_ids if d.strip()}
    figures_dir = args.figures_dir.expanduser().resolve()

    session = get_session(cid, nid)
    diagrams = list(session.content.diagrams or [])
    rows = list_diagrams_for_article(aid)
    sanitizer = ImageSanitizer()

    work: list[tuple[object, object, Path, ExtractedImage]] = []
    for d in diagrams:
        did = (d.id or "").strip()
        if did not in targets:
            continue
        row = _match_row(rows, d.title or "")
        if row is None:
            print(f"{did}: ⊘ no article_diagrams row for title={d.title!r}")
            continue
        ph = (row.image_phash or "").strip()
        img_path = _find_image_path(figures_dir, ph)
        if img_path is None:
            print(f"{did}: ⊘ no PNG in {figures_dir} for phash={ph[:16]}")
            continue
        data = img_path.read_bytes()
        ok, phash = sanitizer.accept(data, phash=ph)
        if not ok:
            print(f"{did}: ⊘ sanitizer rejected {img_path.name}")
            continue
        caption = (row.caption or d.title or "")[:500]
        ctx = (row.summary or caption)[:400]
        img = ExtractedImage(
            image_bytes=data,
            caption=caption,
            context_text=ctx,
            page_or_pos=len(work) + 1,
            phash=phash,
            mime="image/png",
        )
        work.append((d, row, img_path, img))
        print(f"{did}: ▶ {img_path.name} phash={phash[:16]} caption={caption[:60]}")

    if not work:
        print("nothing to run")
        return 1

    if args.dry_run:
        print(f"dry-run: would VLM {len(work)} images")
        return 0

    pairs = run_vlm_images_parallel(
        [w[3] for w in work],
        label=f"rerun_vlm/{nid[:24]}",
    )
    changed = 0
    for (d, row, img_path, img), (_, item) in zip(work, pairs):
        did = (d.id or "").strip()
        if item is None or not item.is_diagram:
            print(f"{did}: ⊘ VLM returned no diagram")
            continue
        raw_mermaid = (item.mermaid or "").strip()
        fenced = _finalize_vlm_mermaid(raw_mermaid, phash_hint=img.phash)
        if not fenced:
            print(f"{did}: ⊘ mermaid rejected after ingest pipeline")
            continue
        inner = strip_mermaid_fences(fenced)
        ok = validate_mermaid_syntax(inner)
        session_mermaid = format_mermaid_for_storage(inner)
        title = (item.title or d.title or row.caption or "").strip()[:300]
        summary = (item.summary or row.summary or "").strip()[:2000]
        print(
            f"{did}: ✓ valid={ok} lines={inner.count(chr(10))+1} from {img_path.name}"
        )
        if not update_diagram_by_phash(
            aid,
            row.image_phash,
            mermaid_code=session_mermaid,
            caption=title or None,
            summary=summary or None,
        ):
            print(f"{did}: ⊘ DB update failed")
            continue
        d.mermaid = session_mermaid
        if title:
            d.title = title
        changed += 1

    if changed:
        content = session.content.model_copy(update={"diagrams": diagrams})
        save_session(
            cid,
            nid,
            session.node_status,
            content,
            session.history,
            memory=session.memory,
        )
        print(f"saved session + article_diagrams | updated={changed}")
    else:
        print("no successful updates")
    return 0 if changed else 2


if __name__ == "__main__":
    raise SystemExit(main())
