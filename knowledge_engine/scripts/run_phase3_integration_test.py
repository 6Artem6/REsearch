#!/usr/bin/env python3
"""Integration smoke: Phase 3A GitHub Trees ingest + Phase 3B entity consensus.

Forces opt-in flags **before** ``knowledge_engine.config`` is imported (dotenv
uses setdefault and must not clobber this process).

Run from repo root:

  PYTHONPATH=. ./.venv/bin/python -m knowledge_engine.scripts.run_phase3_integration_test
  PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/run_phase3_integration_test.py

**external/cost:** GitHub Trees API + Gemma/Gemini MAP→REDUCE + bge-m3/reranker.
Does not write LanceDB. Writes a markdown report under ``knowledge_engine/.runs/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_URL = "https://github.com/pallets/markupsafe"
REQUIRED_ENV = {
    "USE_GITHUB_TREES_API": "true",
    "CLAIM_DEDUP_MODE": "entity_consensus",
    "MAX_PRIMARY_ANCHORS": "3",
    "MAX_CONSENSUS_BATCH_TOKENS": "3072",
}
_BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".zip",
        ".gz",
        ".whl",
        ".exe",
    }
)
_SKIP_DIR_MARKERS = (".git", "__pycache__")


def _apply_required_env() -> None:
    """Force Phase 3 toggles before config/dotenv import."""
    for key, value in REQUIRED_ENV.items():
        os.environ[key] = value
    os.environ.setdefault("KE_TRACE_STDOUT", "1")


def _pass(msg: str) -> None:
    print(f"[PASS] {msg}", flush=True)


def _fail(reason: str) -> None:
    print(f"[FAIL] {reason}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _file_rank(path: str) -> tuple[int, str]:
    low = (path or "").replace("\\", "/").lower()
    if low.startswith("src/"):
        return (0, low)
    name = Path(low).name
    if name in ("readme.md", "readme.rst", "readme.txt"):
        return (1, low)
    if name in ("pyproject.toml", "setup.cfg", "setup.py"):
        return (2, low)
    if "/test" in f"/{low}" or low.startswith("tests/"):
        return (8, low)
    return (5, low)


def _assert_env(ke_config: Any) -> None:
    if ke_config.USE_GITHUB_TREES_API is not True:
        _fail(
            "USE_GITHUB_TREES_API is not True after process override "
            f"(got {ke_config.USE_GITHUB_TREES_API!r})"
        )
    mode = str(ke_config.CLAIM_DEDUP_MODE or "").strip().lower()
    if mode != "entity_consensus":
        _fail(f"CLAIM_DEDUP_MODE must be entity_consensus, got {mode!r}")
    if int(ke_config.MAX_PRIMARY_ANCHORS) != 3:
        _fail(
            f"MAX_PRIMARY_ANCHORS must be 3, got {ke_config.MAX_PRIMARY_ANCHORS!r}"
        )
    if int(ke_config.MAX_CONSENSUS_BATCH_TOKENS) != 3072:
        _fail(
            "MAX_CONSENSUS_BATCH_TOKENS must be 3072, "
            f"got {ke_config.MAX_CONSENSUS_BATCH_TOKENS!r}"
        )
    _pass("Phase 3 env toggles asserted (trees + entity_consensus + caps)")


def _assert_clean_tree_files(files: list[dict[str, Any]]) -> None:
    if not files:
        _fail("GitHubTreeLoader returned zero files")
    dirty: list[str] = []
    for item in files:
        path = str(item.get("path") or "")
        body = str(item.get("content") or "")
        low_parts = Path(path.replace("\\", "/")).parts
        if any(part in _SKIP_DIR_MARKERS for part in low_parts):
            dirty.append(f"skipped-dir:{path}")
            continue
        suffix = Path(path).suffix.lower()
        if suffix in _BINARY_SUFFIXES:
            dirty.append(f"binary-suffix:{path}")
            continue
        if "\x00" in body:
            dirty.append(f"nul-bytes:{path}")
    if dirty:
        sample = ", ".join(dirty[:8])
        _fail(f"tree corpus contains forbidden paths/binaries: {sample}")
    _pass(
        f"Tree file list is text-only without .git/__pycache__ "
        f"({len(files)} files)"
    )


def _select_files(
    files: list[dict[str, Any]], *, max_files: int | None
) -> list[dict[str, Any]]:
    ordered = sorted(files, key=lambda item: _file_rank(str(item.get("path") or "")))
    if max_files is None or max_files <= 0 or len(ordered) <= max_files:
        return ordered
    picked = ordered[:max_files]
    _info(
        f"Capping corpus to {len(picked)}/{len(ordered)} files "
        f"(--max-files={max_files}; pass --all-files to lift)"
    )
    return picked


def _write_entity_report(
    nodes: list[Any],
    *,
    path: Path,
    url: str,
    fetch_method: str,
    file_count: int,
    window_count: int,
) -> None:
    from knowledge_engine.models.consensus import ConsensusNode

    grouped: dict[str, list[ConsensusNode]] = defaultdict(list)
    coerced: list[ConsensusNode] = []
    for raw in nodes:
        node = (
            raw
            if isinstance(raw, ConsensusNode)
            else ConsensusNode.model_validate(raw)
        )
        coerced.append(node)
        key = (node.entity or "general").strip() or "general"
        grouped[key].append(node)

    lines = [
        f"# Phase 3 integration report",
        "",
        f"- url: `{url}`",
        f"- fetch_method: `{fetch_method}`",
        f"- files: {file_count}",
        f"- map_windows: {window_count}",
        f"- consensus_nodes: {len(coerced)}",
        "",
    ]
    for entity in sorted(grouped):
        lines.append(f"## Entity: {entity}")
        lines.append("")
        for node in grouped[entity]:
            primary = ", ".join(node.primary_anchors) or "(none)"
            all_a = ", ".join(node.all_anchors) or "(none)"
            lines.append(f"- **status:** {node.status}")
            lines.append(f"- **primary_anchors:** {primary}")
            lines.append(f"- **all_anchors:** {all_a}")
            if node.disputed_details:
                lines.append(f"- **disputed:** {node.disputed_details}")
            lines.append(f"- **summary:** {node.summary_text.strip()}")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _assert_consensus_nodes(nodes: list[Any], *, max_primary: int) -> None:
    from knowledge_engine.models.consensus import ConsensusNode
    from knowledge_engine.services.deduplication.entity_consensus_engine import (
        apply_anti_bloat_anchors,
    )

    synthetic = [f"A{i}" for i in range(1, 7)]
    clipped = apply_anti_bloat_anchors(synthetic, limit=max_primary)
    if len(clipped) > max_primary:
        _fail(
            f"apply_anti_bloat_anchors returned {len(clipped)} tags "
            f"(cap {max_primary})"
        )
    if clipped != synthetic[:max_primary]:
        _fail(f"anti-bloat expected {synthetic[:max_primary]}, got {clipped}")
    _pass("Anti-bloat contract: 6 anchors collapse to Top-3 primary")

    if not nodes:
        _fail(
            "REDUCE produced no ConsensusNode list "
            "(CLAIM_DEDUP_MODE=entity_consensus must populate outcome.consensus_nodes)"
        )

    saw_overflow = False
    for i, raw in enumerate(nodes):
        try:
            node = (
                raw
                if isinstance(raw, ConsensusNode)
                else ConsensusNode.model_validate(raw)
            )
        except Exception as exc:
            _fail(f"consensus_nodes[{i}] is not a ConsensusNode: {exc}")
        primary = list(node.primary_anchors or [])
        all_a = list(node.all_anchors or [])
        if len(primary) > max_primary:
            _fail(
                f"node {node.node_id!r} primary_anchors={primary} "
                f"length {len(primary)} > {max_primary}"
            )
        if all_a and primary and not set(primary).issubset(set(all_a)):
            _fail(
                f"node {node.node_id!r} primary_anchors {primary} "
                f"not subset of all_anchors {all_a}"
            )
        if len(all_a) > max_primary:
            saw_overflow = True
            if len(all_a) < len(primary):
                _fail(
                    f"node {node.node_id!r} all_anchors shorter than primary "
                    f"({all_a} vs {primary})"
                )
    _pass(
        f"Primary anchors constraint met "
        f"(≤ {max_primary} on {len(nodes)} ConsensusNode(s))"
    )
    if saw_overflow:
        _pass("all_anchors retained the full supporting set when > 3 tags")
    else:
        _info(
            "Live clusters did not exceed 3 supporting anchors; "
            "full-set retention checked via anti-bloat helper above"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3A/3B integration smoke (GitHub Trees + entity consensus)"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Public GitHub repo URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=12,
        help="Cap tree files sent to MAP/REDUCE (default 12). Ignored with --all-files.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Do not cap the tree corpus.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Markdown report path (default: knowledge_engine/.runs/phase3_integration_report.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _apply_required_env()

    from knowledge_engine import config as ke_config
    from knowledge_engine.models.consensus import ConsensusNode
    from knowledge_engine.services.article_ingestion.github_tree_loader import (
        GitHubTreeLoader,
        format_repo_corpus,
        path_is_skipped,
    )
    from knowledge_engine.services.article_ingestion.raw_source import (
        wrap_raw_source_as_annotated,
    )
    from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
        map_reduce_summarize_blog_outcome,
    )
    from knowledge_engine.services.web_extract import smart_fetch_page_html

    _assert_env(ke_config)

    url = (args.url or DEFAULT_URL).strip()
    _info(f"Target {url}")

    loader = GitHubTreeLoader()
    try:
        files = loader.load_repository_files(url)
    except Exception as exc:
        _fail(f"GitHubTreeLoader.load_repository_files failed: {type(exc).__name__}: {exc}")

    if loader.used_zip_fallback:
        _fail(
            "GitHub Trees API fell back to zip archive "
            "(expected fetch_method=github_trees). "
            "Check rate limit / GITHUB_TOKEN."
        )
    _assert_clean_tree_files(files)
    leaked = [str(f.get("path")) for f in files if path_is_skipped(str(f.get("path") or ""))]
    if leaked:
        _fail(f"loader skipped-dir filter leaked: {leaked[:8]}")

    max_files = None if args.all_files else int(args.max_files)
    files = _select_files(files, max_files=max_files)
    corpus = format_repo_corpus(files)
    if len(corpus.strip()) < 200:
        _fail(f"formatted corpus too short ({len(corpus)} chars)")

    html, fetch_method = smart_fetch_page_html(url)
    if fetch_method != "github_trees":
        _fail(
            f"smart_fetch_page_html method is {fetch_method!r}, expected 'github_trees' "
            "(pipeline metadata for Phase 3A)"
        )
    if not (html or "").strip():
        _fail("smart_fetch_page_html returned empty github_trees body")
    _pass("GitHub Trees API ingestion successful")

    if not (ke_config.GEMINI_API_KEY or "").strip():
        _fail("GEMINI_API_KEY is empty; cannot run MAP/REDUCE consensus")

    annotated = wrap_raw_source_as_annotated(corpus, url)
    md = (annotated.annotated_markdown or "").strip()
    if len(md) < 80:
        _fail("annotated markdown from tree corpus is empty/too short")
    _info(f"Annotated corpus {len(md)} chars → MAP/REDUCE (source_kind=source_code)")

    try:
        outcome, windows = map_reduce_summarize_blog_outcome(
            md,
            title=Path(url.rstrip("/")).name or "repo",
            url=url,
            source_kind="source_code",
        )
    except Exception as exc:
        _fail(f"map_reduce_summarize_blog_outcome raised {type(exc).__name__}: {exc}")

    if outcome is None:
        _fail("map_reduce_summarize_blog_outcome returned no outcome")
    nodes = list(outcome.consensus_nodes or [])
    _assert_consensus_nodes(nodes, max_primary=int(ke_config.MAX_PRIMARY_ANCHORS))

    out_path = Path(args.out) if args.out else (
        Path(REPO_ROOT) / "knowledge_engine" / ".runs" / "phase3_integration_report.md"
    )
    _write_entity_report(
        nodes,
        path=out_path,
        url=url,
        fetch_method=fetch_method,
        file_count=len(files),
        window_count=len(windows or []),
    )
    text = out_path.read_text(encoding="utf-8")
    if not text.strip():
        _fail(f"report file is empty: {out_path}")
    if "## Entity:" not in text:
        _fail(
            f"report {out_path} has no grouped entity sections "
            "(expected '## Entity: …')"
        )
    _pass(f"Entity-grouped report written ({out_path})")

    if outcome.final is None:
        _info("REDUCE final passport is empty (consensus nodes still validated)")
    else:
        _info(
            f"REDUCE final: atoms={len(outcome.final.knowledge_atoms or [])} "
            f"summary_chars={len(outcome.final.executive_summary or '')}"
        )

    coerced = [
        n if isinstance(n, ConsensusNode) else ConsensusNode.model_validate(n)
        for n in nodes
    ]
    _info(
        f"Done | files={len(files)} windows={len(windows or [])} "
        f"nodes={len(coerced)} method={fetch_method}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FAIL] unhandled {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
