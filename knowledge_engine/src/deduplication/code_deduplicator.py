"""Isolated Code Deduplication — project-context-enriched code comparison.

The generic Pre-MAP Bulk Gate (pre_map_deduplicator.py) hands Flash Lite a
flat list of code files with only their own AST Top-6 extract, no signal
about what the project/module is FOR. That's not enough for Flash Lite to
reliably recognize a cross-language algorithmic duplicate (e.g. a Union-Find
/ DSU class in C++ vs Python) — and BGE on raw code is noisy, so it can't
gate code pairs the way it gates text. This module builds a richer,
per-candidate context bundle instead:

1. README discovery (module-level + repo-root), split into Markdown-header
   blocks, filtered down to the Top-K blocks per in-RAM BGE concept anchor
   (ANCHOR_PURPOSE / ANCHOR_ALGORITHMS) — no network calls beyond the one
   README fetch itself.
2. A balanced local project-tree snippet around the target file (parent dir
   contents, capped width; sibling directory NAMES only, one level).
3. A self-contained Tree-Sitter Head-3+Tail-3 CORE-function/method
   extractor with FULL verbatim bodies (all comments/docstrings/type
   annotations kept) plus a same-file call-tree summary — independent of
   the outer pipeline's Flash-Lite Triage pass (this module owns its own
   selection).
4. One dedicated Flash Lite call per code batch, framed as "=== CODE MODULE
   X ===" blocks and asked to judge Behavioral Intent & Data Structure
   State (not syntax), reusing the same CanonicalMapContract shape the
   generic Bulk Gate returns.

Fail-open throughout: any fetch/parse/LLM error for a step just yields an
empty/absent value for that step — never blocks the candidate from being
compared, worst case with less context than intended.

Supports two "project" sources per candidate, both optional (candidates
with neither just skip straight to Tree-Sitter extraction, context-free):
- A GitHub raw URL (raw.githubusercontent.com/{owner}/{repo}/{ref}/{path})
  — README/tree come from the GitHub Git Trees API (one recursive call per
  repo, reusing GitHubTreeLoader).
- A local filesystem path (candidate.url is an existing path or a file://
  URL) — README/tree come from the local disk, no network at all.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from knowledge_engine.config import (
    CODE_DEDUP_GITHUB_TIMEOUT_SEC,
    CODE_DEDUP_MAX_TPM,
    CODE_DEDUP_README_TOP_K,
    CODE_DEDUP_TREE_WIDTH,
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.schemas.llm_contracts.pre_map_dedup import CanonicalMapContract
from knowledge_engine.services.article_ingestion.github_tree_loader import (
    SKIP_DIR_NAMES,
    GitHubTreesApiError,
    GitHubTreeLoader,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    estimate_text_tokens,
)
from knowledge_engine.src.curriculum.pre_flight_triage import _cosine
from knowledge_engine.ui.run_log import trace

IGNORED_TREE_DIRS = SKIP_DIR_NAMES | frozenset({"venv"})

ANCHOR_PURPOSE = (
    "overview purpose architecture main features description what is this"
)
ANCHOR_ALGORITHMS = (
    "data structures algorithms complexity performance implementations internal logic"
)

_CODE_DEDUP_SYSTEM = """
You are a Code Duplication Auditor. You are given CODE MODULES: code files
enriched with project context — path, a balanced project-tree snippet,
relevant README blocks (Root and Module), an in-file call tree, and the
Top-6 CORE functions/methods with FULL bodies (all comments, docstrings,
and type annotations kept intact).

Task: compare modules by Behavioral Intent & Data Structure State — WHAT the
code does and WHICH invariants it maintains (data structure, input/output
behavior, algorithmic identity) — NOT syntactic sugar (programming language,
naming style, formatting, declaration order). The same algorithm implemented
in different programming languages (e.g. Union-Find / Disjoint Set Union in
C++ and in Python — the same path-compression find + union over set
representatives) IS a duplicate (ALIAS), even if the syntax is completely
different.

Do NOT treat modules as duplicates when they have a different purpose, a
different data structure, or different invariants, even if they share a
topic or use similar wording.

For each group of duplicates you find, pick ONE canonical_id — the most
complete/highest-quality module by its own extract — and list the rest as
aliases. A module with no duplicates is simply left out of mappings.

Return ONLY JSON: {"mappings": [{"canonical_id": "...", "aliases": ["...", ...]}]}.
""".strip()


@dataclass
class CodeContextBundle:
    id: str
    path: str
    tree_snippet: str = ""
    root_readme: str = ""
    module_readme: str = ""
    call_tree: str = ""
    code_bodies: str = ""


_MD_BADGE_LINK_RE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")
_MD_BADGE_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG_HINT_RE = re.compile(
    r"<\s*(div|p\s+align|table|picture|source|img|br\s*/?)", re.I
)
_NOISE_LINE_RE = re.compile(r"^[\[\]()\-–—|•\s]*$")


def _sanitize_readme_md(text: str) -> str:
    """README Noise Cleaner: strips HTML badge/logo header blocks and
    Markdown badge-image links BEFORE Markdown-header splitting, so BGE
    doesn't score shields.io/logo clutter against ANCHOR_PURPOSE/
    ANCHOR_ALGORITHMS as if it were real content. markdownify/html2text/
    selectolax are not installed in this environment (checked at
    implementation time); BeautifulSoup (bs4) is an existing project
    dependency and is used here instead, ONLY when the text actually
    contains HTML container tags — plain-Markdown READMEs (the common
    case) skip bs4 entirely, so inline `<`/`>` in fenced code samples
    elsewhere in the doc is never at risk of being reinterpreted as HTML.
    Fail-open: a parse error just returns the text with only the
    regex-based badge stripping applied, never raises."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if _HTML_TAG_HINT_RE.search(raw):
        try:
            from bs4 import BeautifulSoup

            raw = BeautifulSoup(raw, "html.parser").get_text("\n")
        except Exception as exc:
            trace(f"CODE_DEDUP readme_sanitize ✗ | bs4 | {type(exc).__name__}: {exc}")
    raw = _MD_BADGE_LINK_RE.sub("", raw)
    raw = _MD_BADGE_IMG_RE.sub("", raw)
    lines = [
        line for line in raw.splitlines() if line.strip() and not _NOISE_LINE_RE.match(line)
    ]
    return "\n".join(lines).strip()


def _split_markdown_by_headers(md: str) -> list[str]:
    """Splits on ATX headers (#, ##, ### ...); a headerless document is one
    block; an empty document has no blocks."""
    text = (md or "").strip()
    if not text:
        return []
    header_re = re.compile(r"^#{1,6}\s+.+$", re.M)
    positions = [m.start() for m in header_re.finditer(text)]
    if not positions:
        return [text]
    blocks: list[str] = []
    if positions[0] > 0:
        lead = text[: positions[0]].strip()
        if lead:
            blocks.append(lead)
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        block = text[pos:end].strip()
        if block:
            blocks.append(block)
    return blocks


def _top_readme_chunks_by_anchor(
    blocks: list[str], *, top_k: int = CODE_DEDUP_README_TOP_K
) -> list[str]:
    """Fail-open BGE filter: Top-K blocks per anchor (ANCHOR_PURPOSE and
    ANCHOR_ALGORITHMS separately, union'd back into document order) — no
    network calls, purely local embedding model. Returns [] on any error or
    empty input rather than raising."""
    if not blocks:
        return []
    try:
        from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

        block_vecs = embed_texts_bge_m3(blocks)
        anchor_vecs = embed_texts_bge_m3([ANCHOR_PURPOSE, ANCHOR_ALGORITHMS])
    except Exception as exc:
        trace(f"CODE_DEDUP readme_anchor ✗ | {type(exc).__name__}: {exc}")
        return []
    if not block_vecs or not anchor_vecs:
        return []
    picked: set[int] = set()
    for anchor_vec in anchor_vecs:
        scored = sorted(
            range(len(block_vecs)),
            key=lambda i: _cosine(block_vecs[i], anchor_vec),
            reverse=True,
        )
        picked.update(scored[:top_k])
    return [blocks[i] for i in sorted(picked)]


def _parse_raw_github_url(url: str) -> tuple[str, str, str, str] | None:
    """(owner, repo, ref, path) for a raw.githubusercontent.com URL, else
    None. No existing utility parses this host — parse_github_url /
    parse_github_blob_url in github_tree_loader.py only handle github.com."""
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if host != "raw.githubusercontent.com":
        return None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 4:
        return None
    owner, repo, ref, *rest = parts
    path = "/".join(rest)
    if not owner or not repo or not ref or not path:
        return None
    return owner, repo, ref, path


def _format_tree_snippet(parent_dir_name: str, entries: list[tuple[str, bool]]) -> str:
    """entries: list of (name, is_dir). Files listed with their name; a
    sibling directory contributes only its name (no recursion into it)."""
    lines = [f"{parent_dir_name or '.'}/"]
    for name, is_dir in entries[:CODE_DEDUP_TREE_WIDTH]:
        lines.append(f"  {name}/" if is_dir else f"  {name}")
    extra = len(entries) - CODE_DEDUP_TREE_WIDTH
    if extra > 0:
        lines.append(f"  ... (+{extra} more)")
    return "\n".join(lines)


def _github_context(
    owner: str,
    repo: str,
    ref: str,
    path: str,
    *,
    tree_cache: dict[tuple[str, str, str], list[dict]] | None = None,
    cache_lock: threading.Lock | None = None,
) -> tuple[str, str, str]:
    """Sync (called via asyncio.to_thread): (tree_snippet, root_readme,
    module_readme). One recursive Git Trees API call per repo gives both
    the directory listing and the README locations in a single round trip.
    Fail-open: any GitHubTreesApiError/timeout yields ("", "", "").

    tree_cache (owner, repo, ref) -> raw Trees API items, shared across the
    concurrent asyncio.gather calls for one deduplicate_code_candidates()
    batch — several candidates from the same repo/ref (e.g. two files cited
    by different blog posts) reuse the same recursive tree fetch instead of
    repeating it. Scoped to one batch by the caller, not a persistent cache."""
    cache_key = (owner, repo, ref)
    items: list[dict] | None = None
    if tree_cache is not None:
        if cache_lock is not None:
            with cache_lock:
                items = tree_cache.get(cache_key)
        else:
            items = tree_cache.get(cache_key)

    if items is None:
        trace(f"CODE_DEDUP github_tree ▶ | {owner}/{repo}@{ref}")
        t0 = time.monotonic()
        try:
            loader = GitHubTreeLoader(timeout_sec=CODE_DEDUP_GITHUB_TIMEOUT_SEC)
            items = loader.fetch_tree_structure(owner, repo, ref)
        except GitHubTreesApiError as exc:
            trace(
                f"CODE_DEDUP github_tree ✗ | {owner}/{repo}@{ref} | "
                f"{time.monotonic() - t0:.2f}s | {exc}"
            )
            return "", "", ""
        except Exception as exc:
            trace(
                f"CODE_DEDUP github_tree ✗ | {owner}/{repo}@{ref} | "
                f"{time.monotonic() - t0:.2f}s | {type(exc).__name__}: {exc}"
            )
            return "", "", ""
        trace(
            f"CODE_DEDUP github_tree ✓ | {owner}/{repo}@{ref} | "
            f"{time.monotonic() - t0:.2f}s | items={len(items)}"
        )
        if tree_cache is not None:
            if cache_lock is not None:
                with cache_lock:
                    tree_cache[cache_key] = items
            else:
                tree_cache[cache_key] = items
    else:
        trace(f"CODE_DEDUP github_tree ✓ (cache hit) | {owner}/{repo}@{ref} | items={len(items)}")

    module_dir = path.rsplit("/", 1)[0] if "/" in path else ""
    prefix = f"{module_dir}/" if module_dir else ""

    children: dict[str, bool] = {}
    for item in items:
        item_path = str(item.get("path") or "")
        if not item_path.startswith(prefix) or item_path == path:
            continue
        rest = item_path[len(prefix) :]
        if not rest:
            continue
        name = rest.split("/", 1)[0]
        if name in IGNORED_TREE_DIRS:
            continue
        is_dir = "/" in rest or str(item.get("type") or "") == "tree"
        children.setdefault(name, is_dir)
    entries = sorted(children.items(), key=lambda kv: (not kv[1], kv[0].lower()))
    tree_snippet = _format_tree_snippet(module_dir, entries) if entries else ""

    readme_paths = {str(item.get("path") or "").lower(): item for item in items}
    root_readme_path = next(
        (
            str(it.get("path"))
            for k, it in readme_paths.items()
            if k == "readme.md"
        ),
        None,
    )
    module_readme_path = next(
        (
            str(it.get("path"))
            for k, it in readme_paths.items()
            if k == f"{module_dir.lower()}/readme.md" and module_dir
        ),
        None,
    )

    def _fetch_readme(readme_path: str | None) -> str:
        if not readme_path:
            return ""
        trace(f"CODE_DEDUP github_readme ▶ | {readme_path}")
        t_readme = time.monotonic()
        try:
            loader2 = GitHubTreeLoader(timeout_sec=CODE_DEDUP_GITHUB_TIMEOUT_SEC)
            content = loader2.fetch_file_content(owner, repo, readme_path, ref)
        except Exception as exc:
            trace(
                f"CODE_DEDUP github_readme ✗ | {readme_path} | "
                f"{time.monotonic() - t_readme:.2f}s | {type(exc).__name__}: {exc}"
            )
            return ""
        if not content:
            trace(
                f"CODE_DEDUP github_readme ✓ | {readme_path} | "
                f"{time.monotonic() - t_readme:.2f}s | empty"
            )
            return ""
        blocks = _split_markdown_by_headers(_sanitize_readme_md(content))
        top = _top_readme_chunks_by_anchor(blocks)
        trace(
            f"CODE_DEDUP github_readme ✓ | {readme_path} | "
            f"{time.monotonic() - t_readme:.2f}s | blocks={len(blocks)} kept={len(top)}"
        )
        return "\n\n---\n\n".join(top)

    root_readme = _fetch_readme(root_readme_path)
    module_readme = (
        _fetch_readme(module_readme_path)
        if module_readme_path and module_readme_path != root_readme_path
        else ""
    )
    return tree_snippet, root_readme, module_readme


def _local_context(target: Path) -> tuple[str, str, str]:
    """Pure local-disk equivalent of _github_context — no network. Fail-open
    on any filesystem error."""
    try:
        parent = target.parent
        if not parent.is_dir():
            return "", "", ""
        entries: list[tuple[str, bool]] = []
        for child in sorted(parent.iterdir(), key=lambda p: p.name.lower()):
            if child.name in IGNORED_TREE_DIRS or child.name.startswith("."):
                continue
            entries.append((child.name, child.is_dir()))
        tree_snippet = _format_tree_snippet(parent.name, entries) if entries else ""

        def _read_readme(directory: Path) -> str:
            for name in ("README.md", "readme.md", "Readme.md"):
                candidate = directory / name
                if candidate.is_file():
                    try:
                        content = candidate.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except Exception:
                        return ""
                    blocks = _split_markdown_by_headers(_sanitize_readme_md(content))
                    top = _top_readme_chunks_by_anchor(blocks)
                    return "\n\n---\n\n".join(top)
            return ""

        module_readme = _read_readme(parent)
        root = parent
        for _ in range(6):
            if (root / ".git").exists() or root.parent == root:
                break
            root = root.parent
        root_readme = _read_readme(root) if root != parent else module_readme
        if root == parent:
            module_readme = ""
        return tree_snippet, root_readme, module_readme
    except Exception as exc:
        trace(f"CODE_DEDUP local_context ✗ | {target} | {type(exc).__name__}: {exc}")
        return "", "", ""


def _resolve_local_path(url: str) -> Path | None:
    parsed = urlparse((url or "").strip())
    if parsed.scheme == "file":
        candidate = Path(parsed.path)
    elif parsed.scheme in ("", "http", "https") and not parsed.netloc:
        candidate = Path(url)
    else:
        return None
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


async def _fetch_project_context(
    url: str,
    path_for_extension: str,
    *,
    tree_cache: dict[tuple[str, str, str], list[dict]] | None = None,
    cache_lock: threading.Lock | None = None,
) -> tuple[str, str, str]:
    """(tree_snippet, root_readme, module_readme) for a candidate's url —
    GitHub raw URL, local path, or neither (graceful empty context)."""
    local = _resolve_local_path(url)
    if local is not None:
        return await asyncio.to_thread(_local_context, local)
    parsed = _parse_raw_github_url(url)
    if parsed is not None:
        owner, repo, ref, path = parsed
        return await asyncio.to_thread(
            _github_context,
            owner,
            repo,
            ref,
            path,
            tree_cache=tree_cache,
            cache_lock=cache_lock,
        )
    return "", "", ""


def _head_tail_core_with_calltree(
    text: str, url: str, *, min_chars: int
) -> tuple[list[str], str]:
    """Self-contained (no Triage dependency) Tree-Sitter Head-3+Tail-3
    CORE-function/method extractor with full verbatim bodies + a same-file
    call-tree summary line per kept function/method. A class contributes
    ALL of its methods individually (not one collapsed class-level unit).
    Trivial accessors (get_/set_/is_/has_ with no calls) are filtered
    exactly like the shared AST Collapsing pass. Returns ([], "") whenever
    no installed grammar parses the source cleanly."""
    from knowledge_engine.services.article_ingestion.ast_code_chunker import (
        EXTENSION_TO_LANGUAGE,
        AstChunkError,
        parser_for_language,
    )
    from knowledge_engine.src.curriculum.pre_flight_triage import (
        _AST_PROBE_LANGUAGES,
        _SEMANTIC_MAX_CALLS_PER_UNIT,
        _looks_like_trivial_accessor,
        _semantic_collect_calls,
        _semantic_find_body,
        _semantic_function_display_name,
        _semantic_function_nodes,
        _semantic_node_text,
        _semantic_signature_unit,
    )

    sample = (text or "").strip()
    if not sample:
        return [], ""
    path = urlparse(url or "").path.lower()
    candidates = [
        lang for ext, lang in EXTENSION_TO_LANGUAGE.items() if path.endswith(ext)
    ]
    source = sample.encode("utf-8", errors="replace")
    for lang in [*dict.fromkeys(candidates), *_AST_PROBE_LANGUAGES]:
        try:
            parser = parser_for_language(lang)
        except AstChunkError:
            continue
        except Exception:
            continue
        try:
            tree = parser.parse(source)
        except Exception:
            continue
        root = tree.root_node
        if getattr(root, "has_error", True):
            continue

        all_nodes = _semantic_function_nodes(root)
        if not all_nodes:
            continue
        kept: list = []
        for node in all_nodes:
            piece = _semantic_signature_unit(source, node)
            if not piece or len(piece) < min_chars:
                continue
            if _looks_like_trivial_accessor(piece):
                continue
            kept.append((node, piece))
        if not kept:
            continue
        kept.sort(key=lambda pair: pair[0].start_byte)

        call_lines: list[str] = []
        for node, piece in kept:
            body = _semantic_find_body(node)
            first_line = piece.splitlines()[0] if piece else ""
            name = _semantic_function_display_name(first_line) or f"fn@{node.start_byte}"
            calls = _semantic_collect_calls(source, body, depth=0)
            uniq_calls = list(dict.fromkeys(calls))[:_SEMANTIC_MAX_CALLS_PER_UNIT]
            call_lines.append(
                f"{name} -> calls: {', '.join(uniq_calls)}"
                if uniq_calls
                else f"{name} -> calls: (none)"
            )

        nodes_only = [node for node, _ in kept]
        selected = (
            nodes_only[:3] + nodes_only[-3:] if len(nodes_only) > 6 else nodes_only
        )
        bodies = [_semantic_node_text(source, node) for node in selected]
        return bodies, "\n".join(call_lines)
    return [], ""


async def _build_code_context_bundle(
    candidate,
    *,
    min_chars: int,
    tree_cache: dict[tuple[str, str, str], list[dict]] | None = None,
    cache_lock: threading.Lock | None = None,
) -> CodeContextBundle:
    bundle = CodeContextBundle(id=candidate.id, path=candidate.url or candidate.id)
    t0 = time.monotonic()
    try:
        tree_snippet, root_readme, module_readme = await _fetch_project_context(
            candidate.url,
            candidate.url,
            tree_cache=tree_cache,
            cache_lock=cache_lock,
        )
        bundle.tree_snippet = tree_snippet
        bundle.root_readme = root_readme
        bundle.module_readme = module_readme
    except Exception as exc:
        trace(f"CODE_DEDUP context ✗ | {candidate.id[:60]} | {type(exc).__name__}: {exc}")
    t_ast = time.monotonic()
    try:
        bodies, call_tree = await asyncio.to_thread(
            _head_tail_core_with_calltree, candidate.text, candidate.url, min_chars=min_chars
        )
        bundle.code_bodies = "\n\n---\n\n".join(bodies)
        bundle.call_tree = call_tree
    except Exception as exc:
        trace(f"CODE_DEDUP ast ✗ | {candidate.id[:60]} | {type(exc).__name__}: {exc}")
    trace(
        f"CODE_DEDUP bundle ✓ | {candidate.id[:60]} | "
        f"total={time.monotonic() - t0:.2f}s (context={t_ast - t0:.2f}s "
        f"ast={time.monotonic() - t_ast:.2f}s)"
    )
    return bundle


def _build_code_dedup_payload(bundles: dict[str, CodeContextBundle]) -> str:
    parts: list[str] = []
    for cid, b in bundles.items():
        parts.append(
            f"=== CODE MODULE {cid} ===\n"
            f"Path: {b.path}\n"
            f"Tree Snippet: {b.tree_snippet or '(none)'}\n"
            f"Root README: {b.root_readme or '(none)'}\n"
            f"Module README: {b.module_readme or '(none)'}\n"
            f"AST Call Tree: {b.call_tree or '(none)'}\n"
            f"CODE (Head-3 + Tail-3 with Full Bodies & Comments):\n"
            f"{b.code_bodies or '(none)'}\n"
        )
    parts.append(
        "TASK: Determine which CODE MODULES implement the exact same "
        "algorithmic functionality / state-invariants / intent (ALIAS groups)."
    )
    return "\n".join(parts)


async def deduplicate_code_candidates(
    candidates: list,
    *,
    anchor: str = "",
    max_tpm: int | None = None,
    min_chars: int = 20,
) -> dict[str, list[str]]:
    """Isolated code-dedup entry point: enrich every candidate with project
    context, then ONE dedicated Flash Lite comparison call over the whole
    group. Fail-open: an oversized payload or any error yields {} (every
    candidate stays CANONICAL — exactly as if this module were not in the
    call path), never raises out to the caller."""
    if not candidates:
        return {}
    limit = max_tpm if max_tpm is not None else CODE_DEDUP_MAX_TPM

    trace(
        f"CODE_DEDUP bundles ▶ | candidates={len(candidates)} "
        f"(parallel via asyncio.gather, shared tree_cache — see "
        f"_build_code_context_bundle timings below)"
    )
    t_bundles = time.monotonic()
    tree_cache: dict[tuple[str, str, str], list[dict]] = {}
    cache_lock = threading.Lock()
    bundle_list = await asyncio.gather(
        *[
            _build_code_context_bundle(
                c, min_chars=min_chars, tree_cache=tree_cache, cache_lock=cache_lock
            )
            for c in candidates
        ]
    )
    bundles: dict[str, CodeContextBundle] = {b.id: b for b in bundle_list}
    trace(
        f"CODE_DEDUP bundles ✓ | candidates={len(candidates)} "
        f"total={time.monotonic() - t_bundles:.2f}s"
    )

    payload = _build_code_dedup_payload(bundles)
    tokens = estimate_text_tokens(_CODE_DEDUP_SYSTEM) + estimate_text_tokens(payload)
    if tokens > limit:
        trace(f"CODE_DEDUP skip | tokens≈{tokens} > {limit} — payload too large")
        return {}

    from knowledge_engine.services.gemini_stateless import (
        is_gemini_available,
        run_gemini_structured_with_chain,
    )

    if not is_gemini_available():
        return {}
    trace(f"CODE_DEDUP llm ▶ | tokens≈{tokens}")
    t_llm = time.monotonic()
    try:
        result = await asyncio.to_thread(
            run_gemini_structured_with_chain,
            GEMINI_LITE_MODEL,
            _CODE_DEDUP_SYSTEM,
            payload,
            anchor or "code_dedup",
            CanonicalMapContract,
            "code_dedup / bulk_gate",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        )
        trace(f"CODE_DEDUP llm ✓ | {time.monotonic() - t_llm:.2f}s")
        trace(
            f"CODE_DEDUP ✓ | candidates={len(candidates)} tokens≈{tokens} "
            f"mappings={len(result.mappings)}"
        )
        return result.to_dict()
    except Exception as exc:
        trace(f"CODE_DEDUP ✗ | tokens≈{tokens} | {type(exc).__name__}: {exc}")
        return {}
