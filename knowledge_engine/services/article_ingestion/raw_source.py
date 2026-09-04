"""Detect raw/source-code URLs so ingest can skip HTML article annotators."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

logger = logging.getLogger(__name__)

_CODE_EXTENSIONS = frozenset(
    {
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hpp",
        ".hxx",
        ".py",
        ".rs",
        ".go",
        ".java",
        ".ts",
        ".js",
        ".cs",
        ".s",
        ".asm",
        ".swift",
        ".kt",
        ".m",
        ".mm",
    }
)


_DOC_EXTENSIONS = frozenset({".md", ".markdown", ".rst", ".txt"})


def _path_has_suffix(path: str, suffixes: frozenset[str]) -> bool:
    for ext in suffixes:
        if path.endswith(ext) or f"{ext}?" in path:
            return True
    return False


def is_code_or_raw_source(url: str, text: str | None = None) -> bool:
    raw_url = (url or "").strip()
    low = raw_url.lower()
    if "raw.githubusercontent.com" in low:
        return True
    path = urlparse(raw_url).path.lower()
    if _path_has_suffix(path, _CODE_EXTENSIONS):
        return True
    if "github.com" in low and _path_has_suffix(path, _DOC_EXTENSIONS):
        return True
    body = (text or "").lstrip()
    if body.startswith(("/*", "//", "#include ", "from __future__", "package ")):
        return True
    return False


def wrap_raw_source_as_annotated(text: str, page_url: str = "") -> AnnotatedArticle:
    """Tag raw code/docs as [P_n] blocks without HTML sidebar/navbar annotation."""
    from knowledge_engine import config as ke_config
    from knowledge_engine.ingest.tiered_code_pruner import maybe_prune_code_for_map

    body = maybe_prune_code_for_map(text, page_url)
    mode = (ke_config.CODE_PARSER_MODE or "linear").strip().lower()
    if mode == "ast":
        try:
            from knowledge_engine.services.article_ingestion.ast_code_chunker import (
                AstCodeChunker,
            )

            return AstCodeChunker().wrap(body, page_url)
        except Exception as exc:
            logger.warning(
                "AST code chunker failed (%s: %s); fallback to linear wrap",
                type(exc).__name__,
                exc,
            )
    return wrap_raw_source_linear(body, page_url)


def maybe_load_github_repo_as_annotated(page_url: str) -> AnnotatedArticle | None:
    """Opt-in Trees API: корпус репозитория как AnnotatedArticle, иначе ``None``."""
    from knowledge_engine.services.article_ingestion.github_tree_loader import (
        maybe_fetch_github_repo_corpus,
    )

    got = maybe_fetch_github_repo_corpus(page_url)
    if got is None:
        return None
    text, _method = got
    try:
        return wrap_raw_source_as_annotated(text, page_url)
    except Exception as exc:
        logger.warning(
            "GitHub Trees wrap failed (%s: %s); fallback to standard pipeline",
            type(exc).__name__,
            exc,
        )
        return None


def wrap_raw_source_linear(text: str, page_url: str = "") -> AnnotatedArticle:
    """Текущая нарезка: пачки ~40 строк → блоки [P_n]."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    paragraph_map: dict[str, str] = {}
    lines_out: list[str] = []
    buf: list[str] = []
    p_idx = 0

    def flush() -> None:
        nonlocal p_idx
        if not buf:
            return
        body = "\n".join(buf).strip()
        buf.clear()
        if len(body) < 2:
            return
        p_idx += 1
        pid = f"P_{p_idx}"
        paragraph_map[pid] = body[:8000]
        lines_out.append(f"[{pid}]\n{body}")

    for line in lines:
        buf.append(line)
        if len(buf) >= 40:
            flush()
    flush()
    ann = AnnotatedArticle(
        annotated_markdown="\n\n".join(lines_out).strip(),
        fig_map={},
        paragraph_map=paragraph_map,
        page_url=(page_url or "").strip(),
    )
    from knowledge_engine.ingest.pipeline_audit import pipeline_audit

    pipeline_audit(
        "Annotate",
        page_url,
        ann.annotated_markdown,
        extra=f"raw_linear P={len(paragraph_map)}",
    )
    return ann
