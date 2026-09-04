"""AST-нарезка исходников (tree-sitter). Язык — только по расширению URL."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

logger = logging.getLogger(__name__)

# Расширение → каноническое имя языка (пакет ``tree_sitter_<name>``, кроме custom).
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".java": "java",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
}

_PARSER_CACHE: dict[str, Any] = {}


class AstChunkError(Exception):
    """Парсинг/политика AST: вызывающий обязан упасть в linear."""


def _language_from_typescript_module(attr: str) -> Any:
    from tree_sitter import Language

    mod = importlib.import_module("tree_sitter_typescript")
    factory = getattr(mod, attr, None)
    if factory is None:
        raise AstChunkError(
            f"tree_sitter_typescript has no {attr}(); cannot load grammar"
        )
    return Language(factory())


def _load_typescript_language() -> Any:
    return _language_from_typescript_module("language_typescript")


def _load_tsx_language() -> Any:
    return _language_from_typescript_module("language_tsx")


_CUSTOM_LANGUAGE_LOADERS: dict[str, Callable[[], Any]] = {
    "typescript": _load_typescript_language,
    "tsx": _load_tsx_language,
}

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 500
TARGET_MIN_LINES = 40
TARGET_MAX_LINES = 80
MAX_FUNCTION_LINES = 150

_FUNC_TYPES = frozenset(
    {
        "function_definition",
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "arrow_function",
        "function",
        "decorated_definition",
    }
)
_CLASS_TYPES = frozenset(
    {
        "class_definition",
        "class_declaration",
        "class",
    }
)


def linear_chunk_code(text: str, page_url: str = "") -> AnnotatedArticle:
    from knowledge_engine.services.article_ingestion.raw_source import (
        wrap_raw_source_linear,
    )

    return wrap_raw_source_linear(text, page_url)


def language_from_url(url: str) -> str | None:
    """Язык только по ``Path(...).suffix``; нет пути/расширения → ``None`` (linear)."""
    raw = (url or "").strip()
    if not raw:
        return None
    suffix = Path(urlparse(raw).path).suffix.lower()
    if not suffix:
        return None
    return EXTENSION_TO_LANGUAGE.get(suffix)


def _load_language(lang_name: str) -> Any:
    name = (lang_name or "").strip()
    if not name:
        raise AstChunkError("empty tree-sitter language name")
    loader = _CUSTOM_LANGUAGE_LOADERS.get(name)
    if loader is not None:
        try:
            return loader()
        except AstChunkError:
            raise
        except Exception as exc:
            raise AstChunkError(
                f"failed to load custom tree-sitter grammar {name!r}"
            ) from exc
    module_name = f"tree_sitter_{name}"
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise AstChunkError(
            f"tree-sitter grammar {module_name} is not installed"
        ) from exc
    factory = getattr(mod, "language", None)
    if factory is None:
        raise AstChunkError(f"{module_name} has no language()")
    try:
        from tree_sitter import Language

        return Language(factory())
    except Exception as exc:
        raise AstChunkError(f"failed to init Language for {name!r}") from exc


def _make_parser(language: Any) -> Any:
    from tree_sitter import Parser

    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.language = language
        return parser


def parser_for_language(lang_name: str) -> Any:
    """Ленивый parser registry: кэш + dynamic import / custom loaders."""
    name = (lang_name or "").strip()
    cached = _PARSER_CACHE.get(name)
    if cached is not None:
        return cached
    language = _load_language(name)
    parser = _make_parser(language)
    _PARSER_CACHE[name] = parser
    return parser


def _line_count(node) -> int:
    return int(node.end_point[0]) - int(node.start_point[0]) + 1


def _any_oversized_function(node) -> bool:
    ntype = getattr(node, "type", "") or ""
    if ntype in _FUNC_TYPES and _line_count(node) > MAX_FUNCTION_LINES:
        return True
    for child in getattr(node, "children", []) or []:
        if _any_oversized_function(child):
            return True
    return False


def _slice(source: bytes, node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _top_level_units(source: bytes, root) -> list[str]:
    units: list[str] = []
    for child in root.children:
        if child.type in ("comment", "line_comment", "block_comment"):
            text = _slice(source, child).strip()
            if text:
                units.append(text)
            continue
        if child.type in _CLASS_TYPES and _line_count(child) > TARGET_MAX_LINES:
            units.extend(_split_large_class(source, child))
            continue
        text = _slice(source, child).strip()
        if text:
            units.append(text)
    return units


def _split_large_class(source: bytes, class_node) -> list[str]:
    """Крупный класс: преамбула + методы отдельными единицами (тела не режем)."""
    methods = [
        ch
        for ch in class_node.children
        if ch.type in _FUNC_TYPES or ch.type in _CLASS_TYPES
    ]
    if not methods:
        text = _slice(source, class_node).strip()
        return [text] if text else []
    units: list[str] = []
    class_start = class_node.start_byte
    first_method = methods[0].start_byte
    preamble = source[class_start:first_method].decode("utf-8", errors="replace").strip()
    if preamble:
        units.append(preamble)
    for method in methods:
        body = _slice(source, method).strip()
        if body:
            units.append(body)
    tail = source[methods[-1].end_byte : class_node.end_byte].decode(
        "utf-8", errors="replace"
    ).strip()
    if tail:
        units.append(tail)
    return units


def _pack_units(units: list[str]) -> list[str]:
    from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
        estimate_text_tokens,
    )

    chunks: list[str] = []
    buf: list[str] = []
    buf_lines = 0
    buf_toks = 0

    def flush() -> None:
        nonlocal buf_lines, buf_toks
        if not buf:
            return
        chunks.append("\n\n".join(buf).strip())
        buf.clear()
        buf_lines = 0
        buf_toks = 0

    for unit in units:
        text = (unit or "").strip()
        if not text:
            continue
        lines = text.count("\n") + 1
        toks = estimate_text_tokens(text)
        if not buf:
            buf.append(text)
            buf_lines, buf_toks = lines, toks
            if buf_lines >= TARGET_MAX_LINES or buf_toks >= TARGET_MAX_TOKENS:
                flush()
            continue
        next_lines = buf_lines + lines
        next_toks = buf_toks + toks
        if next_lines <= TARGET_MAX_LINES and next_toks <= TARGET_MAX_TOKENS:
            buf.append(text)
            buf_lines, buf_toks = next_lines, next_toks
            continue
        flush()
        buf.append(text)
        buf_lines, buf_toks = lines, toks
        if buf_lines >= TARGET_MAX_LINES or buf_toks >= TARGET_MAX_TOKENS:
            flush()
    flush()
    return [c for c in chunks if c]


def _to_annotated(chunks: list[str], page_url: str) -> AnnotatedArticle:
    paragraph_map: dict[str, str] = {}
    lines_out: list[str] = []
    for i, body in enumerate(chunks, start=1):
        text = (body or "").strip()
        if len(text) < 2:
            continue
        pid = f"P_{i}"
        paragraph_map[pid] = text[:8000]
        lines_out.append(f"[{pid}]\n{text}")
    if not lines_out:
        raise AstChunkError("AST produced no chunks")
    return AnnotatedArticle(
        annotated_markdown="\n\n".join(lines_out).strip(),
        fig_map={},
        paragraph_map=paragraph_map,
        page_url=(page_url or "").strip(),
    )


class AstCodeChunker:
    """Стратегия ``CODE_PARSER_MODE=ast``: tree-sitter + упаковка мелких узлов."""

    def wrap(self, text: str, page_url: str = "") -> AnnotatedArticle:
        try:
            return self._wrap_ast(text, page_url)
        except Exception as exc:
            logger.warning(
                "AstCodeChunker fallback to linear (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return linear_chunk_code(text, page_url)

    def _wrap_ast(self, text: str, page_url: str = "") -> AnnotatedArticle:
        lang = language_from_url(page_url)
        if not lang:
            raise AstChunkError("no tree-sitter grammar for this source")
        parser = parser_for_language(lang)
        source = (text or "").replace("\r\n", "\n").encode("utf-8")
        if not source.strip():
            raise AstChunkError("empty source")
        tree = parser.parse(source)
        root = tree.root_node
        if getattr(root, "has_error", False):
            raise AstChunkError("tree-sitter ERROR node")
        if _any_oversized_function(root):
            raise AstChunkError(f"function longer than {MAX_FUNCTION_LINES} lines")
        units = _top_level_units(source, root)
        if not units:
            raise AstChunkError("no top-level AST units")
        packed = _pack_units(units)
        return _to_annotated(packed, page_url)
