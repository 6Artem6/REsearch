"""Depth-1 AST local-import expansion for GitHub ``/blob/`` ingest.

Uses ``ast`` (Python) and tree-sitter (C/C++/JS/TS). No regex over source.
GitHub I/O goes through ``GitHubTreeLoader`` only (one Trees API listing).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from knowledge_engine.services.article_ingestion.ast_code_chunker import (
    EXTENSION_TO_LANGUAGE,
    language_from_url,
)

logger = logging.getLogger(__name__)

MAX_LOCAL_DEPENDENCIES = 5
SUPPORTING_CONTEXT_MARK = "Supporting Context"

_PY_SOURCE_SUFFIXES = (".py", ".pyi", ".pyw")
_C_HEADER_SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx")
_JS_SOURCE_SUFFIXES = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")

_TS_QUOTE_INCLUDE_TYPES = frozenset({"string_literal", "string"})
_TS_SYSTEM_INCLUDE_TYPES = frozenset({"system_lib_string"})
_TS_IMPORT_PARENTS = frozenset(
    {"import_statement", "export_statement", "import_from", "export_from"}
)


def _posix(path: str) -> str:
    return (path or "").replace("\\", "/").strip("/")


def _normalize_posix(path: str) -> str:
    parts: list[str] = []
    for piece in _posix(path).split("/"):
        if piece in ("", "."):
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    return "/".join(parts)


def _parent_dir(path: str) -> str:
    norm = _posix(path)
    if "/" not in norm:
        return ""
    return norm.rsplit("/", 1)[0]


def _join_posix(*parts: str) -> str:
    bits = [_posix(p) for p in parts if _posix(p)]
    return _normalize_posix("/".join(bits))


def _unquote_literal(raw: str) -> str:
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        return text[1:-1]
    if len(text) >= 2 and text[0] == "<" and text[-1] == ">":
        return text[1:-1]
    return text


def _language_for(target_path: str, language: str) -> str:
    explicit = (language or "").strip().lower()
    if explicit:
        return explicit
    suffix = Path(_posix(target_path)).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix) or ""


def extract_local_imports(
    code: str,
    language: str,
    *,
    target_path: str = "",
) -> list[str]:
    """Return local import specs (relative modules / quoted headers / relative JS)."""
    return DependencyResolver().extract_local_imports(
        code, language, target_path=target_path
    )


def resolve_dependency_paths(
    tree_paths: list[str],
    target_path: str,
    local_imports: list[str],
    *,
    language: str = "",
    max_files: int = MAX_LOCAL_DEPENDENCIES,
) -> list[str]:
    return DependencyResolver(max_files=max_files).resolve_dependency_paths(
        tree_paths,
        target_path,
        local_imports,
        language=language,
    )


class DependencyResolver:
    """Depth-1 local dependency resolver (AST → Git Trees paths)."""

    def __init__(self, *, max_files: int = MAX_LOCAL_DEPENDENCIES) -> None:
        self.max_files = max(1, int(max_files))

    def extract_local_imports(
        self,
        code: str,
        language: str,
        *,
        target_path: str = "",
    ) -> list[str]:
        lang = _language_for(target_path, language)
        src = code or ""
        if not src.strip() or not lang:
            return []
        try:
            if lang == "python":
                return self._extract_python(src, target_path=target_path)
            if lang in ("c", "cpp"):
                return self._extract_c_family(src, lang)
            if lang in ("javascript", "typescript", "tsx"):
                return self._extract_js_family(src, lang)
        except Exception as exc:
            logger.warning(
                "DependencyResolver extract fallback (%s): %s: %s",
                lang,
                type(exc).__name__,
                exc,
            )
            return []
        logger.info(
            "DependencyResolver: no AST extractor for language=%s; skip",
            lang,
        )
        return []

    def resolve_dependency_paths(
        self,
        tree_paths: list[str],
        target_path: str,
        local_imports: list[str],
        *,
        language: str = "",
    ) -> list[str]:
        available = {_posix(p) for p in tree_paths if _posix(p)}
        target = _posix(target_path)
        lang = _language_for(target, language)
        out: list[str] = []
        seen: set[str] = set()
        for spec in local_imports:
            if len(out) >= self.max_files:
                break
            matched = False
            for candidate in self._candidates_for_spec(spec, target, lang):
                if candidate == target or candidate not in available:
                    continue
                if candidate in seen:
                    matched = True
                    break
                seen.add(candidate)
                out.append(candidate)
                matched = True
                break
            if matched:
                continue
            fallback = self._filename_in_tree(spec, target, available, lang)
            if fallback and fallback != target and fallback not in seen:
                seen.add(fallback)
                out.append(fallback)
        return out[: self.max_files]

    def _extract_python(self, code: str, *, target_path: str) -> list[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            logger.warning("DependencyResolver Python ast.parse failed: %s", exc)
            return []
        specs: list[str] = []
        seen: set[str] = set()
        ancestors = set(_posix(target_path).split("/")[:-1]) if target_path else set()

        def add(spec: str) -> None:
            item = (spec or "").strip()
            if not item or item in seen:
                return
            seen.add(item)
            specs.append(item)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                level = int(node.level or 0)
                module = (node.module or "").strip()
                if level > 0:
                    prefix = "." * level
                    if module:
                        add(f"{prefix}{module}")
                    else:
                        for alias in node.names or []:
                            name = (alias.name or "").strip()
                            if name and name != "*":
                                add(f"{prefix}{name}")
                    continue
                top = module.split(".", 1)[0] if module else ""
                if top and top in ancestors:
                    add(module)
                continue
            if isinstance(node, ast.Import):
                for alias in node.names or []:
                    name = (alias.name or "").strip()
                    top = name.split(".", 1)[0] if name else ""
                    if top and top in ancestors:
                        add(name)
        return specs

    def _extract_c_family(self, code: str, lang: str) -> list[str]:
        root, source = self._parse_tree_sitter(code, lang)
        if root is None:
            return []
        specs: list[str] = []
        seen: set[str] = set()
        for node in _walk_ts(root):
            ntype = getattr(node, "type", "") or ""
            if ntype != "preproc_include":
                continue
            local_path = ""
            system = False
            for child in getattr(node, "children", []) or []:
                ctype = getattr(child, "type", "") or ""
                raw = _ts_text(source, child)
                if ctype in _TS_SYSTEM_INCLUDE_TYPES:
                    system = True
                    break
                if ctype in _TS_QUOTE_INCLUDE_TYPES:
                    local_path = _unquote_literal(raw)
            if system or not local_path:
                continue
            if local_path.startswith("<") and local_path.endswith(">"):
                continue
            if local_path not in seen:
                seen.add(local_path)
                specs.append(local_path)
        return specs

    def _extract_js_family(self, code: str, lang: str) -> list[str]:
        root, source = self._parse_tree_sitter(code, lang)
        if root is None:
            return []
        specs: list[str] = []
        seen: set[str] = set()

        def add_if_relative(raw: str) -> None:
            path = _unquote_literal(raw)
            if not path.startswith("."):
                return
            if path not in seen:
                seen.add(path)
                specs.append(path)

        for node, parent in _walk_ts_with_parent(root):
            ntype = getattr(node, "type", "") or ""
            parent_type = getattr(parent, "type", "") or "" if parent is not None else ""
            if ntype in _TS_QUOTE_INCLUDE_TYPES and parent_type in _TS_IMPORT_PARENTS:
                add_if_relative(_ts_text(source, node))
                continue
            if ntype != "call_expression":
                continue
            children = list(getattr(node, "children", []) or [])
            fn = children[0] if children else None
            fn_type = getattr(fn, "type", "") or ""
            fn_text = _ts_text(source, fn) if fn is not None else ""
            if not (
                fn_type == "import"
                or (fn_type == "identifier" and fn_text == "require")
            ):
                continue
            for child, _parent in _walk_ts_with_parent(node):
                if (getattr(child, "type", "") or "") in _TS_QUOTE_INCLUDE_TYPES:
                    add_if_relative(_ts_text(source, child))
                    break
        return specs

    def _parse_tree_sitter(self, code: str, lang: str) -> tuple[Any, bytes] | tuple[None, bytes]:
        from knowledge_engine.services.article_ingestion.ast_code_chunker import (
            parser_for_language,
        )

        try:
            parser = parser_for_language(lang)
        except Exception as exc:
            logger.warning(
                "DependencyResolver: tree-sitter %s unavailable (%s: %s); skip",
                lang,
                type(exc).__name__,
                exc,
            )
            return None, b""
        source = (code or "").encode("utf-8")
        try:
            tree = parser.parse(source)
            root = getattr(tree, "root_node", None)
        except Exception as exc:
            logger.warning(
                "DependencyResolver: tree-sitter parse failed (%s): %s",
                lang,
                exc,
            )
            return None, source
        return root, source

    def _candidates_for_spec(
        self, spec: str, target_path: str, lang: str
    ) -> list[str]:
        raw = (spec or "").strip()
        if not raw:
            return []
        target_dir = _parent_dir(target_path)
        if lang == "python":
            return self._python_candidates(raw, target_dir, target_path)
        if lang in ("c", "cpp"):
            return self._c_candidates(raw, target_dir)
        if lang in ("javascript", "typescript", "tsx"):
            return self._js_candidates(raw, target_dir)
        joined = _join_posix(target_dir, raw)
        return [joined] if joined else []

    def _python_candidates(
        self, spec: str, target_dir: str, target_path: str
    ) -> list[str]:
        if spec.startswith("."):
            level = 0
            rest = spec
            while rest.startswith("."):
                level += 1
                rest = rest[1:]
            base = target_dir
            ups = max(0, level - 1)
            for _ in range(ups):
                base = _parent_dir(base)
            dotted = rest.replace("/", ".")
            return _python_files_for_module(base, dotted)
        # Same-package absolute: markupsafe.foo from src/markupsafe/x.py
        parts = spec.split(".")
        if not parts or not parts[0]:
            return []
        ancestors = _posix(target_path).split("/")[:-1]
        if parts[0] not in ancestors:
            return []
        idx = ancestors.index(parts[0])
        pkg_root = "/".join(ancestors[: idx + 1])
        remainder = ".".join(parts[1:])
        if remainder:
            return _python_files_for_module(pkg_root, remainder)
        return _python_files_for_module(_parent_dir(pkg_root), parts[0])

    def _c_candidates(self, spec: str, target_dir: str) -> list[str]:
        name = spec.strip().replace("\\", "/")
        out = [_join_posix(target_dir, name)]
        if not Path(name).suffix:
            out.extend(
                _join_posix(target_dir, name + suf) for suf in _C_HEADER_SUFFIXES
            )
        return _unique_keep_order(out)

    def _filename_in_tree(
        self,
        spec: str,
        target_path: str,
        available: set[str],
        lang: str,
    ) -> str | None:
        if lang not in ("c", "cpp"):
            return None
        needle = Path(spec.strip().replace("\\", "/")).name
        if not needle:
            return None
        matches = [
            p for p in available if p == needle or p.endswith("/" + needle)
        ]
        if not matches:
            return None
        tparts = _posix(target_path).split("/")

        def _prefix_score(path: str) -> int:
            pp = path.split("/")
            i = 0
            while i < min(len(tparts), len(pp)) and tparts[i] == pp[i]:
                i += 1
            return i

        matches.sort(key=_prefix_score, reverse=True)
        return matches[0]

    def _js_candidates(self, spec: str, target_dir: str) -> list[str]:
        relative = _join_posix(target_dir, spec)
        if not relative:
            return []
        out = [relative]
        if not Path(relative).suffix:
            out.extend(f"{relative}{suf}" for suf in _JS_SOURCE_SUFFIXES)
            out.extend(f"{relative}/index{suf}" for suf in _JS_SOURCE_SUFFIXES)
        seen: set[str] = set()
        uniq: list[str] = []
        for item in out:
            if item in seen:
                continue
            seen.add(item)
            uniq.append(item)
        return uniq


def _python_files_for_module(base_dir: str, dotted: str) -> list[str]:
    if not dotted:
        if not base_dir:
            return []
        return [f"{base_dir}/__init__.py", f"{base_dir}.py"]
    rel = dotted.replace(".", "/")
    joined = _join_posix(base_dir, rel) if base_dir else _posix(rel)
    return [
        f"{joined}.py",
        f"{joined}.pyi",
        f"{joined}/__init__.py",
    ]


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = _normalize_posix(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _walk_ts(node: Any):
    if node is None:
        return
    yield node
    for child in getattr(node, "children", []) or []:
        yield from _walk_ts(child)


def _walk_ts_with_parent(node: Any, parent: Any = None):
    if node is None:
        return
    yield node, parent
    for child in getattr(node, "children", []) or []:
        yield from _walk_ts_with_parent(child, node)


def _ts_text(source: bytes, node: Any) -> str:
    if node is None:
        return ""
    try:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


def format_blob_with_supporting_context(
    target_body: str,
    deps: list[tuple[str, str]],
) -> str:
    parts = [(target_body or "").rstrip()]
    for path, body in deps:
        text = (body or "").rstrip()
        if not path or not text:
            continue
        parts.append(f"[{SUPPORTING_CONTEXT_MARK}: {path}]\n{text}")
    return "\n\n".join(p for p in parts if p).strip() + "\n"


def maybe_fetch_github_blob_with_deps(url: str) -> tuple[str, str] | None:
    """Opt-in blob fetch + depth-1 local AST deps via GitHubTreeLoader.

    Returns ``(text, \"github_blob\")`` or ``None`` (caller keeps legacy raw fetch).
    """
    from knowledge_engine import config as ke_config
    from knowledge_engine.services.article_ingestion.github_tree_loader import (
        GitHubTreeLoader,
        parse_github_blob_url,
        path_is_skipped,
        suffix_is_allowed,
    )
    from knowledge_engine.ui.run_log import trace

    if not ke_config.USE_GITHUB_TREES_API:
        return None
    parsed = parse_github_blob_url(url)
    if parsed is None:
        return None
    owner, repo, ref, blob_path = parsed
    blob_path = _posix(blob_path)
    if path_is_skipped(blob_path) or not suffix_is_allowed(blob_path):
        return None

    loader = GitHubTreeLoader()
    try:
        tree_items = loader.fetch_tree(owner, repo, ref)
    except Exception as exc:
        logger.warning(
            "DependencyResolver: Trees API failed (%s: %s); blob fallback",
            type(exc).__name__,
            exc,
        )
        return None

    filtered = loader.filter_tree_items(tree_items)
    tree_paths = [_posix(str(item.get("path") or "")) for item in filtered]
    tree_paths = [p for p in tree_paths if p]
    if blob_path not in set(tree_paths):
        # Still try the target blob even if size filter dropped it? No — size cap.
        logger.info(
            "DependencyResolver: %s not in filtered tree (skip/size); blob fallback",
            blob_path,
        )
        return None

    target_body = loader.fetch_file_content(owner, repo, blob_path, ref)
    if not (target_body or "").strip():
        return None

    lang = language_from_url(blob_path) or language_from_url(url) or ""
    resolver = DependencyResolver(max_files=MAX_LOCAL_DEPENDENCIES)
    specs = resolver.extract_local_imports(
        target_body, lang, target_path=blob_path
    )
    resolved = resolver.resolve_dependency_paths(
        tree_paths, blob_path, specs, language=lang
    )
    deps: list[tuple[str, str]] = []
    for dep_path in resolved:
        body = loader.fetch_file_content(owner, repo, dep_path, ref)
        if not (body or "").strip():
            continue
        deps.append((dep_path, body))

    name = Path(blob_path).name
    trace(
        f"[DependencyResolver] Resolved {len(deps)} local AST dependencies for {name}"
    )
    combined = format_blob_with_supporting_context(target_body, deps)
    return combined, "github_blob"
