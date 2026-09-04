"""Three-tier AST + Flash Lite compression of source before Gemma MAP."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from knowledge_engine.services.article_ingestion.ast_code_chunker import (
    language_from_url,
)

logger = logging.getLogger(__name__)

CodeTier = Literal["HIGH", "MEDIUM", "LOW"]
"""Importance / RAG priority — what Flash Lite classifies from the catalog."""

RenderFormat = Literal["FULL", "HEADER_ONLY", "SKIP"]
"""How much text a function contributes to the MAP document. Orthogonal to
``CodeTier``: importance is a judgment about the function's role (Flash Lite's
call, from the catalog); render format is the assembler's decision about
output size, made from tier *plus* the function's own metadata (docstring,
length, whether a HIGH-tier function calls it) — never a static tier->format
table. See ``resolve_render_format`` (defined near ``assemble_tiered_context``,
after ``AstFunctionSpan`` exists)."""


_TS_FUNC_TYPES = frozenset(
    {
        "function_definition",
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "method_declaration",
        # Function-like macros (#define NAME(args) ...) — real callables with
        # a name/params/expansion; plain preproc_def (#define NAME value) is
        # a constant, not a function, and is deliberately excluded.
        "preproc_function_def",
    }
)
_TS_CALL_TYPES = frozenset({"call_expression", "call"})
_TS_COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment"})
_TS_BODY_TYPES = frozenset(
    {"compound_statement", "statement_block", "block", "body"}
)
_PRUNE_LANGS = frozenset({"python", "c", "cpp", "javascript", "typescript", "tsx"})

# C/C++ syntax keywords that must never reach the symbol catalog as a
# "function" name. Tree-sitter's error-recovery parsing can mis-tag a
# statement node as function_definition when it can't cleanly parse a nearby
# construct (observed on real CPython source: an `else if` block inside a
# malformed region got typed as function_definition) — no legitimate
# function/macro is ever literally named after one of these keywords in any
# of _PRUNE_LANGS, so filtering by name is a safe, language-agnostic guard
# regardless of which grammar mis-parsed.
_SYNTAX_KEYWORD_BLACKLIST = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "return",
        "goto",
        "struct",
        "union",
        "typedef",
        "enum",
    }
)


class TieredClassificationResult(BaseModel):
    """Flash Lite contract: partition AST function names into three tiers."""

    model_config = ConfigDict(extra="ignore")

    high_functions: list[str] = Field(
        default_factory=list,
        description=(
            "Names of HIGH functions: architecture, algorithms, entry points, "
            "system state. Gemma receives the full implementation."
        ),
    )
    # RU: ключевая логика — тело функции без сокращений.
    medium_functions: list[str] = Field(
        default_factory=list,
        description=(
            "Names of MEDIUM helpers/wrappers. Gemma receives signature, "
            "comments/docstrings, and HIGH call marks only — no body."
        ),
    )
    # RU: хелперы — сигнатура + комментарии + вызовы HIGH.
    low_functions: list[str] = Field(
        default_factory=list,
        description=(
            "Names of LOW utilities (alloc, logging, getters). "
            "Dropped entirely from the Gemma document."
        ),
    )
    # RU: тривиальные утилиты — полностью исключаются.

    @field_validator(
        "high_functions", "medium_functions", "low_functions", mode="before"
    )
    @classmethod
    def _names(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            raw = [v]
        elif isinstance(v, (list, tuple, set)):
            raw = list(v)
        else:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            name = str(item or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out


@dataclass
class AstFunctionSpan:
    name: str
    start_line: int
    end_line: int
    signature: str
    leading_comment: str = ""
    docstring: str = ""
    body: str = ""
    calls: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class ExternalCallLink:
    caller: str
    callee: str
    callee_file: str
    line: int


@dataclass
class ExternalCallGraph:
    target_functions: list[AstFunctionSpan]
    dep_signatures: list[tuple[str, str]]
    cross_calls: list[ExternalCallLink]


# STATIC PREFIX — this entire block is a byte-stable module constant reused
# verbatim on every classify_code_tiers_flash_lite() call, so it sits first
# in the Gemini request and only the per-call <function_catalog> user
# message (dynamic suffix) varies. That ordering is what makes the prefix
# eligible for the provider's implicit context-cache reuse across calls —
# nothing here should ever be interpolated with request-specific data.
#
# A hash-keyed numeric {hash: tier} wire format was evaluated (measured
# against the real Gemini API) as a way to shrink output tokens further.
# It's a dead end: Gemini's structured-output schema rejects dynamic-key
# objects outright (`additionalProperties is not supported in the Gemini
# API`), and the schema-legal substitute — a JSON array of {hash, tier}
# objects — measured ~34% SLOWER in practice (avg 13.3s vs 9.9s over 3
# trials each on pystate.c, 170 symbols): the per-entry "hash"/"tier" key
# labels outweigh the savings from short hashes vs. full names. Reverted;
# keep this name-based three-list format.
_TIER_SYSTEM = (
    "You are an expert static analysis assistant specializing in AST context "
    "pruning. You classify functions/macros from ONE source file for a "
    "knowledge-ingest MAP pass.\n"
    "You receive ONLY a <function_catalog> — name, signature, and leading "
    "comment for each symbol. Bodies are withheld by design (token budget); "
    "judge importance from name/signature/comment alone.\n\n"
    "Assign every catalog name to exactly one IMPORTANCE tier — this is RAG "
    "priority, a judgment about the symbol's architectural role. It does "
    "NOT by itself decide how much text a symbol contributes to the output "
    "document (that RenderFormat decision is a separate, downstream step "
    "that also weighs call-graph and docstring signals you don't see here).\n\n"
    "TIERS:\n"
    "- HIGH: kept in full downstream. Reserved EXCLUSIVELY for core domain "
    "orchestrators, state machines, and complex non-trivial algorithms.\n"
    "- MEDIUM: kept as signature-only downstream. Setup, allocation, "
    "dependency wiring, and helpers called by HIGH-tier code.\n"
    "- LOW: dropped downstream. Trivial accessors, state checks, boilerplate "
    "wrappers, and noise artifacts.\n\n"
    "SYMBOL CATEGORIES & NAME-PATTERN RULES:\n\n"
    "1. LIFECYCLE & MANAGEMENT (Assign: MEDIUM):\n"
    "   Category: setup, allocation, teardown, and initialization logic.\n"
    "   Name patterns: contains init, fini, new, free, alloc, delete, clear, "
    "reset, reinit, prealloc, bind, unbind.\n"
    "   Rule: signature matters for API surface; internal allocation detail "
    "rarely matters during high-level analysis.\n\n"
    "2. ACCESSORS & BOOLEAN CHECKS (Assign: LOW):\n"
    "   Category: single-statement getters, setters, state inspection, "
    "condition validators.\n"
    "   Name patterns: starts with or contains is_, has_, get_, set_, "
    "check_, can_.\n"
    "   Rule: self-documenting through their signature alone; drop entirely.\n\n"
    "3. UTILITY & HELPER WRAPPERS (Assign: MEDIUM, or LOW if trivial):\n"
    "   Category: repetitive internal macros, boilerplate list manipulation, "
    "single-pass utility bridges.\n"
    "   Name patterns: test_, ensure_, zap_, find_, lookup_.\n"
    "   Rule: MEDIUM by default; downgrade to LOW only if the "
    "name/signature/comment gives no sign of branching or business logic.\n\n"
    "4. CORE DOMAIN FLOW (Assign: HIGH):\n"
    "   Category: primary execution loops, key algorithms, coordination "
    "routines, complex business logic. Example names: take_gil, drop_gil, "
    "PyEval_SaveThread.\n"
    "   Rule: HIGH only if the symbol orchestrates multi-step domain "
    "behavior — not merely because it's long.\n\n"
    "Do not invent names outside the catalog. If a name matches no pattern "
    "above and its architectural role is unclear, use MEDIUM (never LOW for "
    "unknown role).\n"
    "Return TieredClassificationResult JSON only: high_functions, "
    "medium_functions, low_functions."
)

_CATALOG_COMMENT_MAX_CHARS = 250


def _catalog_entry(fn: AstFunctionSpan) -> str:
    """One <function_catalog> bullet — name/signature/comment, NEVER body."""
    lines = [f"* Name: {fn.name}"]
    sig = (fn.signature or fn.name).strip().replace("\n", " ")
    lines.append(f"  Signature: {sig}")
    comment = (fn.leading_comment or fn.docstring or "").strip().replace("\n", " ")
    if comment:
        if len(comment) > _CATALOG_COMMENT_MAX_CHARS:
            comment = comment[:_CATALOG_COMMENT_MAX_CHARS].rstrip() + "…"
        lines.append(f"  Comment: {comment}")
    return "\n".join(lines)


def _build_function_catalog(functions: list[AstFunctionSpan]) -> str:
    entries = "\n".join(_catalog_entry(fn) for fn in functions)
    return f"<function_catalog>\n{entries}\n</function_catalog>"


def _lang_from_path(path: str) -> str:
    raw = (path or "").strip()
    if not raw:
        return ""
    return language_from_url(raw) or language_from_url(f"https://x/{Path(raw).name}") or ""


def _unique_names(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = (item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _ts_text(source: bytes, node: Any) -> str:
    if node is None:
        return ""
    try:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _walk_ts(node: Any):
    if node is None:
        return
    yield node
    for child in getattr(node, "children", []) or []:
        yield from _walk_ts(child)


def _parse_ts(code: str, lang: str) -> tuple[Any, bytes] | tuple[None, bytes]:
    from knowledge_engine.services.article_ingestion.ast_code_chunker import (
        parser_for_language,
    )

    try:
        parser = parser_for_language(lang)
    except Exception as exc:
        logger.warning("tiered prune: parser %s unavailable (%s)", lang, exc)
        return None, b""
    source = (code or "").replace("\r\n", "\n").encode("utf-8")
    try:
        tree = parser.parse(source)
        return getattr(tree, "root_node", None), source
    except Exception as exc:
        logger.warning("tiered prune: parse failed (%s): %s", lang, exc)
        return None, source


def _declarator_identifier(source: bytes, node: Any) -> str:
    if node is None:
        return ""
    ntype = getattr(node, "type", "") or ""
    if ntype in ("identifier", "property_identifier", "field_identifier"):
        return _ts_text(source, node).strip()
    if ntype in (
        "function_declarator",
        "pointer_declarator",
        "parenthesized_declarator",
        "array_declarator",
        "qualified_identifier",
    ):
        for child in getattr(node, "children", []) or []:
            if (getattr(child, "type", "") or "") in ("parameter_list", "parameters"):
                continue
            name = _declarator_identifier(source, child)
            if name:
                return name
    for child in getattr(node, "children", []) or []:
        name = _declarator_identifier(source, child)
        if name:
            return name
    return ""


def _function_name_ts(source: bytes, node: Any) -> str:
    named = None
    if hasattr(node, "child_by_field_name"):
        named = node.child_by_field_name("name") or node.child_by_field_name("declarator")
    if named is not None:
        ident = _declarator_identifier(source, named)
        if ident:
            return ident
    return _declarator_identifier(source, node)


def _leading_comments_ts(source: bytes, node: Any) -> str:
    prev = getattr(node, "prev_named_sibling", None) or getattr(
        node, "prev_sibling", None
    )
    chunks: list[str] = []
    while prev is not None and (getattr(prev, "type", "") or "") in _TS_COMMENT_TYPES:
        chunks.append(_ts_text(source, prev).strip())
        prev = getattr(prev, "prev_named_sibling", None) or getattr(
            prev, "prev_sibling", None
        )
    chunks.reverse()
    return "\n".join(c for c in chunks if c)


def _signature_and_body_ts(source: bytes, node: Any) -> tuple[str, str]:
    full = _ts_text(source, node).strip()
    body_node = None
    if hasattr(node, "child_by_field_name"):
        body_node = node.child_by_field_name("body")
    if body_node is None:
        for child in getattr(node, "children", []) or []:
            if (getattr(child, "type", "") or "") in _TS_BODY_TYPES:
                body_node = child
                break
    if body_node is None:
        return full.rstrip(";"), full
    sig = source[node.start_byte : body_node.start_byte].decode(
        "utf-8", errors="replace"
    ).strip()
    return sig, full


def _calls_in_ts(source: bytes, node: Any) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for child in _walk_ts(node):
        if (getattr(child, "type", "") or "") not in _TS_CALL_TYPES:
            continue
        fn = None
        if hasattr(child, "child_by_field_name"):
            fn = child.child_by_field_name("function")
        if fn is None:
            kids = list(getattr(child, "children", []) or [])
            fn = kids[0] if kids else None
        name = _declarator_identifier(source, fn)
        if not name:
            continue
        line = int(getattr(child, "start_point", (0, 0))[0]) + 1
        key = (name, line)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, line))
    return out


def _span_from_ts(source: bytes, node: Any) -> AstFunctionSpan | None:
    name = _function_name_ts(source, node)
    if not name:
        return None
    sig, body = _signature_and_body_ts(source, node)
    start = int(getattr(node, "start_point", (0, 0))[0]) + 1
    end = int(getattr(node, "end_point", (0, 0))[0]) + 1
    return AstFunctionSpan(
        name=name,
        start_line=start,
        end_line=end,
        signature=sig or name,
        leading_comment=_leading_comments_ts(source, node),
        body=body,
        calls=_calls_in_ts(source, node),
    )


def _has_type(node: Any, wanted: str) -> bool:
    for child in _walk_ts(node):
        if (getattr(child, "type", "") or "") == wanted:
            return True
    return False


def _is_top_level_ts(node: Any) -> bool:
    parent = getattr(node, "parent", None)
    if parent is None:
        return True
    return (getattr(parent, "type", "") or "") in (
        "translation_unit",
        "program",
        "module",
        "source_file",
    )


def _extract_ts_functions(code: str, lang: str, *, bodies: bool) -> list[AstFunctionSpan]:
    root, source = _parse_ts(code, lang)
    if root is None:
        return []
    spans: list[AstFunctionSpan] = []
    for node in _walk_ts(root):
        ntype = getattr(node, "type", "") or ""
        is_proto = ntype == "declaration" and _is_top_level_ts(node) and _has_type(
            node, "function_declarator"
        )
        if ntype not in _TS_FUNC_TYPES and not is_proto:
            continue
        span = _span_from_ts(source, node)
        if span is None:
            continue
        if span.name in _SYNTAX_KEYWORD_BLACKLIST:
            logger.warning(
                "tiered prune: dropped syntax-keyword false positive "
                "name=%r type=%s line=%d (tree-sitter mis-parse)",
                span.name,
                ntype,
                span.start_line,
            )
            continue
        if not bodies:
            span.body = ""
            span.calls = []
        spans.append(span)
    return spans


def _call_name_py(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _extract_python_functions(code: str, *, bodies: bool) -> list[AstFunctionSpan]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    lines = (code or "").replace("\r\n", "\n").split("\n")
    spans: list[AstFunctionSpan] = []

    def _src(start: int, end: int) -> str:
        return "\n".join(lines[start - 1 : end]).strip()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            doc = ast.get_docstring(node) or ""
            sig_end = start
            if node.body:
                first = node.body[0]
                sig_end = max(start, int(getattr(first, "lineno", start)) - 1)
                if (
                    doc
                    and isinstance(first, ast.Expr)
                    and isinstance(getattr(first, "value", None), ast.Constant)
                ):
                    sig_end = int(getattr(first, "end_lineno", first.lineno) or first.lineno)
            calls: list[tuple[str, int]] = []
            if bodies:
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    name = _call_name_py(child.func)
                    if not name:
                        continue
                    line = int(getattr(child, "lineno", start) or start)
                    calls.append((name, line))
            spans.append(
                AstFunctionSpan(
                    name=node.name,
                    start_line=start,
                    end_line=end,
                    signature=_src(start, sig_end) or f"def {node.name}(...)",
                    docstring=doc,
                    body=_src(start, end) if bodies else "",
                    calls=calls,
                )
            )
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return spans


def extract_functions_from_source(
    code: str,
    lang: str,
    *,
    bodies: bool = True,
) -> list[AstFunctionSpan]:
    language = (lang or "").strip().lower()
    if language == "python":
        return _extract_python_functions(code, bodies=bodies)
    if language in _PRUNE_LANGS:
        return _extract_ts_functions(code, language, bodies=bodies)
    return []


def extract_ast_signatures_and_calls(
    file_path: str,
    dep_files: list[str] | None = None,
) -> ExternalCallGraph:
    """Target file: full AST spans. Deps: signatures only + cross-calls into target."""
    target_path = (file_path or "").strip()
    target_code = Path(target_path).read_text(encoding="utf-8", errors="replace")
    lang = _lang_from_path(target_path)
    target_fns = extract_functions_from_source(target_code, lang, bodies=True)

    dep_signatures: list[tuple[str, str]] = []
    dep_index: dict[str, str] = {}
    for dep_path in dep_files or []:
        path = (dep_path or "").strip()
        if not path or not Path(path).is_file():
            continue
        dep_lang = _lang_from_path(path) or lang
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for span in extract_functions_from_source(text, dep_lang, bodies=False):
            dep_signatures.append((path, span.signature or span.name))
            dep_index.setdefault(span.name, path)

    cross: list[ExternalCallLink] = []
    for fn in target_fns:
        for callee, line in fn.calls:
            dep_file = dep_index.get(callee)
            if not dep_file:
                continue
            cross.append(
                ExternalCallLink(
                    caller=fn.name,
                    callee=callee,
                    callee_file=dep_file,
                    line=line,
                )
            )
    return ExternalCallGraph(
        target_functions=target_fns,
        dep_signatures=dep_signatures,
        cross_calls=cross,
    )


def extract_ast_signatures_and_calls_from_text(
    target_code: str,
    *,
    target_path: str,
    dep_files: list[tuple[str, str]] | None = None,
) -> ExternalCallGraph:
    lang = _lang_from_path(target_path)
    target_fns = extract_functions_from_source(target_code, lang, bodies=True)
    dep_signatures: list[tuple[str, str]] = []
    dep_index: dict[str, str] = {}
    for path, text in dep_files or []:
        dep_lang = _lang_from_path(path) or lang
        for span in extract_functions_from_source(text, dep_lang, bodies=False):
            dep_signatures.append((path, span.signature or span.name))
            dep_index.setdefault(span.name, path)
    cross: list[ExternalCallLink] = []
    for fn in target_fns:
        for callee, line in fn.calls:
            dep_file = dep_index.get(callee)
            if not dep_file:
                continue
            cross.append(
                ExternalCallLink(
                    caller=fn.name,
                    callee=callee,
                    callee_file=dep_file,
                    line=line,
                )
            )
    return ExternalCallGraph(
        target_functions=target_fns,
        dep_signatures=dep_signatures,
        cross_calls=cross,
    )


def _normalize_classification(
    catalog: list[str],
    parsed: TieredClassificationResult,
) -> TieredClassificationResult:
    high = set(parsed.high_functions)
    medium = set(parsed.medium_functions)
    low = set(parsed.low_functions)
    out_h: list[str] = []
    out_m: list[str] = []
    out_l: list[str] = []
    for name in catalog:
        if name in high:
            out_h.append(name)
        elif name in low and name not in medium:
            out_l.append(name)
        else:
            out_m.append(name)
    return TieredClassificationResult(
        high_functions=out_h,
        medium_functions=out_m,
        low_functions=out_l,
    )


def classify_code_tiers_flash_lite(
    functions: list[AstFunctionSpan],
) -> TieredClassificationResult:
    """Classify importance from a signature+comment catalog — zero body overhead.

    ``functions`` may carry bodies (the caller needs them later for FULL
    rendering) but this call never puts them in the prompt: only
    name/signature/leading_comment reach Flash Lite (``_build_function_catalog``).
    """
    catalog = _unique_names([fn.name for fn in functions])
    if not catalog:
        return TieredClassificationResult()
    from knowledge_engine.services.gemini_stateless import (
        GeminiUnavailableError,
        is_gemini_available,
    )
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    if not is_gemini_available():
        return TieredClassificationResult(high_functions=list(catalog))
    user = (
        f"Function catalog ({len(catalog)} names — classify each one):\n\n"
        f"{_build_function_catalog(functions)}"
    )
    try:
        parsed = run_gemini_lite_structured(
            _TIER_SYSTEM,
            user,
            "tiered_code_prune",
            TieredClassificationResult,
            "tiered_code_prune",
        )
    except (GeminiUnavailableError, Exception) as exc:
        logger.warning("tiered prune: Flash Lite failed (%s); keep all HIGH", exc)
        return TieredClassificationResult(high_functions=list(catalog))
    return _normalize_classification(catalog, parsed)


def _comment_mark(lang: str, text: str) -> str:
    if lang == "python":
        return f"# {text}"
    return f"/* {text} */"


def _medium_stub(span: AstFunctionSpan, high: set[str], lang: str) -> str:
    parts: list[str] = []
    parts.append(span.signature.rstrip())
    if span.docstring and span.docstring not in (span.signature or ""):
        if lang == "python":
            doc = span.docstring.replace('"""', "")
            parts.append(f'    """{doc}"""')
        else:
            parts.append(f"/* {span.docstring} */")
    high_calls = [(n, ln) for n, ln in span.calls if n in high]
    if high_calls:
        marks = ", ".join(f"{n}() at line {ln}" for n, ln in high_calls)
        parts.append(_comment_mark(lang, f"Calls HIGH: {marks}"))
    if lang != "python":
        parts.append("{ /* body omitted */ }")
    else:
        parts.append("    ...")
    return "\n".join(parts).strip()


def _importance_map(classification: TieredClassificationResult) -> dict[str, CodeTier]:
    """name -> importance tier, from the (already mutually-exclusive) classification lists."""
    out: dict[str, CodeTier] = {}
    for name in classification.high_functions:
        out[name] = "HIGH"
    for name in classification.medium_functions:
        out[name] = "MEDIUM"
    for name in classification.low_functions:
        out[name] = "LOW"
    return out


_LONG_FUNCTION_LINE_THRESHOLD = 40


@dataclass
class RenderDecision:
    format: RenderFormat
    reason: str


def _names_called_by_high(
    functions: list[AstFunctionSpan], high: set[str]
) -> set[str]:
    """Reverse call edge: names that at least one HIGH-tier function calls into."""
    out: set[str] = set()
    for span in functions:
        if span.name not in high:
            continue
        for callee, _line in span.calls:
            out.add(callee)
    return out


def resolve_render_format(
    tier: CodeTier,
    span: AstFunctionSpan,
    *,
    called_by_high: bool = False,
    compressed: bool = False,
) -> RenderDecision:
    """Assembly-time policy — deliberately NOT a static tier->format table.

    Tier informs the decision but never dictates it alone: the same tier can
    resolve to a different format depending on the function's own metadata
    (docstring/comment presence, body length) and its place in the call graph.
    ``compressed`` is an optional stricter-budget switch a caller may set to
    push long, uncommented MEDIUM functions down to SKIP.
    """
    has_comment = bool((span.docstring or span.leading_comment or "").strip())
    line_count = max(1, span.end_line - span.start_line + 1)

    if tier == "HIGH":
        return RenderDecision("FULL", "high_priority_default")

    if tier == "MEDIUM":
        if called_by_high:
            return RenderDecision("HEADER_ONLY", "referenced_by_high_tier")
        if has_comment:
            return RenderDecision("HEADER_ONLY", "documented_medium")
        if compressed and line_count > _LONG_FUNCTION_LINE_THRESHOLD:
            return RenderDecision("SKIP", "long_uncommented_medium_compressed")
        return RenderDecision("HEADER_ONLY", "medium_default")

    # tier == "LOW"
    if has_comment:
        return RenderDecision("HEADER_ONLY", "low_with_documented_side_effects")
    return RenderDecision("SKIP", "trivial_getter")


def assemble_tiered_context(
    raw_code: str,
    classification: TieredClassificationResult,
    external_calls: ExternalCallGraph | None,
    *,
    lang: str = "c",
    compressed: bool = False,
) -> str:
    graph = external_calls or ExternalCallGraph([], [], [])
    by_name = {fn.name: fn for fn in graph.target_functions}
    if not by_name:
        by_name = {
            fn.name: fn for fn in extract_functions_from_source(raw_code, lang)
        }
    high = set(classification.high_functions)
    importance = _importance_map(classification)
    called_by_high = _names_called_by_high(list(by_name.values()), high)

    ext_lines: list[str] = ["## AST External Signatures & Cross-Calls"]
    if graph.dep_signatures:
        current = ""
        for path, sig in graph.dep_signatures:
            if path != current:
                ext_lines.append(f"### {Path(path).name}")
                current = path
            ext_lines.append(f"- `{sig}`")
    else:
        ext_lines.append("(no local AST dependency signatures)")
    if graph.cross_calls:
        ext_lines.append("### Cross-calls (target → dependency)")
        for link in graph.cross_calls:
            ext_lines.append(
                f"- `{link.caller}` → `{link.callee}()` at line {link.line} "
                f"({Path(link.callee_file).name})"
            )

    engine: list[str] = ["## High & Medium Priority Code Engine"]
    ordered = list(by_name.values())
    ordered.sort(key=lambda s: (s.start_line, s.name))
    emitted: set[str] = set()
    render_counts: dict[RenderFormat, int] = {"FULL": 0, "HEADER_ONLY": 0, "SKIP": 0}
    cursor = 0
    src_lines = (raw_code or "").replace("\r\n", "\n").split("\n")
    for span in ordered:
        gap = "\n".join(src_lines[cursor : span.start_line - 1]).strip()
        if gap:
            engine.append(gap)
        tier = importance.get(span.name, "MEDIUM")
        decision = resolve_render_format(
            tier,
            span,
            called_by_high=span.name in called_by_high,
            compressed=compressed,
        )
        render_counts[decision.format] += 1
        logger.debug(
            "Symbol: %s | Tier: %s -> Resolved RenderFormat: %s (reason: %s)",
            span.name,
            tier,
            decision.format,
            decision.reason,
        )
        if decision.format == "SKIP":
            cursor = max(cursor, span.end_line)
            continue
        if decision.format == "FULL":
            engine.append(span.body.strip() or span.signature)
        else:  # HEADER_ONLY
            engine.append(_medium_stub(span, high, lang))
        emitted.add(span.name)
        cursor = max(cursor, span.end_line)
    tail = "\n".join(src_lines[cursor:]).strip()
    if tail:
        engine.append(tail)
    if not emitted and not graph.target_functions:
        engine.append((raw_code or "").strip())

    from knowledge_engine.ui.run_log import trace

    trace(
        f"[Pipeline Audit] Phase: TieredCodePrune assemble | "
        f"Render FULL={render_counts['FULL']} "
        f"HEADER_ONLY={render_counts['HEADER_ONLY']} "
        f"SKIP={render_counts['SKIP']}"
    )
    return "\n\n".join(p for p in [*ext_lines, *engine] if p).strip() + "\n"


def split_combined_blob(text: str) -> tuple[str, list[tuple[str, str]]]:
    from knowledge_engine.ingest.dependency_resolver import SUPPORTING_CONTEXT_MARK

    raw = (text or "").replace("\r\n", "\n")
    needle = f"[{SUPPORTING_CONTEXT_MARK}: "
    idx = raw.find(needle)
    if idx < 0:
        return raw, []
    target = raw[:idx].rstrip()
    rest = raw[idx:]
    deps: list[tuple[str, str]] = []
    while rest.startswith(needle):
        rest = rest[len(needle) :]
        close = rest.find("]\n")
        if close < 0:
            break
        path = rest[:close].strip()
        rest = rest[close + 2 :]
        nxt = rest.find(needle)
        body = rest if nxt < 0 else rest[:nxt]
        if path and body.strip():
            deps.append((path, body.strip()))
        rest = "" if nxt < 0 else rest[nxt:]
    return target, deps


def maybe_prune_code_for_map(text: str, page_url: str = "") -> str:
    """Compress a code file for Gemma MAP. Fail-open to the original text."""
    lang = _lang_from_path(page_url)
    if lang not in _PRUNE_LANGS:
        return text
    from knowledge_engine.services.gemini_stateless import is_gemini_available

    if not is_gemini_available():
        return text
    target, deps = split_combined_blob(text)
    graph = extract_ast_signatures_and_calls_from_text(
        target, target_path=page_url, dep_files=deps
    )
    if not graph.target_functions:
        return text
    classification = classify_code_tiers_flash_lite(graph.target_functions)
    if not classification.high_functions and classification.low_functions:
        return text
    assembled = assemble_tiered_context(
        target, classification, graph, lang=lang
    )
    from knowledge_engine.ui.run_log import trace

    # Render FULL/HEADER_ONLY/SKIP counts are logged by assemble_tiered_context
    # itself (assembly-time policy decisions, not a static function of tier) —
    # this trace covers only the Flash Lite importance verdict.
    trace(
        f"[Pipeline Audit] Phase: TieredCodePrune | Target: {page_url or 'code'} | "
        f"Importance HIGH={len(classification.high_functions)} "
        f"MEDIUM={len(classification.medium_functions)} "
        f"LOW={len(classification.low_functions)} | "
        f"Final Chars: {len(assembled)}"
    )
    return assembled
