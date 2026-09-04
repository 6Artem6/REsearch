"""Pre-Flight Triage: локальный (без LLM, без лишних HTTP) двухфазный отбор
Exa/practical кандидатов до тяжёлого MAP+REDUCE инджеста, с гарантированным
in-memory добором до CURRICULUM_PREFLIGHT_FINAL_ARTICLES.

Phase 1 (stage1_zero_http_gate) — один Exa fetch + один cross-encoder проход
    (bge-reranker-v2-m3) по ВСЕМ highlights → RAM-очередь кандидатов по
    убыванию релевантности; здесь никто не отбрасывается.
Phase 2 (run_pre_flight_triage, In-Memory Replenishment Loop) — батчами (по
    CURRICULUM_PREFLIGHT_STAGE2_TOP_K) из этой очереди:
      Stage 2  Parallel Fetch — httpx GET + Trafilatura, абзацы короче
               CURRICULUM_PREFLIGHT_PARAGRAPH_MIN_CHARS отбрасываются.
               Code Preservation Policy: источники исходного кода обходят
               Trafilatura целиком — построчный min_chars-фильтр убивал бы
               почти весь короткострочный код. Детекция кода — см.
               detect_code_content(): HTML-теги ненадёжны на GitHub (raw.
               githubusercontent.com отдаёт голый plain text без разметки;
               веб-UI GitHub рендерит blob через textarea/React-таблицы,
               которые Trafilatura разбирает в бесструктурный текст) —
               детекция трёхслойная (URL/домен → bge-m3 вектор → Tree-Sitter
               AST), а не привязка к <pre>/<code> или текстовым регуляркам.
               Embedded-код в обычных HTML-статьях детектируется одним
               батч-вызовом embed() на все <code>-теги страницы (не по
               одному на тег). AST Collapsing (_code_paragraphs_for_embedding):
               для чистых code-источников на BGE идут не сырые построчные
               куски, а Tree-Sitter-сигнатуры/докстринги/верхние комментарии
               + вызовы функций (depth<=2) — тела функций сворачиваются;
               реальный сырой текст всё равно уходит в MAP+REDUCE инджест
               нетронутым через Zero-Waste Handover.
      Stage 3  MMR Diversity  — BGE-M3 эмбеддинги абзацев + greedy MMR.
      Stage 4  Quality Gate   — bge-reranker-v2-m3 по MMR-абзацам + keyword
               coverage → Triage Score = 0.5*Peak + 0.3*Mean_Top3 +
               0.2*KeywordCoverage; Hard Gate отсеивает score < threshold.
               Code-источники ИММУННЫ к этому NL cross-encoder скору (он
               системно топит исходный код на нетематическом языке против
               темы) — окончательную оценку код получает позже, на этапе
               Flash Lite ingest gate (document_triage_engine/paper_structure_analyzer).
    Если очередной батч не набрал нужное число source — берётся следующий
    батч ИЗ ТОЙ ЖЕ очереди, без повторных обращений к Exa/Flash Lite.
Phase 3 (safety fallback) — если очередь исчерпана, а source всё ещё меньше
    цели: (a) отклонённые whitelist-домены допускаются по сниженному порогу
    CURRICULUM_PREFLIGHT_WHITELIST_HARD_GATE_THRESHOLD, затем (b) добор
    лучшими по Triage Score из оставшихся отклонённых (Best-of-Rest).
    Stage 2 отказы (anti-bot/thin body/no paragraphs) тоже попадают в пул
    отказов на этом этапе — иначе Phase 3 не из чего добирать, если именно
    Stage 2 срезал кандидатов до кворума.

Zero-Waste Handover — выжившие несут уже скачанный HTML дальше в инджест
через pop_preflight_html() (см. source_material_pipeline._ingest_blog_url_precheck),
без повторного fetch.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
from urllib.parse import urlparse

from knowledge_engine.config import (
    CURRICULUM_PREFLIGHT_CODE_MIN_CHARS,
    CURRICULUM_PREFLIGHT_CODE_VECTOR_SIM_THRESHOLD,
    CURRICULUM_PREFLIGHT_FETCH_CONCURRENCY,
    CURRICULUM_PREFLIGHT_FETCH_TIMEOUT_SEC,
    CURRICULUM_PREFLIGHT_FINAL_ARTICLES,
    CURRICULUM_PREFLIGHT_HARD_GATE_THRESHOLD,
    CURRICULUM_PREFLIGHT_MMR_LAMBDA,
    CURRICULUM_PREFLIGHT_MMR_TOP_K,
    CURRICULUM_PREFLIGHT_PARAGRAPH_MIN_CHARS,
    CURRICULUM_PREFLIGHT_STAGE2_TOP_K,
    CURRICULUM_PREFLIGHT_WHITELIST_HARD_GATE_THRESHOLD,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.ui.run_log import trace

# github_blob/github_trees/github_zip (см. web_extract.smart_fetch_page_html)
# отдают сырой исходный текст как есть — НЕ html. Прогонять через
# Trafilatura — категориальная ошибка, а не вопрос качества. Держим как
# дополнительный zero-cost сигнал Layer 1 наряду с собственной проверкой
# domain/extension у detect_code_content() (бесплатно — fetcher уже это вычислил).
_CODE_FETCH_METHODS = frozenset({"github_blob", "github_trees", "github_zip"})

# Layer 1 — известные хосты с сырым кодом. raw.githubusercontent.com/gist
# отдают чистый текст БЕЗ html вообще; github.com/*/blob/* рендерит файл
# через textarea/React-таблицу, которую Trafilatura превращает в
# бесструктурный текст — в обоих случаях нет <pre>/<code> разметки, за
# которую можно зацепиться, так что домен — единственный надёжный
# zero-cost сигнал.
_CODE_DOMAINS = frozenset({"raw.githubusercontent.com", "gist.github.com"})
# Расширения, для которых у Tree-Sitter в проекте нет грамматики, но которые
# всё равно однозначно сигналят "это код" по URL для иммунитета Layer 1.
_EXTRA_CODE_EXTENSIONS = (".cu", ".asm", ".s", ".sh", ".m", ".scala", ".lua")

_CODE_ANCHOR_TEXT = (
    "Source code. Programming language implementation in C, Python, Rust, "
    "Go or JavaScript — function definitions, variable declarations, "
    "control flow syntax, braces and semicolons."
)
_PROSE_ANCHOR_TEXT = (
    "Prose article text. Natural language explanation and narrative "
    "written for human readers, in full sentences."
)
# Грамматики Tree-Sitter, которые в проекте установлены всегда — используются
# как общие пробы Layer 3, когда URL не даёт языковой подсказки (Layer 2
# сработал, но неизвестно, на каком языке валидировать).
_AST_PROBE_LANGUAGES = ("python", "c", "javascript")


def _is_code_domain_or_extension(url: str) -> bool:
    """Layer 1 (zero-cost gate): known code hosts + URL/extension match."""
    from knowledge_engine.services.article_ingestion.ast_code_chunker import (
        EXTENSION_TO_LANGUAGE,
    )

    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower()
    if host in _CODE_DOMAINS or host.endswith(".githubusercontent.com"):
        return True
    path = parsed.path.lower()
    if host == "github.com" and "/blob/" in path:
        return True
    if any(path.endswith(ext) for ext in EXTENSION_TO_LANGUAGE):
        return True
    return any(path.endswith(ext) for ext in _EXTRA_CODE_EXTENSIONS)


def _vector_classify_code_batch(texts: list[str]) -> list[bool]:
    """Layer 2, batched: ONE bge-m3 embed() call scores every candidate
    snippet against the code/prose anchors, instead of one embed() call per
    candidate (the N+1 pattern _extract_embedded_code_blocks used to hit —
    59 embed() calls for 59 <code> tags on a single docs.python.org page)."""
    samples = [(t or "").strip()[:2000] for t in texts]
    idx_nonempty = [i for i, s in enumerate(samples) if s]
    if not idx_nonempty:
        return [False] * len(texts)
    from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

    vecs = embed_texts_bge_m3(
        [_CODE_ANCHOR_TEXT, _PROSE_ANCHOR_TEXT] + [samples[i] for i in idx_nonempty]
    )
    code_vec, prose_vec, *sample_vecs = vecs
    results = [False] * len(texts)
    for pos, i in enumerate(idx_nonempty):
        code_sim = _cosine(sample_vecs[pos], code_vec)
        prose_sim = _cosine(sample_vecs[pos], prose_vec)
        results[i] = (
            code_sim > prose_sim
            and code_sim >= CURRICULUM_PREFLIGHT_CODE_VECTOR_SIM_THRESHOLD
        )
    trace(
        f"PRE_FLIGHT code_detect vector batch | n={len(idx_nonempty)} "
        f"positive={sum(results)} threshold={CURRICULUM_PREFLIGHT_CODE_VECTOR_SIM_THRESHOLD:.2f}"
    )
    return results


def _vector_classify_code(text: str) -> bool:
    """Layer 2: bge-m3 embedding of the snippet vs code/prose anchor
    vectors — used when the URL alone (Layer 1) is inconclusive."""
    return _vector_classify_code_batch([text])[0]


def _ast_validate_code(text: str, url: str) -> bool:
    """Layer 3: only a syntactically well-formed Tree-Sitter tree counts as
    final confirmation for a Layer-2-only (vector) signal — final structural
    judgement stays with Tree-Sitter/Flash Lite, not duplicated here."""
    sample = (text or "")[:4000].strip()
    if not sample:
        return False
    from knowledge_engine.services.article_ingestion.ast_code_chunker import (
        EXTENSION_TO_LANGUAGE,
        AstChunkError,
        parser_for_language,
    )

    path = urlparse(url or "").path.lower()
    candidates = [
        lang for ext, lang in EXTENSION_TO_LANGUAGE.items() if path.endswith(ext)
    ]
    for lang in [*dict.fromkeys(candidates), *_AST_PROBE_LANGUAGES]:
        try:
            parser = parser_for_language(lang)
        except AstChunkError:
            continue
        except Exception:
            continue
        try:
            tree = parser.parse(sample.encode("utf-8", errors="replace"))
        except Exception:
            continue
        root = tree.root_node
        if not getattr(root, "has_error", True) and root.child_count > 0:
            trace(f"PRE_FLIGHT code_detect ast ✓ | lang={lang}")
            return True
    return False


def detect_code_content(url: str, text: str, method: str = "") -> bool:
    """3-layer source-code detector, in ascending cost order.

    Layer 1 (zero-cost): fetch method + URL domain/extension. Covers the two
    cases HTML-tag-based detection structurally cannot: raw.githubusercontent
    .com / gist raw text (no html at all) and github.com/*/blob/* pages
    (textarea/React table that Trafilatura reduces to unstructured text).
    Layer 2 (bge-m3 vector classify): only runs when Layer 1 is inconclusive
    — snippet embedding vs code/prose anchor vectors.
    Layer 3 (Tree-Sitter AST): only runs when Layer 2 flagged code_snippet —
    a real, error-free syntax tree is the final confirmation, so a stray
    Layer-2 false positive (e.g. a terse changelog) cannot grant immunity on
    its own.
    """
    if method in _CODE_FETCH_METHODS:
        return True
    if _is_code_domain_or_extension(url):
        return True
    if not _vector_classify_code(text):
        return False
    return _ast_validate_code(text, url)


def _code_paragraphs_from_raw_text(text: str, *, min_chars: int) -> list[str]:
    """Blank-line-block split for raw source (github passthrough). Never
    empty for non-empty input — unlike Trafilatura's per-line split, which
    treats every short code line as UI-crumb trash."""
    body = (text or "").strip()
    if not body:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    if not blocks:
        blocks = [body]
    out: list[str] = []
    for block in blocks:
        if len(block) <= 1200:
            if len(block) >= min_chars:
                out.append(block)
            continue
        for i in range(0, len(block), 1000):
            chunk = block[i : i + 1000].strip()
            if len(chunk) >= min_chars:
                out.append(chunk)
    if out:
        return out
    return [body[:2000]] if len(body) >= min_chars else []


# ---------------------------------------------------------------------------
# AST Collapsing для BGE: большой сырой файл исходников (например,
# ceval_gil.c, 107KB), разрезанный _code_paragraphs_from_raw_text выше по
# блокам пустых строк, даёт сотни крошечных абзацев (448 для ceval_gil.c в
# прогоне perf-аудита) — один гигантский batch-вызов embed() (17.4с) только
# для Stage 3 MMR. Проходу BGE MMR нужен лишь сигнал *релевантности*, а не
# полные тела функций: сигнатуры, ведущие комментарии, docstring'и и
# неглубокий (depth<=2) список имён вызовов дают тот же ранжирующий сигнал
# с ~20-30 юнитов вместо сотен. ТЕЛА функций/структур здесь сворачиваются
# (опускаются) — это меняет только то, что эмбеддится для *скоринга*
# Stage 3/4; нетронутый сырой текст — это по-прежнему то, что уходит в
# stash_preflight_html() для реального MAP+REDUCE-инджеста.
# ---------------------------------------------------------------------------
_SEMANTIC_FUNC_TYPES = frozenset(
    {
        "function_definition",
        "function_declaration",
        "function_item",  # rust
        "method_definition",
        "generator_function_declaration",
        "decorated_definition",
    }
)
_SEMANTIC_CLASS_TYPES = frozenset(
    {
        "struct_specifier",
        "class_specifier",  # C++ (tree-sitter-c/cpp использует это, не class_definition)
        "class_definition",
        "class_declaration",
        "class_item",
        "struct_item",
        "type_definition",
    }
)
_SEMANTIC_BODY_TYPES = frozenset(
    {
        "block",
        "compound_statement",
        "function_body",
        "suite",
        "statement_block",
        "field_declaration_list",  # C++ class/struct body
    }
)
_SEMANTIC_CALL_TYPES = frozenset({"call_expression", "call"})
_SEMANTIC_COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment"})
_SEMANTIC_STRING_TYPES = frozenset({"string", "string_literal"})
_SEMANTIC_CALL_DEPTH = 2
_SEMANTIC_MAX_CALLS_PER_UNIT = 12


def _semantic_node_text(source: bytes, node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _semantic_find_body(func_node):
    for child in reversed(func_node.children):
        if child.type in _SEMANTIC_BODY_TYPES:
            return child
    return None


def _semantic_collect_calls(source: bytes, node, *, depth: int) -> list[str]:
    """Depth counts BLOCK nesting (while/if/for bodies), not raw AST node
    depth — a single statement like `x = f();` is 3-4 AST levels deep
    (expression_statement -> assignment_expression -> call_expression)
    before it ever reaches a second block, so counting every wrapper node
    would exhaust the budget on line one and never see anything nested."""
    if node is None or depth > _SEMANTIC_CALL_DEPTH:
        return []
    out: list[str] = []
    for child in node.children:
        if child.type in _SEMANTIC_CALL_TYPES and child.children:
            name = _semantic_node_text(source, child.children[0]).strip()
            if name and len(name) < 80:
                out.append(name)
        next_depth = depth + 1 if child.type in _SEMANTIC_BODY_TYPES else depth
        out.extend(_semantic_collect_calls(source, child, depth=next_depth))
    return out


def _semantic_docstring(source: bytes, body) -> str:
    if body is None or not body.children:
        return ""
    stmt = body.children[0]
    target = stmt
    if stmt.type == "expression_statement" and stmt.children:
        target = stmt.children[0]
    if target.type in _SEMANTIC_STRING_TYPES:
        return _semantic_node_text(source, target).strip()[:400]
    return ""


def _semantic_signature_unit(source: bytes, node) -> str | None:
    body = _semantic_find_body(node)
    if body is not None:
        sig = (
            source[node.start_byte : body.start_byte]
            .decode("utf-8", errors="replace")
            .strip()
        )
    else:
        sig = _semantic_node_text(source, node).strip()
    if not sig:
        return None
    piece = sig
    doc = _semantic_docstring(source, body)
    if doc:
        piece += "\n" + doc
    calls = _semantic_collect_calls(source, body, depth=0)
    if calls:
        uniq_calls = list(dict.fromkeys(calls))[:_SEMANTIC_MAX_CALLS_PER_UNIT]
        piece += "\ncalls: " + ", ".join(uniq_calls)
    return piece


_TRIVIAL_ACCESSOR_NAME_RE = re.compile(r"\b(get|set|is|has)[_A-Z][A-Za-z0-9_]*\s*\(")


def _looks_like_trivial_accessor(piece: str) -> bool:
    """HIGH-priority filter for CODE: a get_/set_/is_/has_-named function
    with no collected calls (nothing interesting happens in the body) reads
    as boilerplate — skip it so the AST extract stays signal, not noise.
    A real call inside the body (calls: line present) always keeps it."""
    if "\ncalls: " in piece:
        return False
    return bool(_TRIVIAL_ACCESSOR_NAME_RE.search(piece))


def _walk_semantic_extracts(source: bytes, root, *, min_chars: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for node in root.children:
        ntype = node.type
        if ntype in _SEMANTIC_COMMENT_TYPES:
            text = _semantic_node_text(source, node).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text[:800])
            continue
        if ntype in _SEMANTIC_FUNC_TYPES or ntype in _SEMANTIC_CLASS_TYPES:
            piece = _semantic_signature_unit(source, node)
            if not piece or piece in seen:
                continue
            if ntype in _SEMANTIC_FUNC_TYPES and _looks_like_trivial_accessor(piece):
                continue
            seen.add(piece)
            if len(piece) >= min_chars:
                out.append(piece)
    return out


def _walk_semantic_extracts_with_nodes(
    source: bytes, root, *, min_chars: int
) -> list[tuple[object, str]]:
    """Same top-level walk and filtering as _walk_semantic_extracts, but
    keeps the (node, piece) pairing instead of discarding the node — needed
    to re-locate a CORE-classified unit's AST node after Triage."""
    out: list[tuple[object, str]] = []
    seen: set[str] = set()
    for node in root.children:
        ntype = node.type
        if ntype in _SEMANTIC_COMMENT_TYPES:
            continue
        if ntype in _SEMANTIC_FUNC_TYPES or ntype in _SEMANTIC_CLASS_TYPES:
            piece = _semantic_signature_unit(source, node)
            if not piece or piece in seen:
                continue
            if ntype in _SEMANTIC_FUNC_TYPES and _looks_like_trivial_accessor(piece):
                continue
            seen.add(piece)
            if len(piece) >= min_chars:
                out.append((node, piece))
    return out


def _semantic_function_nodes(node) -> list:
    """Recursively find every function/method definition node at any
    nesting depth (top-level functions AND methods inside classes), in
    source order. Stops descending once a function-type node is matched —
    does not chase nested closures/lambdas inside a function body, and
    does not double-count a decorated_definition's own function_definition
    child."""
    found: list = []
    for child in node.children:
        if child.type in _SEMANTIC_FUNC_TYPES:
            found.append(child)
            continue
        found.extend(_semantic_function_nodes(child))
    return found


_SEMANTIC_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


def _semantic_function_display_name(sig: str) -> str:
    m = _SEMANTIC_NAME_RE.search(sig)
    return m.group(1) if m else ""


def _bulk_gate_code_context(
    text: str, url: str, core_terse_units: list[str], *, min_chars: int
) -> list[str]:
    """Heuristic Top-6 CORE-function/method selection for the Bulk Gate
    prompt (Head-3 + Tail-3 by file order when more than 6 survive Triage,
    otherwise all of them) with FULL verbatim bodies — every comment,
    docstring, and type annotation kept intact, nothing stripped — instead
    of the terse signature-only units Triage classifies on. A CORE-
    classified class/struct contributes ALL of its methods (Triage already
    approved the whole class, so every method inherits CORE status) rather
    than the class as one opaque unit. Appends one same-file call-tree line
    per CORE function/method (e.g. "union_sets -> calls: find_set"),
    covering every CORE unit even if the Head/Tail cut left it out of the
    selected bodies, to compensate for skipped mid-file utilities.
    Fail-open: returns core_terse_units unchanged if re-parsing fails or
    nothing matches — the caller never ends up with less than it started
    with."""
    if not core_terse_units:
        return []
    sample = (text or "").strip()
    if not sample:
        return list(core_terse_units)

    from knowledge_engine.services.article_ingestion.ast_code_chunker import (
        EXTENSION_TO_LANGUAGE,
        AstChunkError,
        parser_for_language,
    )

    path = urlparse(url or "").path.lower()
    candidates = [
        lang for ext, lang in EXTENSION_TO_LANGUAGE.items() if path.endswith(ext)
    ]
    source = sample.encode("utf-8", errors="replace")
    core_set = set(core_terse_units)
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

        top_level = _walk_semantic_extracts_with_nodes(source, root, min_chars=min_chars)
        core_nodes: list = []
        for node, piece in top_level:
            if piece not in core_set:
                continue
            if node.type in _SEMANTIC_CLASS_TYPES:
                core_nodes.extend(_semantic_function_nodes(node))
            else:
                core_nodes.append(node)
        if not core_nodes:
            continue
        core_nodes.sort(key=lambda n: n.start_byte)

        call_lines: list[str] = []
        for node in core_nodes:
            body = _semantic_find_body(node)
            sig_bytes = (
                source[node.start_byte : body.start_byte]
                if body is not None
                else source[node.start_byte : node.end_byte]
            )
            sig = sig_bytes.decode("utf-8", errors="replace").strip()
            name = _semantic_function_display_name(sig) or f"fn@{node.start_byte}"
            calls = _semantic_collect_calls(source, body, depth=0)
            uniq_calls = list(dict.fromkeys(calls))[:_SEMANTIC_MAX_CALLS_PER_UNIT]
            call_lines.append(
                f"{name} -> calls: {', '.join(uniq_calls)}"
                if uniq_calls
                else f"{name} -> calls: (none)"
            )

        selected = (
            core_nodes[:3] + core_nodes[-3:] if len(core_nodes) > 6 else core_nodes
        )
        bodies = [_semantic_node_text(source, node) for node in selected]
        call_tree = "CALL TREE (CORE-функции файла):\n" + "\n".join(call_lines)
        trace(
            f"PRE_MAP_DEDUP bulk_gate_code_context ✓ | lang={lang} "
            f"core_funcs={len(core_nodes)} selected={len(selected)} "
            f"raw_chars={len(sample)}"
        )
        return bodies + [call_tree]
    return list(core_terse_units)


def _ast_semantic_extracts(text: str, url: str, *, min_chars: int) -> list[str]:
    """Tree-Sitter AST Collapsing: signatures + docstrings + top comments +
    shallow (depth<=2) call names, function/struct bodies omitted. Returns
    [] (triggering the raw-chunk fallback) whenever no installed grammar
    parses the source cleanly — never a partial/best-effort tree."""
    sample = (text or "").strip()
    if not sample:
        return []
    from knowledge_engine.services.article_ingestion.ast_code_chunker import (
        EXTENSION_TO_LANGUAGE,
        AstChunkError,
        parser_for_language,
    )

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
            # NB: реальный C с тяжёлой условной компиляцией #if/#ifdef/#else
            # (например, ceval_gil.c из самого CPython) регулярно на этом
            # спотыкается — грамматические парсеры не гоняют препроцессор, а
            # асимметричные фигурные скобки между ветками без него в принципе
            # не разрешить. Падает в raw-chunk fallback ниже; это корректный,
            # безопасный исход, а не баг, который надо обходить.
            continue
        extracts = _walk_semantic_extracts(source, root, min_chars=min_chars)
        if extracts:
            trace(
                f"PRE_FLIGHT ast_collapse ✓ | lang={lang} extracts={len(extracts)} "
                f"raw_chars={len(sample)}"
            )
            return extracts
    return []


def _code_paragraphs_for_embedding(text: str, url: str, *, min_chars: int) -> list[str]:
    """AST Collapsing first (see _ast_semantic_extracts) — falls back to the
    raw blank-line-block split (_code_paragraphs_from_raw_text) whenever
    Tree-Sitter can't produce a clean, non-empty result (no grammar, ERROR
    node, or a file with no top-level func/struct/comment nodes at all)."""
    extracts = _ast_semantic_extracts(text, url, min_chars=min_chars)
    if extracts:
        return extracts
    return _code_paragraphs_from_raw_text(text, min_chars=min_chars)


def _extract_embedded_code_blocks(html: str, url: str, *, min_chars: int) -> list[str]:
    """<pre>/<code> blocks kept as whole units — a code sample embedded in
    an otherwise normal article must not be shredded by the prose paragraph
    min-chars filter line by line. Standalone (non-<pre>) <code> spans are
    collected first and classified in ONE batched Layer-2 embed() call
    (see _vector_classify_code_batch), not one call per tag; each Layer-2
    survivor is then confirmed individually via Layer 3 (Tree-Sitter — no
    embed cost). Layer 1 (detect_code_content's domain/method check) is not
    re-run per tag: this function only runs once the whole-page URL/method
    already failed Layer 1, so it would be a no-op recheck."""
    if not (html or "").strip():
        return []
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    out: list[str] = []
    seen: set[int] = set()
    for tag in soup.find_all("pre"):
        text = tag.get_text("\n").strip()
        if len(text) >= min_chars:
            out.append(text)
        seen.add(id(tag))

    candidates: list[str] = []
    for tag in soup.find_all("code"):
        if any(id(parent) in seen for parent in tag.parents):
            continue
        text = tag.get_text("\n").strip()
        if len(text) >= min_chars:
            candidates.append(text)
    if candidates:
        flags = _vector_classify_code_batch(candidates)
        for text, is_code in zip(candidates, flags):
            if is_code and _ast_validate_code(text, url):
                out.append(text)
    return out


# ---------------------------------------------------------------------------
# Мост Zero-Waste Handover. Stage 2 уже скачивает полное тело страницы для
# каждого выжившего. Инджест (source_material_pipeline._ingest_blog_url_precheck)
# иначе снова вызвал бы smart_fetch_page_html() для ТОГО ЖЕ url — этот
# pop-on-read кэш даёт переиспользовать уже готовый параметр `raw_html`.
# Ограничен по конструкции: хранит тела только для хитов, находящихся в
# полёте через triage прямо сейчас, снимается ровно один раз стороной инджеста.
# ---------------------------------------------------------------------------
_html_lock = threading.Lock()
_html_cache: dict[str, str] = {}


def stash_preflight_html(url: str, html: str) -> None:
    key = (url or "").strip()
    if not key or not html:
        return
    with _html_lock:
        _html_cache[key] = html


def pop_preflight_html(url: str) -> str | None:
    key = (url or "").strip()
    if not key:
        return None
    with _html_lock:
        return _html_cache.pop(key, None)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def greedy_mmr_select(
    vectors: list[list[float]],
    scores: list[float],
    *,
    top_k: int = 6,
    lambda_param: float = 0.65,
) -> list[int]:
    """Standard greedy MMR — pure function, no RAG/LanceDB coupling.

    final(i) = lambda*relevance(i) - (1-lambda)*max_sim(i, selected)
    Returns indices into vectors/scores, in selection order (best first).
    """
    n = len(vectors)
    if n == 0 or top_k <= 0:
        return []
    if len(scores) != n:
        raise ValueError("vectors and scores must be the same length")
    k = min(top_k, n)
    remaining = set(range(n))
    selected: list[int] = []
    while len(selected) < k and remaining:
        best_i = -1
        best_val = float("-inf")
        for i in remaining:
            redundancy = 0.0
            if selected:
                redundancy = max(_cosine(vectors[i], vectors[j]) for j in selected)
            val = lambda_param * scores[i] - (1.0 - lambda_param) * redundancy
            if val > best_val:
                best_val = val
                best_i = i
        selected.append(best_i)
        remaining.discard(best_i)
    return selected


def _is_whitelisted(url: str) -> bool:
    from knowledge_engine.src.source_evaluator.evaluator import match_whitelist

    matched, _category = match_whitelist(url)
    return matched


def _hit_highlight_text(hit: CurriculumSearchHit) -> str:
    extracts = hit.key_extracts or []
    if extracts:
        return " ".join(extracts)[:2000]
    return (hit.snippet or hit.title or "")[:2000]


def stage1_zero_http_gate(
    core_theme: str,
    hits: list[CurriculumSearchHit],
) -> list[CurriculumSearchHit]:
    """Rank ALL candidates by local cross-encoder score over Exa highlights —
    no HTTP, no LLM call, and — unlike the old one-shot 30-50% cutoff — NO
    candidate is dropped here. This full ranking IS the RAM-resident priority
    queue that Phase 2 (In-Memory Replenishment, see run_pre_flight_triage)
    draws from in batches; a low-ranked candidate stays available for
    replenishment instead of being permanently discarded up front.

    Whitelisted domains are pinned ahead of non-whitelisted ones so they are
    guaranteed to reach Stage 2 early, ties broken by cross-encoder score.
    """
    if not hits:
        return []
    theme = (core_theme or "").strip()
    if not theme:
        trace("PRE_FLIGHT stage1 ⊘ | empty core_theme — queue in original order")
        return list(hits)

    from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs

    texts = [_hit_highlight_text(h) for h in hits]
    scores = score_relevance_pairs(theme, texts)
    ranked = sorted(
        zip(hits, scores),
        key=lambda p: (_is_whitelisted(p[0].url), p[1]),
        reverse=True,
    )
    queue = [h for h, _s in ranked]
    whitelisted_n = sum(1 for h in hits if _is_whitelisted(h.url))
    trace(
        f"PRE_FLIGHT stage1 ✓ | queued={len(queue)} whitelisted={whitelisted_n} "
        "(no drop — full ranked queue for replenishment)"
    )
    return queue


def _extract_paragraphs(html: str, url: str, *, min_chars: int) -> list[str]:
    if not (html or "").strip():
        return []
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=False,
            include_formatting=False,
        )
    except Exception as exc:
        trace(f"PRE_FLIGHT stage2 trafilatura ✗ | {url[:60]} | {exc}")
        return []
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n")]
    return [p for p in paragraphs if len(p) >= min_chars]


async def stage2_parallel_fetch(
    hits: list[CurriculumSearchHit],
    *,
    top_k: int | None = None,
    concurrency: int | None = None,
    timeout: float | None = None,
    min_paragraph_chars: int | None = None,
) -> tuple[
    dict[str, tuple[CurriculumSearchHit, str, list[str], bool]],
    list[CurriculumSearchHit],
]:
    """Parallel GET for the top_k Stage-1 survivors; Trafilatura paragraph split.

    Liveness / soft-404 / anti-bot are checked directly on the downloaded
    body — no separate check_url_live() HEAD+GET round trip. Returns
    (url -> (hit, html, paragraphs, is_code_source), dead_hits) — dead_hits
    are hits that fetched a live, sufficiently-sized body but yielded no
    extractable paragraphs; kept (unlike a hard anti-bot/thin-body drop) so
    the caller can still count them toward Phase 3 replenishment instead of
    losing track of why the batch came up short.
    """
    if not hits:
        return {}, []
    k = top_k if top_k is not None else CURRICULUM_PREFLIGHT_STAGE2_TOP_K
    subset = hits[: max(1, k)]
    conc = (
        concurrency
        if concurrency is not None
        else CURRICULUM_PREFLIGHT_FETCH_CONCURRENCY
    )
    tmo = timeout if timeout is not None else CURRICULUM_PREFLIGHT_FETCH_TIMEOUT_SEC
    min_chars = (
        min_paragraph_chars
        if min_paragraph_chars is not None
        else CURRICULUM_PREFLIGHT_PARAGRAPH_MIN_CHARS
    )
    code_min_chars = CURRICULUM_PREFLIGHT_CODE_MIN_CHARS

    from knowledge_engine.services.web_extract import (
        is_anti_bot_fetch_result,
        smart_fetch_page_html,
    )

    sem = asyncio.Semaphore(max(1, conc))

    async def _one(
        hit: CurriculumSearchHit,
    ) -> tuple[CurriculumSearchHit, str, list[str], bool] | CurriculumSearchHit | None:
        async with sem:
            try:
                html, method = await asyncio.wait_for(
                    asyncio.to_thread(smart_fetch_page_html, hit.url),
                    timeout=tmo,
                )
            except Exception as exc:
                trace(f"PRE_FLIGHT stage2 fetch ✗ | {hit.url[:60]} | {exc}")
                return None
            if is_anti_bot_fetch_result("", method, html=html):
                trace(f"PRE_FLIGHT stage2 ⊘ anti_bot | {hit.url[:60]}")
                return None
            if len((html or "").strip()) < 200:
                trace(f"PRE_FLIGHT stage2 ⊘ thin_body | {hit.url[:60]}")
                return None
            is_code_source = detect_code_content(hit.url, html, method)
            if is_code_source:
                paragraphs = _code_paragraphs_for_embedding(
                    html, hit.url, min_chars=code_min_chars
                )
            else:
                paragraphs = _extract_paragraphs(html, hit.url, min_chars=min_chars)
                paragraphs += _extract_embedded_code_blocks(
                    html, hit.url, min_chars=code_min_chars
                )
            if not paragraphs:
                trace(f"PRE_FLIGHT stage2 ⊘ no_paragraphs | {hit.url[:60]}")
                return hit
            return hit, html, paragraphs, is_code_source

    results = await asyncio.gather(*[_one(h) for h in subset])
    out: dict[str, tuple[CurriculumSearchHit, str, list[str], bool]] = {}
    dead: list[CurriculumSearchHit] = []
    for r in results:
        if r is None:
            continue
        if isinstance(r, CurriculumSearchHit):
            dead.append(r)
            continue
        hit, html, paragraphs, is_code_source = r
        out[hit.url] = (hit, html, paragraphs, is_code_source)
    trace(
        f"PRE_FLIGHT stage2 ✓ | fetched={len(subset)} survived={len(out)} "
        f"code_sources={sum(1 for entry in out.values() if entry[3])} "
        f"dead={len(dead)}"
    )
    return out, dead


def stage3_mmr_paragraphs(
    core_theme: str,
    paragraphs: list[str],
    *,
    top_k: int | None = None,
    lambda_param: float | None = None,
) -> list[str]:
    """BGE-M3 embeddings + greedy MMR — diverse, relevant paragraphs for one article."""
    if not paragraphs:
        return []
    k = top_k if top_k is not None else CURRICULUM_PREFLIGHT_MMR_TOP_K
    lam = lambda_param if lambda_param is not None else CURRICULUM_PREFLIGHT_MMR_LAMBDA

    from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

    theme = (core_theme or "").strip()
    if theme:
        vecs = embed_texts_bge_m3([theme, *paragraphs])
        theme_vec, para_vecs = vecs[0], vecs[1:]
        scores = [_cosine(theme_vec, v) for v in para_vecs]
    else:
        para_vecs = embed_texts_bge_m3(paragraphs)
        scores = [1.0] * len(para_vecs)

    idx = greedy_mmr_select(para_vecs, scores, top_k=k, lambda_param=lam)
    return [paragraphs[i] for i in idx]


def stage3_mmr_paragraphs_batch(
    core_theme: str,
    paragraphs_per_hit: list[list[str]],
    *,
    top_k: int | None = None,
    lambda_param: float | None = None,
) -> list[list[str]]:
    """Батчевый Stage 3 — ОДИН BGE-M3 forward pass на весь раунд кандидатов
    вместо одного embed_texts_bge_m3() на каждый URL (тот же класс проблемы,
    что и Stage 4 — см. докстринг stage4_quality_gate_batch: N URL → N
    отдельных вызовов, каждый ~5-17s из-за MPS dispatch overhead на
    маленьких батчах — подтверждено логом реального прогона, 4 hit'а ≈35s
    суммарно). ``core_theme`` одинаков для всего раунда — эмбедится ОДИН
    раз, а не по разу на каждый hit, как в per-hit stage3_mmr_paragraphs
    выше (которая остаётся для прямых/точечных вызовов и тестов)."""
    n = len(paragraphs_per_hit)
    if n == 0:
        return []
    k = top_k if top_k is not None else CURRICULUM_PREFLIGHT_MMR_TOP_K
    lam = lambda_param if lambda_param is not None else CURRICULUM_PREFLIGHT_MMR_LAMBDA

    from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

    theme = (core_theme or "").strip()
    flat_texts: list[str] = [theme] if theme else []
    spans: list[tuple[int, int]] = []
    for paragraphs in paragraphs_per_hit:
        start = len(flat_texts)
        flat_texts.extend(paragraphs or [])
        spans.append((start, len(flat_texts)))

    if not flat_texts:
        return [[] for _ in paragraphs_per_hit]

    vecs = embed_texts_bge_m3(flat_texts)
    theme_vec = vecs[0] if theme else None

    results: list[list[str]] = []
    for (start, end), paragraphs in zip(spans, paragraphs_per_hit):
        para_vecs = vecs[start:end]
        if not para_vecs:
            results.append([])
            continue
        if theme_vec is not None:
            scores = [_cosine(theme_vec, v) for v in para_vecs]
        else:
            scores = [1.0] * len(para_vecs)
        idx = greedy_mmr_select(para_vecs, scores, top_k=k, lambda_param=lam)
        results.append([paragraphs[i] for i in idx])
    return results


def _keyword_coverage(keywords: list[str], text: str) -> float:
    """Partial credit per keyword: a multi-word keyword ("reference counting")
    does not need to match as one exact phrase — each of its own words is
    counted separately and a keyword's credit is the fraction of its words
    found. A keyword-sparse but genuinely on-topic page (different phrasing,
    e.g. official docs) no longer gets rounded down to a hard 0 for that
    keyword the way an exact-phrase-only check would."""
    kws = [k.strip().lower() for k in (keywords or []) if (k or "").strip()]
    if not kws:
        return 1.0
    low = (text or "").lower()
    total = 0.0
    for kw in kws:
        if kw in low:
            total += 1.0
            continue
        tokens = [t for t in kw.split() if t]
        if not tokens:
            continue
        matched = sum(1 for t in tokens if t in low)
        total += matched / len(tokens)
    return total / len(kws)


def stage4_quality_gate_batch(
    core_theme: str,
    keywords: list[str],
    paragraphs_per_hit: list[list[str]],
) -> list[float]:
    """Батчевый Stage 4 — ОДИН cross-encoder forward pass на весь раунд
    кандидатов вместо одного ``model.predict()`` на каждый URL (было: N URL
    → N отдельных вызовов, каждый ~6-8s из-за MPS dispatch overhead на
    маленьких батчах — см. лог-аудит OPTIMIZATION STEP 3, 4 URL ≈26s
    последовательно). Формула Triage Score не меняется — peak/mean_top3
    считаются из того же общего forward pass, просто разбитого по границам
    hits вместо N отдельных проходов."""
    theme = (core_theme or "").strip()
    n = len(paragraphs_per_hit)
    if not theme or n == 0:
        return [0.0] * n

    from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs

    flat_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for paragraphs in paragraphs_per_hit:
        start = len(flat_texts)
        flat_texts.extend(paragraphs or [])
        spans.append((start, len(flat_texts)))

    flat_scores = score_relevance_pairs(theme, flat_texts) if flat_texts else []

    results: list[float] = []
    for (start, end), paragraphs in zip(spans, paragraphs_per_hit):
        scores = flat_scores[start:end]
        if not scores:
            results.append(0.0)
            continue
        ranked = sorted(scores, reverse=True)
        peak = ranked[0]
        top3 = ranked[: min(3, len(ranked))]
        mean_top3 = sum(top3) / len(top3)
        coverage = _keyword_coverage(keywords, " ".join(paragraphs))
        total = 0.5 * peak + 0.3 * mean_top3 + 0.2 * coverage
        trace(
            f"PRE_FLIGHT stage4 score | peak={peak:.3f} mean_top3={mean_top3:.3f} "
            f"coverage={coverage:.3f} total={total:.3f}"
        )
        results.append(total)
    return results


def stage4_quality_gate(
    core_theme: str,
    keywords: list[str],
    mmr_paragraphs: list[str],
) -> float:
    """Triage Score = 0.5*Peak + 0.3*Mean_Top3 + 0.2*KeywordCoverage.

    Однохитовая обёртка над ``stage4_quality_gate_batch`` (см. её докстринг —
    раунд в ``run_pre_flight_triage`` вызывает батчевую версию напрямую)."""
    if not mmr_paragraphs or not (core_theme or "").strip():
        return 0.0
    scores = stage4_quality_gate_batch(core_theme, keywords, [mmr_paragraphs])
    return scores[0] if scores else 0.0


async def run_pre_flight_triage(
    hits: list[CurriculumSearchHit],
    *,
    core_theme: str,
    keywords: list[str],
    final_articles: int | None = None,
    hard_gate_threshold: float | None = None,
    whitelist_hard_gate_threshold: float | None = None,
    replenish_batch_size: int | None = None,
) -> list[CurriculumSearchHit]:
    """Two-phase local pipeline with guaranteed replenishment.

    Phase 1 (stage1_zero_http_gate): one Exa fetch, one cross-encoder pass
    over the FULL candidate pool -> RAM-resident priority queue, nothing
    dropped yet.

    Phase 2 (this loop): pull consecutive batches off the front of that
    queue, run Stage 2-4 on each batch, and stop as soon as `final_articles`
    survivors have cleared the hard gate. If a batch does not fill the quota,
    the NEXT batch is pulled from the SAME already-ranked queue — no repeat
    Exa or Flash Lite calls, strictly in-memory.

    Phase 3 (safety fallback), only if the whole queue is exhausted and
    survivors are still short: (a) admit hard-gate-rejected candidates on
    whitelisted domains above a relaxed threshold, then (b) fill any
    remainder with the highest-scoring remaining rejects regardless of
    threshold (Best-of-Rest) — so MAP+REDUCE gets `final_articles` sources
    whenever the candidate pool was large enough to provide them.

    Survivors' HTML is stashed for Zero-Waste Handover (pop_preflight_html);
    the returned hits are exactly the ones downstream ingest should process.
    """
    if not hits:
        return []
    threshold = (
        hard_gate_threshold
        if hard_gate_threshold is not None
        else CURRICULUM_PREFLIGHT_HARD_GATE_THRESHOLD
    )
    relaxed_threshold = (
        whitelist_hard_gate_threshold
        if whitelist_hard_gate_threshold is not None
        else CURRICULUM_PREFLIGHT_WHITELIST_HARD_GATE_THRESHOLD
    )
    n_final = (
        final_articles
        if final_articles is not None
        else CURRICULUM_PREFLIGHT_FINAL_ARTICLES
    )
    batch_size = max(
        1,
        (
            replenish_batch_size
            if replenish_batch_size is not None
            else CURRICULUM_PREFLIGHT_STAGE2_TOP_K
        ),
    )

    queue = stage1_zero_http_gate(core_theme, hits)
    if not queue:
        return []

    survivors: list[tuple[CurriculumSearchHit, float, str]] = []
    rejected: list[tuple[CurriculumSearchHit, float, str]] = []
    cursor = 0
    round_n = 0

    while len(survivors) < n_final and cursor < len(queue):
        batch = queue[cursor : cursor + batch_size]
        cursor += len(batch)
        round_n += 1
        fetched, dead = await stage2_parallel_fetch(batch, top_k=len(batch))
        for dead_hit in dead:
            # Нет пригодного тела (anti-bot/тонкая страница/нет абзацев) —
            # всё равно попадает в fallback-пул со score 0.0 / без html,
            # чтобы Phase 3 было из чего добирать, если именно Stage 2
            # оставил батч не добравшим квоту. Промах кэша Zero-Waste
            # Handover ниже по потоку — штатный no-op (инджест просто сам
            # заново скачает страницу).
            rejected.append((dead_hit, 0.0, ""))
        # Stage 3 и Stage 4 — оба ОДИН батчевый BGE-M3/cross-encoder forward
        # pass на весь раунд кандидатов вместо N отдельных вызовов (было:
        # N URL → N отдельных model.predict()/embed(), см. лог-аудит
        # OPTIMIZATION STEP 3 для Stage 4, и реальный прогон 2026-08-29 для
        # Stage 3 — 4 URL по одному съедали ~35s из-за MPS dispatch overhead
        # на маленьких батчах).
        round_hits: list[CurriculumSearchHit] = []
        round_html: list[str] = []
        round_is_code: list[bool] = []
        round_paragraphs_raw: list[list[str]] = []
        for hit in batch:
            entry = fetched.get(hit.url)
            if entry is None:
                continue  # учтено выше в `dead`
            _h, html, paragraphs, is_code_source = entry
            round_hits.append(hit)
            round_html.append(html)
            round_is_code.append(is_code_source)
            round_paragraphs_raw.append(paragraphs)

        round_paragraphs = await asyncio.to_thread(
            stage3_mmr_paragraphs_batch, core_theme, round_paragraphs_raw
        )

        round_scores = await asyncio.to_thread(
            stage4_quality_gate_batch, core_theme, keywords, round_paragraphs
        )

        for hit, html, is_code_source, score in zip(
            round_hits, round_html, round_is_code, round_scores
        ):
            if is_code_source:
                # Code Preservation Policy: NL cross-encoder систематически
                # занижает оценку исходного кода против prose-темы — это не
                # сигнал качества для кода, так что он никогда не отсеивает
                # код-источник по score. Реальный score сохраняется только
                # для ранжирования; финальное суждение о содержимом — позже,
                # на Flash Lite в inbound ingest gate.
                trace(
                    f"PRE_FLIGHT stage4 code_immunity ✓ | {hit.url[:60]} | "
                    f"score={score:.3f} (cross-encoder pessimization ignored)"
                )
                survivors.append((hit, score, html))
            elif score >= threshold:
                survivors.append((hit, score, html))
            else:
                trace(
                    f"PRE_FLIGHT hard_gate ⊘ | {hit.url[:60]} | "
                    f"score={score:.3f} < {threshold:.2f}"
                )
                rejected.append((hit, score, html))
        trace(
            f"PRE_FLIGHT replenish round={round_n} ✓ | queue_pos={cursor}/{len(queue)} "
            f"survivors={len(survivors)}/{n_final} rejected={len(rejected)}"
        )

    if len(survivors) < n_final and rejected:
        need = n_final - len(survivors)
        rejected.sort(key=lambda t: -t[1])
        promoted: list[tuple[CurriculumSearchHit, float, str]] = []
        remaining: list[tuple[CurriculumSearchHit, float, str]] = []
        for hit, score, html in rejected:
            if (
                len(promoted) < need
                and score >= relaxed_threshold
                and _is_whitelisted(hit.url)
            ):
                promoted.append((hit, score, html))
            else:
                remaining.append((hit, score, html))
        if promoted:
            trace(
                f"PRE_FLIGHT safety_fallback whitelist_relax ▶ | "
                f"promoted={len(promoted)} threshold={relaxed_threshold:.2f}"
            )
        survivors.extend(promoted)
        rejected = remaining

    if len(survivors) < n_final and rejected:
        need = n_final - len(survivors)
        rejected.sort(key=lambda t: -t[1])
        best_of_rest = rejected[:need]
        trace(
            f"PRE_FLIGHT safety_fallback best_of_rest ▶ | promoted={len(best_of_rest)} "
            f"scores={[round(s, 3) for _h, s, _html in best_of_rest]}"
        )
        survivors.extend(best_of_rest)

    survivors.sort(key=lambda t: -t[1])
    final = survivors[:n_final]
    for hit, _score, html in final:
        stash_preflight_html(hit.url, html)

    trace(
        f"PRE_FLIGHT ✓ | in={len(hits)} queued={len(queue)} rounds={round_n} "
        f"final={len(final)}/{n_final}"
    )
    return [hit for hit, _score, _html in final]
