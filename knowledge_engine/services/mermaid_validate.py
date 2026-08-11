"""Валидация и детерминированная санитизация Mermaid."""

from __future__ import annotations

import re

from knowledge_engine.ui.run_log import trace

_MERMAID_HEAD = re.compile(
    r"^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|"
    r"stateDiagram(?:-v2)?\b|erDiagram\b|xychart(?:-beta)?\b)",
    re.IGNORECASE | re.MULTILINE,
)

_FORBIDDEN = re.compile(r"```|<script", re.IGNORECASE)

_BENCHMARK_TEXT_HINTS = re.compile(
    r"\b(qps|recall@?k?|latency|throughput|benchmark|performance\s+vs|vs\s+recall)\b",
    re.IGNORECASE,
)

_BROKEN_INIT = re.compile(r"%%\s*\{init:", re.I)
_INIT_WELL_FORMED = re.compile(
    r"%%\s*\{init:\s*\{[\s\S]*?\}\s*\}%%\s*",
    re.I,
)

_LINE_START_OK = re.compile(
    r"^(?:"
    r"sequenceDiagram\b|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|"
    r"stateDiagram(?:-v2)?\b|erDiagram\b|xychart(?:-beta)?\b|"
    r"participant\s|actor\s|autonumber\b|"
    r"Note\s+(?:over|left of|right of)\s|"
    r"rect\s|loop\s|alt\s|opt\s|par\s|else\b|end\b|and\s*$|"
    r"activate\s|deactivate\s|critical\s|break\s|"
    r"subgraph\s|title\s|x-axis\s|y-axis\s|line\s|bar\s|"
    r"class\s|style\s|linkStyle\s|direction\s|"
    r"[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:|"
    r"%%"
    r")",
    re.IGNORECASE,
)

_ARROW_LINE = re.compile(
    r"^[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:\s*(.*)$",
    re.IGNORECASE,
)


def has_broken_init_directive(code: str) -> bool:
    inner = strip_mermaid_fences(code)
    if not _BROKEN_INIT.search(inner):
        return False
    if _INIT_WELL_FORMED.search(inner):
        return False
    return True


def strip_mermaid_init_directive(code: str) -> str:
    inner = strip_mermaid_fences(code)
    if not inner:
        return ""
    peeled = _INIT_WELL_FORMED.sub("", inner).strip()
    while _INIT_WELL_FORMED.search(peeled):
        peeled = _INIT_WELL_FORMED.sub("", peeled, count=1).strip()
    if has_broken_init_directive(peeled):
        peeled = _BROKEN_INIT.sub("", peeled, count=1).strip()
    return peeled


def _dedupe_init_directives(inner: str) -> str:
    """Удалить все %%{init}%% — сервер/UI добавят один при необходимости."""
    s = (inner or "").strip()
    while _INIT_WELL_FORMED.search(s):
        s = _INIT_WELL_FORMED.sub("", s, count=1).strip()
    s = re.sub(r"%%\s*\{init:[\s\S]*?\}%%\s*", "", s, flags=re.I).strip()
    return s


def _fix_unbalanced_quotes(line: str) -> str:
    t = line.rstrip()
    if t.count('"') % 2 == 1:
        return t + '"'
    return t


_CONTINUATION_LINE = re.compile(
    r"^(?:and\s+[a-z]|coherency\b|output\s|queries\b|[a-z].*activity)",
    re.I,
)


def _merge_orphan_lines(inner: str) -> str:
    lines = (inner or "").split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if out and out[-1].strip():
                out.append("")
            continue
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        merge = False
        if _CONTINUATION_LINE.match(stripped):
            merge = True
        elif not _LINE_START_OK.match(stripped):
            merge = True
        if merge:
            joined = (prev.rstrip() + " " + stripped).strip()
            out[-1] = joined
            continue
        out.append(line)
    lines = [_fix_unbalanced_quotes(ln) for ln in out]
    return "\n".join(lines)


def _lift_pseudo_xychart(inner: str) -> str:
    """VLM иногда кладёт xychart-beta внутрь flowchart LR — восстановить нормальный синтаксис."""
    if not re.search(r"flowchart\s", inner, re.I):
        return inner
    if "xychart" not in inner.lower():
        return inner
    embed = re.search(r'\["xychart-beta\s+(.+?)"\s*\]', inner, re.I | re.S)
    blob = f"xychart-beta {embed.group(1)}" if embed else inner.replace("_", " ")
    title_m = re.search(r"title\s+'([^']+)'", blob, re.I)
    x_m = re.search(r"x-axis\s+'([^']+)'\s*(\[[^\]]+\])", blob, re.I)
    y_m = re.search(r"y-axis\s+'([^']+)'\s*([\d.]+)", blob, re.I)
    bar_m = re.search(r"bar\s*(\[[0-9.,\s]+\])", inner, re.I)
    if not (title_m and x_m and bar_m):
        return inner
    title = title_m.group(1).replace('"', "'")
    xlab = x_m.group(1)
    labels = x_m.group(2)
    ylab = (y_m.group(1) if y_m else "Y").replace('"', "'")
    ymax = "100000"
    ym = re.search(r"(\d{3,7})_bar", inner.replace(" ", ""))
    if ym:
        ymax = ym.group(1)
    return (
        "xychart-beta\n"
        f'    title "{title}"\n'
        f'    x-axis "{xlab}" {labels}\n'
        f'    y-axis "{ylab}" 0 --> {ymax}\n'
        f"    bar {bar_m.group(1)}"
    )


def sanitize_mermaid_syntax(code: str) -> str:
    """Детерминированная починка без LLM."""
    from knowledge_engine.utils.mermaid_sanitizer import sanitize_mermaid_code

    inner = strip_mermaid_fences(sanitize_mermaid_code((code or "").strip()))
    if not inner:
        return ""
    inner = _dedupe_init_directives(inner)
    lifted = _lift_pseudo_xychart(inner)
    if lifted != inner:
        inner = lifted
    else:
        inner = _merge_orphan_lines(inner)
    inner = re.sub(r"\blineplot\b", "line", inner, flags=re.I)
    inner = inner.replace('activity""', 'activity"')
    inner = re.sub(r"\boutput buff\b", 'output buffer"', inner)
    lines = [_fix_unbalanced_quotes(ln) for ln in inner.split("\n")]
    inner = "\n".join(lines)
    inner = re.sub(r"\n{3,}", "\n\n", inner).strip()
    from knowledge_engine.utils.mermaid_linter import lint_mermaid_code

    ok, lint_errors = lint_mermaid_code(inner)
    if not ok:
        trace(
            "MERMAID_LINT post-sanitize | "
            + "; ".join(lint_errors[:8])
            + (f" …+{len(lint_errors) - 8}" if len(lint_errors) > 8 else "")
        )
    return inner


def normalize_stored_mermaid(code: str) -> str:
    """Чтение из БД / session: sanitize + форматирование без Gemma."""
    from knowledge_engine.web.llm_text_repair import repair_diagram_markdown

    raw = (code or "").strip()
    if not raw:
        return ""
    sanitized = sanitize_mermaid_syntax(raw)
    if not sanitized:
        return repair_diagram_markdown(raw)
    return repair_diagram_markdown(sanitized)


def _needs_ui_repair(inner: str) -> bool:
    """Эвристика: слабый синтаксис, который ломает Mermaid.js."""
    low = (inner or "").lower()
    if re.search(r"^flowchart\b", inner.strip(), re.I) and (
        "xychart" in low or "x-axis" in low.replace("_", " ")
    ):
        return True
    if is_xychart_mermaid(inner) and re.search(r"\blineplot\b", low):
        return True
    return False


def _validate_sequence_lines(inner: str) -> bool:
    if not re.search(r"^sequenceDiagram\b", inner, re.I | re.M):
        return True
    for line in inner.split("\n"):
        t = line.strip()
        if not t or t.startswith("%%"):
            continue
        if not _LINE_START_OK.match(t):
            return False
        m = _ARROW_LINE.match(t)
        if m:
            body = (m.group(1) or "").strip()
            if body and not (
                body.startswith('"') or re.match(r"^[A-Za-z0-9_]+\s*:", body)
            ):
                if " " in body and not body.startswith('"'):
                    return False
    return True


def _validate_xychart_body(inner: str) -> bool:
    if not is_xychart_mermaid(inner):
        return True
    low = inner.lower()
    if "flowchart" in low or re.match(r"graph\s", inner.strip(), re.I):
        return False
    if "xychart_beta" in low or re.search(r"flowchart\b[\s\S]+xychart", low):
        return False
    if re.search(r"\blineplot\b", low):
        return False
    if "x-axis" not in low and "x-axis" not in inner:
        if "x-axis" not in low.replace("_", "-"):
            return False
    if "y-axis" not in low and "y-axis" not in inner:
        if "y-axis" not in low.replace("_", "-"):
            return False
    if "line" not in low and "bar" not in low:
        return False
    return True


def strip_mermaid_fences(code: str) -> str:
    inner = (code or "").strip()
    if inner.startswith("```"):
        inner = re.sub(r"^```(?:mermaid)?\s*", "", inner, flags=re.I).strip()
        inner = re.sub(r"```\s*$", "", inner).strip()
    return inner


def is_xychart_mermaid(code: str) -> bool:
    inner = strip_mermaid_fences(code).lstrip()
    return bool(re.match(r"xychart(?:-beta)?\b", inner, re.I))


def is_misclassified_benchmark_flowchart(code: str, caption: str = "") -> bool:
    inner = strip_mermaid_fences(code)
    if not inner:
        return False
    if is_xychart_mermaid(inner):
        return False
    low = inner.lower()
    combined = f"{caption} {inner}"
    if not _BENCHMARK_TEXT_HINTS.search(combined):
        return False
    if "flowchart" not in low and not re.match(r"graph\s", low):
        return False
    if re.search(
        r"\[[^\]]*(ось\s*[xy]|x-axis|y-axis|axis)\b",
        inner,
        re.IGNORECASE,
    ):
        return True
    nodes = re.findall(r"\[([^\]]+)\]", inner)
    bench_nodes = sum(
        1 for n in nodes if re.search(r"qps|recall|ось\s*[xy]", n, re.IGNORECASE)
    )
    return bench_nodes >= 2


def validate_mermaid_syntax(code: str) -> bool:
    raw = (code or "").strip()
    if len(raw) < 8 or len(raw) > 12000:
        return False
    if _FORBIDDEN.search(raw):
        return False
    inner = strip_mermaid_fences(raw)
    from knowledge_engine.utils.mermaid_sanitizer import is_mermaid_syntax_valid

    # Fast gate: flowchart+xychart mix / unmatched quotes → Gemma repair path.
    if not is_mermaid_syntax_valid(inner):
        return False
    if has_broken_init_directive(inner):
        return False
    if not _MERMAID_HEAD.search(inner):
        return False
    if inner.count("\n") > 200:
        return False
    if not _validate_xychart_body(inner):
        return False
    low = inner.lower()
    if is_xychart_mermaid(inner) and (
        "flowchart" in low or re.match(r"graph\s", inner.strip(), re.I)
    ):
        return False
    if not _validate_sequence_lines(inner):
        return False
    if _needs_ui_repair(inner):
        return False
    return True


def format_mermaid_for_storage(inner: str) -> str:
    """Fenced mermaid: sanitize → validate; без llm_text_repair если уже валидно."""
    from knowledge_engine.web.llm_text_repair import (
        _ensure_sequence_init,
        repair_diagram_markdown,
    )

    sanitized = sanitize_mermaid_syntax((inner or "").strip())
    if sanitized and validate_mermaid_syntax(sanitized):
        body = _ensure_sequence_init(sanitized)
        return f"```mermaid\n{body}\n```"
    return repair_diagram_markdown(sanitize_mermaid_syntax(inner)).strip()


def _format_for_storage(inner: str) -> str:
    return format_mermaid_for_storage(inner).strip()


def process_mermaid_for_ingest(
    raw_vlm: str,
    *,
    allow_gemma_repair: bool = True,
) -> str:
    """
    Этапы: sanitize → format → validate → (опционально) Gemma repair → validate.
    Возвращает fenced mermaid или "" если сохранять нельзя.
    """
    from knowledge_engine.config import gemma_cloud_api_key_available
    from knowledge_engine.services.mermaid_gemma_repair import repair_invalid_mermaid

    raw = (raw_vlm or "").strip()
    if not raw:
        return ""

    def _try_candidate(source: str) -> str:
        formatted = _format_for_storage(source)
        inner = strip_mermaid_fences(formatted)
        if validate_mermaid_syntax(inner):
            return formatted
        return ""

    hit = _try_candidate(raw)
    if hit:
        return hit

    if allow_gemma_repair and gemma_cloud_api_key_available():
        from knowledge_engine.utils.mermaid_sanitizer import (
            has_mixed_flowchart_xychart,
            sanitize_mermaid_raw_text,
        )

        pre = sanitize_mermaid_raw_text(sanitize_mermaid_syntax(raw) or raw)
        err = "validate_mermaid_syntax failed after sanitize"
        if has_mixed_flowchart_xychart(pre):
            err = (
                "flowchart mixed with xychart-beta "
                "(separate diagram types; extract pure xychart-beta or flowchart)"
            )
        gemma_out = repair_invalid_mermaid(pre, err)
        if gemma_out:
            hit = _try_candidate(gemma_out)
            if hit:
                trace("MERMAID_INGEST ✓ | gemma repair accepted")
                return hit
            trace("MERMAID_INGEST ⊘ | gemma repair still invalid")

    trace(
        "MERMAID_INGEST ⊘ | rejected after sanitize"
        + ("+gemma" if allow_gemma_repair else "")
    )
    return ""


def sanitize_mermaid_from_vlm(code: str) -> str:
    """Лёгкая санитизация сразу после VLM (без Gemma, без жёсткого reject)."""
    return sanitize_mermaid_syntax(code)
