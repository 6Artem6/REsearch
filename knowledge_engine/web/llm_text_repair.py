"""Починка типичного «сырого» текста от LLM (литералы \\n, inline mermaid)."""

from __future__ import annotations

import json
import re

_RAW_MERMAID_INLINE = re.compile(
    r"^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|"
    r"stateDiagram(?:-v2)?\b|erDiagram\b)",
    re.IGNORECASE,
)

_SEQ_BLOCK_KW = re.compile(
    r"\s+(participant\s|actor\s|rect\s|loop\s|alt\s|opt\s|par\s|and\s|else\s|"
    r"critical\s|break\s)",
    re.IGNORECASE,
)
_SEQ_NOTE = re.compile(r"\s+(Note\s+(?:over|left of|right of)\s)", re.IGNORECASE)
_SEQ_ACT = re.compile(r"\s+(activate\s|deactivate\s)", re.IGNORECASE)
_SEQ_ARROW = re.compile(r"\s+([A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:)")


def repair_llm_literal_escapes(text: str) -> str:
    """Литералы \\n / \\t → реальные символы (частый JSON/structured output)."""
    if not text:
        return ""
    t = text
    for _ in range(6):
        t2 = t.replace("\\r\\n", "\n").replace("\\r", "\n")
        t2 = t2.replace("\\n", "\n").replace("\\t", "\t")
        if t2 == t:
            break
        t = t2
    return t


_LIST_SECTIONS: tuple[tuple[str, str], ...] = (
    ("pros", "Плюсы"),
    ("cons", "Минусы и риски"),
    ("cons_and_risks", "Минусы и риски"),
    ("takeaways", "Ключевые выводы"),
    ("failure_modes", "Типичные сбои"),
)


def _looks_like_analysis_object(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    if "title" not in keys and "description" not in keys:
        return False
    return bool(
        keys & {"pros", "cons", "cons_and_risks", "takeaways", "failure_modes"}
    )


def _format_analysis_object(obj: dict) -> str:
    parts: list[str] = []
    title = str(obj.get("title") or "").strip()
    if title:
        parts.append(f"## {title}")
    desc = str(obj.get("description") or "").strip()
    if desc:
        parts.append(desc)
    seen_labels: set[str] = set()
    for key, label in _LIST_SECTIONS:
        if label in seen_labels:
            continue
        raw = obj.get(key)
        if not raw:
            continue
        items = [str(x).strip() for x in raw if str(x).strip()]
        if not items:
            continue
        seen_labels.add(label)
        parts.append(f"### {label}")
        parts.extend(f"- {it}" for it in items)
    return "\n\n".join(parts).strip()


def _iter_json_objects(text: str):
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if _looks_like_analysis_object(obj):
            yield i, end, obj
        i = end if end > i else i + 1


def repair_structured_analysis_json(text: str) -> str:
    """
    LLM иногда вставляет trade-off JSON (title/pros/cons/takeaways/failure_modes)
    в tutor_message или summary — превращаем в Markdown для UI.
    """
    raw = (text or "").strip()
    if not raw or "{" not in raw:
        return text or ""

    if raw.startswith("{") or raw.startswith("```"):
        fenced = raw
        if fenced.startswith("```"):
            fenced = re.sub(r"^```(?:json)?\s*", "", fenced, flags=re.IGNORECASE)
            fenced = re.sub(r"```\s*$", "", fenced.strip())
        try:
            whole = json.loads(fenced.strip())
            if _looks_like_analysis_object(whole):
                return _format_analysis_object(whole)
        except json.JSONDecodeError:
            pass

    spans = list(_iter_json_objects(raw))
    if not spans:
        return text

    out = raw
    for start, end, obj in reversed(spans):
        md = _format_analysis_object(obj)
        before = raw[:start].rstrip()
        last_line = before.split("\n")[-1].strip() if before else ""
        title = str(obj.get("title") or "").strip()
        if last_line and title:
            if title.lower() in last_line.lower() or last_line.lower() in title.lower():
                md_lines = md.split("\n", 1)
                if md_lines and md_lines[0].startswith("## "):
                    md = md_lines[1].strip() if len(md_lines) > 1 else ""
        replacement = md if md else raw[start:end]
        out = out[:start] + replacement + out[end:]

    return out.strip()


def _strip_outer_quotes(s: str) -> str:
    t = (s or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1].strip()
    return t


def _strip_fence(d: str) -> str:
    inner = re.sub(r"^```(?:mermaid)?\s*", "", d.strip(), count=1, flags=re.IGNORECASE)
    inner = re.sub(r"```\s*$", "", inner.strip(), flags=re.IGNORECASE)
    return inner.strip()


def _quote_subgraph_titles(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*subgraph\s+)(.+)$", line, re.IGNORECASE)
        if not m:
            out.append(line)
            continue
        rest = m.group(2).strip()
        if rest.startswith('"') or rest.startswith("'"):
            out.append(line)
            continue
        if re.match(r"^\w[\w-]*\s*\[", rest):
            out.append(line)
            continue
        safe = rest.replace('"', "'")
        out.append(f"{m.group(1)}\"{safe}\"")
    return "\n".join(out)


def _quote_participant_aliases(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*participant\s+\S+\s+as\s+)(.+)$", line, re.IGNORECASE)
        if not m:
            out.append(line)
            continue
        alias = m.group(2).strip()
        if alias.startswith('"') or alias.startswith("'"):
            out.append(line)
            continue
        if "/" in alias or "(" in alias or "  " in alias:
            safe = alias.replace('"', "'")
            out.append(f"{m.group(1)}\"{safe}\"")
        else:
            out.append(line)
    return "\n".join(out)


def _quote_loop_labels(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*loop\s+)(.+)$", line, re.IGNORECASE)
        if not m:
            out.append(line)
            continue
        rest = m.group(2)
        arrow = _SEQ_ARROW.search(rest)
        label = rest[: arrow.start()] if arrow else rest
        label = label.strip()
        tail = rest[arrow.start():] if arrow else ""
        if not label or label.startswith('"') or label.startswith("'"):
            out.append(line)
            continue
        if "(" in label or " " in label:
            safe = label.replace('"', "'")
            out.append(f"{m.group(1)}\"{safe}\"{tail}")
        else:
            out.append(line)
    return "\n".join(out)


def _wrap_long_label(text: str, max_len: int = 38) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    words = t.split()
    lines: list[str] = []
    line = ""
    for w in words:
        if not line:
            line = w
        elif len(line) + 1 + len(w) <= max_len:
            line += f" {w}"
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)


def _ensure_sequence_init(inner: str) -> str:
    if not re.search(r"^sequenceDiagram", inner, re.I | re.M):
        return inner
    if re.search(r"%%\s*\{init:", inner, re.I):
        return inner
    init = (
        "%%{init: {'themeVariables': {'fontSize': '10px'}, "
        "'sequence': {'wrap': true, 'width': 240, 'messageFontSize': 10, "
        "'noteFontSize': 10, 'actorFontSize': 11, 'messageMargin': 48, "
        "'boxMargin': 10, 'mirrorActors': false}}}%%\n"
    )
    return init + inner


def _sanitize_note_lines(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    note_re = re.compile(
        r"^(\s*Note\s+(?:over|left of|right of)\s+[^:]+:\s*)(.*)$",
        re.IGNORECASE,
    )
    for line in lines:
        m = note_re.match(line)
        if not m:
            out.append(line)
            continue
        prefix = m.group(1)
        body = (m.group(2) or "").strip().strip('"').replace('"', "'")
        body = _wrap_long_label(body, 36)
        if not body:
            out.append(line)
            continue
        out.append(prefix + f'"{body}"')
    return "\n".join(out)


def _quote_arrow_messages(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    arrow_re = re.compile(r"^(\s*[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:\s*)(.*)$")
    for line in lines:
        m = arrow_re.match(line)
        if not m:
            out.append(line)
            continue
        prefix = m.group(1)
        body = (m.group(2) or "").strip().strip('"')
        if not body:
            out.append(line)
            continue
        body = _wrap_long_label(body.replace('"', "'"), 40)
        out.append(f'{prefix}"{body}"')
    return "\n".join(out)


def _format_mermaid_inner(inner: str) -> str:
    s = _quote_subgraph_titles(repair_llm_literal_escapes(inner).strip())
    if not s:
        return s
    s = _ensure_sequence_init(s)

    s = re.sub(
        r"^(sequenceDiagram(?:\s+autonumber)?)\s+",
        r"\1\n",
        s,
        count=1,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"^(graph\s+(?:TD|LR|BT|RL))\s+",
        r"\1\n",
        s,
        count=1,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"^(flowchart\s+(?:TD|LR|BT|RL)?)\s+",
        r"\1\n",
        s,
        count=1,
        flags=re.IGNORECASE,
    )

    s = _SEQ_BLOCK_KW.sub(r"\n\1", s)
    s = _SEQ_NOTE.sub(r"\n\1", s)
    s = _SEQ_ACT.sub(r"\n\1", s)
    s = re.sub(r"\s+(subgraph\s)", r"\n\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(end)\b", r"\n\1", s, flags=re.IGNORECASE)
    s = _SEQ_ARROW.sub(r"\n\1", s)

    s = _quote_participant_aliases(s)
    s = _quote_loop_labels(s)
    s = _sanitize_note_lines(s)
    s = _quote_arrow_messages(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _mermaid_node_id(raw: str) -> str:
    t = (raw or "").strip()
    if not t:
        return "node"
    m = re.match(r"^(.+?)\s+\(([^)]+)\)\s*$", t)
    if m:
        base = m.group(1).strip()
        inner = m.group(2).strip()
        id_ = f"{base.replace(' ', '_')}_{inner.replace(' ', '_')}"
        return f'{id_}["{base.replace(chr(34), chr(39))} ({inner.replace(chr(34), chr(39))})"]'
    if re.search(r"[\s/]", t):
        id_ = re.sub(r"[^\w]+", "_", t).strip("_") or "node"
        return f'{id_}["{t.replace(chr(34), chr(39))}"]'
    return t.replace(" ", "_")


def _parse_link_chain(chain: str) -> list[tuple[str, str, str]] | None:
    s = re.sub(r"--\(([^)]+)\)-->", r"-->|\1|", chain)
    parts = [p.strip() for p in re.split(r"\s*-->\s*", s) if p.strip()]
    if len(parts) < 2:
        return None
    edges: list[tuple[str, str, str]] = []
    nodes: list[str] = []
    for i, part in enumerate(parts):
        if i == 0:
            nodes.append(part)
            continue
        label = ""
        lm = re.match(r"^\|([^|]+)\|\s*(.+)$", part, re.DOTALL)
        if lm:
            label = lm.group(1).strip()
            part = lm.group(2).strip()
        from_n = nodes[-1]
        edges.append((from_n, part, label))
        nodes.append(part)
    return edges


def _lift_pseudo_flowchart(text: str) -> str:
    t = _strip_outer_quotes(repair_llm_literal_escapes(text)).strip()
    if not t or _RAW_MERMAID_INLINE.search(t.split("\n", 1)[0]):
        return ""
    if not re.search(r"--\([^)]+\)-->|-->", t):
        return ""
    segments = (
        [s for s in re.split(r"\s*\|\s*", t) if re.search(r"-->|--\(", s)]
        if "|" in t
        else [t]
    )
    if not segments:
        return ""
    lines = ["flowchart LR"]
    for idx, seg in enumerate(segments):
        title = ""
        chain = seg.strip()
        m = re.match(r"^([^:]{2,56}):\s*(.+)$", chain, re.DOTALL)
        if m and re.search(r"-->|--\(", m.group(2)):
            title = m.group(1).strip()
            chain = m.group(2).strip()
        edges = _parse_link_chain(chain)
        if not edges:
            continue
        sub_id = f"sg_{idx}"
        if title:
            safe = title.replace('"', "'").replace("]", "")
            lines.append(f"subgraph {sub_id} [{safe}]")
        prefix = "  " if title else ""
        for from_n, to_n, label in edges:
            a = _mermaid_node_id(from_n)
            b = _mermaid_node_id(to_n)
            lbl = f"|{label.replace(chr(34), chr(39))}|" if label else ""
            lines.append(f"{prefix}{a} -->{lbl} {b}")
        if title:
            lines.append("end")
    return "\n".join(lines) if len(lines) > 1 else ""


def _repair_diagram_once(diagram: str) -> str:
    d = _strip_outer_quotes(repair_llm_literal_escapes((diagram or "").strip()))
    if not d:
        return d

    if d.startswith("```"):
        inner = _format_mermaid_inner(_strip_fence(d))
        return f"```mermaid\n{inner}\n```" if inner else d

    if _RAW_MERMAID_INLINE.search(d):
        inner = _format_mermaid_inner(d)
        return f"```mermaid\n{inner}\n```" if inner else d

    lifted = _lift_pseudo_flowchart(d)
    if lifted:
        inner = _format_mermaid_inner(lifted)
        return f"```mermaid\n{inner}\n```" if inner else d
    return d


def repair_diagram_markdown(diagram: str) -> str:
    """Mermaid в content.diagram: unescape, fence, переносы, sequenceDiagram."""
    d = (diagram or "").strip()
    for _ in range(4):
        d2 = _repair_diagram_once(d)
        if d2 == d:
            break
        d = d2
    return d
