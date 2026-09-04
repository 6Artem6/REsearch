"""Накопление диаграмм, кода и карточек в content ноды (стабильные id для якорей)."""

from __future__ import annotations

import re

from knowledge_engine.services.mermaid_validate import normalize_stored_mermaid
from knowledge_engine.src.node_deep_dive.code_snippet_heuristic import (
    filter_code_snippets,
    is_likely_code_snippet,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    CodeAsset,
    DiagramAsset,
    NodeContentBlock,
    NodeDataInput,
    RichReferenceItem,
)

_MAX_DIAGRAMS = 12
_MAX_CODE = 16
_MAX_CARDS = 16


def _next_id(prefix: str, items: list) -> str:
    nums: list[int] = []
    pat = re.compile(rf"^{prefix}-(\d+)$", re.I)
    for item in items:
        raw_id = (
            getattr(item, "id", None)
            or getattr(item, "asset_id", None)
            or (item.get("id") if isinstance(item, dict) else None)
            or (item.get("asset_id") if isinstance(item, dict) else None)
        )
        m = pat.match(str(raw_id or "").strip())
        if m:
            nums.append(int(m.group(1)))
    return f"{prefix}-{max(nums, default=0) + 1}"


def _norm_mermaid(code: str) -> str:
    raw = (code or "").strip()
    if not raw:
        return ""
    from knowledge_engine.services.mermaid_validate import (
        strip_mermaid_fences,
        validate_mermaid_syntax,
    )
    from knowledge_engine.utils.mermaid_sanitizer import sanitize_mermaid_raw_text

    # Light regex cleanup always (spaced brackets etc.). Skip the heavy
    # deterministic rewrite when the diagram already compiles — it can
    # mangle valid xychart into flowchart noise.
    light = sanitize_mermaid_raw_text(raw)
    if validate_mermaid_syntax(strip_mermaid_fences(light or raw)):
        return light or raw
    return normalize_stored_mermaid(raw)


def normalize_node_content_diagrams(content: NodeContentBlock) -> NodeContentBlock:
    """
    Sanitize every Mermaid asset before serve/persist.

    Legacy ``diagram`` and each ``diagrams[].mermaid`` go through
    ``normalize_stored_mermaid`` (regex cleanup + formatting, no Gemma).
    """
    diagrams_in = list(content.diagrams or [])
    diagrams_out: list[DiagramAsset] = []
    for asset in diagrams_in:
        norm = _norm_mermaid(asset.mermaid)
        if not norm:
            continue
        if norm == (asset.mermaid or "").strip():
            diagrams_out.append(asset)
        else:
            diagrams_out.append(asset.model_copy(update={"mermaid": norm}))
    diagrams_out = _ensure_diagram_titles(diagrams_out)[:_MAX_DIAGRAMS]

    legacy = _norm_mermaid(content.diagram) if (content.diagram or "").strip() else ""
    if not legacy and diagrams_out:
        legacy = diagrams_out[0].mermaid
    elif legacy and not diagrams_out:
        diagrams_out = [
            DiagramAsset(
                id="diagram-1",
                title=infer_diagram_title(legacy),
                mermaid=legacy,
            )
        ]

    if diagrams_out == diagrams_in and legacy == (content.diagram or "").strip():
        return content
    return content.model_copy(
        update={
            "diagrams": diagrams_out,
            "diagram": legacy,
        }
    )


def _diagrams_from_block(block: NodeContentBlock) -> list[DiagramAsset]:
    normalized = normalize_node_content_diagrams(block)
    if normalized.diagrams:
        return list(normalized.diagrams)
    raw = _norm_mermaid(normalized.diagram)
    if not raw:
        return []
    return [DiagramAsset(id="diagram-1", title=infer_diagram_title(raw), mermaid=raw)]


def _infer_code_language(code: str) -> str:
    raw = (code or "").strip()
    m = re.match(r"^```(\w+)", raw)
    if m:
        return m.group(1).lower()[:32]
    if re.search(r"\b(def|class|import)\b", raw):
        return "python"
    if re.search(r"\b(SELECT|INSERT|CREATE)\b", raw, re.I):
        return "sql"
    return "python"


def infer_code_title(code: str) -> str:
    raw = (code or "").strip()
    if not raw:
        return "Фрагмент кода"
    inner = re.sub(r"^```[\w-]*\s*\n?", "", raw, count=1, flags=re.I)
    inner = re.sub(r"\n?```\s*$", "", inner.strip())
    for ln in inner.splitlines():
        s = ln.strip()
        if s.startswith("#") and not s.startswith("#!"):
            return s.lstrip("#").strip()[:200]
    m = re.search(r"\bclass\s+(\w+)", inner)
    if m:
        return f"Класс {m.group(1)}"[:200]
    m = re.search(r"\bdef\s+(\w+)", inner)
    if m:
        return f"Функция {m.group(1)}"[:200]
    first = inner.splitlines()[0].strip() if inner else ""
    if 8 <= len(first) <= 120:
        return first[:200]
    return "Фрагмент кода"


def infer_diagram_title(mermaid: str) -> str:
    raw = (mermaid or "").strip()
    if not raw:
        return "Схема"
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("%%"):
            title = s.lstrip("%").strip()
            if title:
                return title[:200]
    m = re.search(r'subgraph\s+["\']?([^"\'\]\n{]+)', raw, re.I)
    if m:
        return m.group(1).strip()[:200]
    m = re.search(r"title\s*:\s*(.+)$", raw, re.I | re.M)
    if m:
        return m.group(1).strip()[:200]
    return "Схема"


def _ensure_code_titles(codes: list[CodeAsset]) -> list[CodeAsset]:
    out: list[CodeAsset] = []
    for asset in codes:
        title = (asset.title or "").strip() or infer_code_title(asset.code)
        lang = (asset.language or "").strip() or _infer_code_language(asset.code)
        out.append(asset.model_copy(update={"title": title, "language": lang}))
    return out


def _ensure_diagram_titles(diagrams: list[DiagramAsset]) -> list[DiagramAsset]:
    out: list[DiagramAsset] = []
    for asset in diagrams:
        title = (asset.title or "").strip() or infer_diagram_title(asset.mermaid)
        out.append(asset.model_copy(update={"title": title}))
    return out


def _code_from_block(block: NodeContentBlock) -> list[CodeAsset]:
    if block.code_assets:
        return list(block.code_assets)
    out: list[CodeAsset] = []
    for i, snippet in enumerate(block.code_snippets or [], start=1):
        text = (snippet or "").strip()
        if text and is_likely_code_snippet(text):
            title = infer_code_title(text)
            lang = _infer_code_language(text)
            out.append(CodeAsset(id=f"code-{i}", title=title, language=lang, code=text))
    return out


def _refs_with_ids(refs: list[RichReferenceItem]) -> list[RichReferenceItem]:
    """Стабильные уникальные card-N (LLM часто шлёт card-1 на каждую ссылку)."""
    out: list[RichReferenceItem] = []
    used: set[str] = set()
    for r in refs or []:
        aid = (r.asset_id or "").strip()
        if not aid or aid in used:
            aid = _next_id("card", out)
        used.add(aid)
        out.append(r.model_copy(update={"asset_id": aid}))
    return out


def hydrate_content_diagrams_from_articles(
    content: NodeContentBlock,
    node: NodeDataInput,
    curriculum_id: str,
    *,
    extra_urls: list[str] | None = None,
) -> NodeContentBlock:
    """Схемы из article_diagrams (VLM) → content.diagrams на ноде."""
    from knowledge_engine.services.article_diagram_context import (
        build_diagram_assets_for_node,
    )
    from knowledge_engine.ui.run_log import trace

    incoming = build_diagram_assets_for_node(
        node,
        curriculum_id,
        max_diagrams=_MAX_DIAGRAMS,
        extra_urls=extra_urls,
    )
    if not incoming:
        return content

    diagrams = _diagrams_from_block(content)
    known = {_norm_mermaid(d.mermaid) for d in diagrams if d.mermaid}
    added = 0
    for asset in incoming:
        norm = _norm_mermaid(asset.mermaid)
        if not norm or norm in known:
            continue
        known.add(norm)
        diagrams.append(
            asset.model_copy(
                update={
                    "id": _next_id("diagram", diagrams),
                    "mermaid": norm,
                },
            )
        )
        added += 1
    if not added:
        return content
    diagrams = diagrams[:_MAX_DIAGRAMS]
    latest = diagrams[-1].mermaid if diagrams else _norm_mermaid(content.diagram)
    trace(
        f"NODE_DIVE hydrate diagrams ✓ | node={node.node_id} "
        f"added={added} total={len(diagrams)}"
    )
    return content.model_copy(
        update={
            "diagrams": diagrams,
            "diagram": latest,
        }
    )


def resolve_referenced_diagram(
    content: NodeContentBlock,
    diagram_id: str | None,
) -> DiagramAsset | None:
    """
    Match catalog id / diagram-N / 1-based index against existing content.diagrams.

    Never invents Mermaid; unknown id → None.
    """
    raw = (diagram_id or "").strip()
    if not raw:
        return None
    diagrams = _diagrams_from_block(content)
    if not diagrams:
        return None

    key = raw.lower()
    # Exact asset id
    for d in diagrams:
        aid = (d.id or "").strip()
        if aid and aid.lower() == key:
            return d
    # diagram:diagram-N or [diagram:diagram-N]
    m = re.search(r"diagram[-_]?(\d+)", key, re.I)
    if m:
        n = int(m.group(1))
        for d in diagrams:
            aid = (d.id or "").strip()
            if aid.lower() == f"diagram-{n}":
                return d
        if 1 <= n <= len(diagrams):
            return diagrams[n - 1]
    # Bare integer index
    if key.isdigit():
        n = int(key)
        if 1 <= n <= len(diagrams):
            return diagrams[n - 1]
    return None


def merge_content_assets(
    prev: NodeContentBlock,
    *,
    referenced_diagram_id: str | None = None,
    code_snippets: list[str] | None = None,
    references: list[RichReferenceItem] | None = None,
    summary: str | None = None,
    # Deprecated: ignored — tutors must not supply raw Mermaid.
    diagram: str = "",
    diagram_title: str = "",
) -> NodeContentBlock:
    _ = diagram, diagram_title  # LLM Mermaid append path disabled
    diagrams = _diagrams_from_block(prev)
    codes = _code_from_block(prev)
    refs = _refs_with_ids(list(prev.references or []))

    selected = resolve_referenced_diagram(prev, referenced_diagram_id)
    if selected is not None:
        # Prefer selected as panel "current" without inventing a new asset.
        rest = [d for d in diagrams if (d.id or "") != (selected.id or "")]
        diagrams = [selected, *rest]
    elif (referenced_diagram_id or "").strip():
        from knowledge_engine.ui.run_log import trace

        trace(
            f"NODE_DIVE diagram ref ignore | id={referenced_diagram_id!r} "
            f"not in catalog ({len(diagrams)} assets)"
        )

    for snippet in filter_code_snippets(code_snippets):
        text = snippet.strip()
        if not text:
            continue
        if codes and codes[-1].code == text:
            continue
        codes.append(
            CodeAsset(
                id=_next_id("code", codes),
                title=infer_code_title(text),
                language=_infer_code_language(text),
                code=text[:12_000],
            )
        )
    codes = _ensure_code_titles(codes)[:_MAX_CODE]
    diagrams = _ensure_diagram_titles(diagrams)[:_MAX_DIAGRAMS]

    if references:
        by_url = {r.url.strip(): r for r in refs if (r.url or "").strip()}
        for r in references:
            url = (r.url or "").strip()
            if not url:
                continue
            if url in by_url:
                old = by_url[url]
                by_url[url] = old.model_copy(
                    update={
                        "title": (r.title or old.title or "").strip(),
                        "why_read": (r.why_read or old.why_read or "").strip(),
                        "key_focus": (r.key_focus or old.key_focus or "").strip(),
                        "read_time_minutes": r.read_time_minutes
                        or old.read_time_minutes,
                    }
                )
            else:
                aid = _next_id("card", refs)
                by_url[url] = r.model_copy(update={"asset_id": aid})
        refs = list(by_url.values())[:_MAX_CARDS]
        refs = _refs_with_ids(refs)

    if selected is not None:
        latest_diagram = _norm_mermaid(selected.mermaid)
    else:
        # No catalog selection: keep panel diagram as-is (do not invent / rewrite).
        latest_diagram = (prev.diagram or "").strip() or (
            _norm_mermaid(diagrams[0].mermaid) if diagrams else ""
        )
    legacy_snippets = [c.code for c in codes][:4]

    summary_val = (summary or "").strip() or prev.summary

    return NodeContentBlock(
        summary=summary_val,
        summary_html=prev.summary_html,
        diagram=latest_diagram,
        diagrams=diagrams,
        references=refs,
        code_snippets=legacy_snippets,
        code_assets=codes,
    )
