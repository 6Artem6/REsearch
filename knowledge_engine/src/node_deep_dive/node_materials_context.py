"""Блок [AVAILABLE NODE MATERIALS] для промптов тьютора и dense-лекции."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.curriculum.schemas import LearningMaterials
from knowledge_engine.src.node_deep_dive.content_assets import (
    _code_from_block,
    _diagrams_from_block,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    NodeContentBlock,
    NodeDataInput,
    RichReferenceItem,
)

_TAG = "[AVAILABLE NODE MATERIALS]"
_MAX_CODE_CHARS = 2400
_MAX_DIAGRAM_CHARS = 2000
_MAX_ITEMS = 12


def _clip(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def _first_code_line(code: str) -> str:
    for line in (code or "").splitlines():
        s = line.strip()
        if s:
            return s[:120]
    return ""


def _diagram_excerpt(mermaid: str) -> str:
    lines = [ln.strip() for ln in (mermaid or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    head = " ".join(lines[:4])
    return _clip(head, 280)


def _format_learning_materials(lm: LearningMaterials | None) -> list[str]:
    if lm is None:
        return []
    out: list[str] = []
    p = getattr(lm, "primary_whitelist_source", None)
    if p is not None:
        name = (getattr(p, "source_name", "") or "").strip()
        chapter = (getattr(p, "chapter_or_article", "") or "").strip()
        label = name or chapter or "primary_whitelist_source"
        detail = chapter if chapter and chapter != name else ""
        line = f"- Material (whitelist): **{label}**"
        if detail:
            line += f" — {detail[:200]}"
        out.append(line)
    return out


def _format_learning_resources(resources: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for item in resources or []:
        if not isinstance(item, dict):
            continue
        title = (
            str(item.get("title") or item.get("name") or item.get("label") or "")
        ).strip()
        url = str(item.get("url") or item.get("href") or "").strip()
        if url:
            label = title or url[:80]
            out.append(
                f"- Material URL [{label}]: {url} (только для JSON references, не в tutor_message)"
            )
        elif title:
            out.append(f"- Material: {title[:200]}")
        if len(out) >= 6:
            break
    return out


def _format_resource_urls(urls: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in urls or []:
        url = str(raw or "").strip()
        if len(url) < 8:
            continue
        out.append(f"- Material URL: {url}")
        if len(out) >= 6:
            break
    return out


def _format_references(refs: list[RichReferenceItem] | None) -> list[str]:
    out: list[str] = []
    for r in refs or []:
        title = (r.title or r.source_name or "").strip()
        url = (r.url or "").strip()
        aid = (r.asset_id or "").strip()
        prefix = f"[{aid}] " if aid else ""
        if url:
            label = title or url[:80]
            out.append(f"- Material {prefix}**{label}** | registry_url: {url}")
        elif title:
            out.append(f"- Material {prefix}**{title}**")
        if len(out) >= 6:
            break
    return out


def format_available_node_materials_block(
    node: NodeDataInput,
    content: NodeContentBlock | None = None,
) -> str:
    """
    Явный перечень кода, диаграмм и ресурсов ноды для «экскурсии» тьютора.
    Пустая строка, если материалов нет.
    """
    lines: list[str] = []
    block = content or NodeContentBlock()

    for asset in _code_from_block(block)[:6]:
        code = _clip(asset.code, _MAX_CODE_CHARS)
        if not code:
            continue
        title = (asset.title or "").strip()
        label = f" [{asset.id}]" if asset.id else ""
        if title:
            label = f" [{asset.id}: {title}]" if asset.id else f" [{title}]"
        hint = _first_code_line(code)
        lines.append(f"- Code{label}: `{hint}` …\n" f"```\n{code}\n```")

    for diag in _diagrams_from_block(block)[:6]:
        excerpt = _diagram_excerpt(diag.mermaid)
        if not excerpt and not diag.mermaid:
            continue
        title = (diag.title or diag.id or "diagram").strip()
        lines.append(f"- Diagram [{diag.id}: {title}]: {excerpt}")
        if diag.mermaid:
            lines.append(f"```mermaid\n{_clip(diag.mermaid, _MAX_DIAGRAM_CHARS)}\n```")

    lines.extend(_format_learning_materials(node.learning_materials))
    lines.extend(_format_learning_resources(node.learning_resources))
    lines.extend(_format_resource_urls(node.resource_urls))
    lines.extend(_format_references(block.references))

    summary = (block.summary or "").strip()
    if summary and len(summary) >= 80 and len(lines) < _MAX_ITEMS:
        lines.append(f"- Panel summary (excerpt): {_clip(summary, 400)}")

    if not lines:
        return ""

    trimmed = lines[:_MAX_ITEMS]
    body = "\n".join(trimmed)
    return (
        f"{_TAG}\n"
        "Тьютор ОБЯЗАН вести объяснение как экскурсию по этим материалам ноды "
        "(не абстрактный учебник). Минимум 1–2 прямых обращения с разбором элементов.\n"
        f"{body}\n"
        f"[END {_TAG.strip('[]')}]\n"
    )


def node_materials_present(
    node: NodeDataInput,
    content: NodeContentBlock | None = None,
) -> bool:
    return bool(format_available_node_materials_block(node, content).strip())
