"""Каталог схем панели Materials — единственный допустимый список для ссылок в тексте."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.src.node_deep_dive.content_assets import _diagrams_from_block
from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock


@dataclass(frozen=True)
class DiagramCatalogEntry:
    index: int
    asset_id: str
    title: str


def build_diagram_catalog(
    content: NodeContentBlock | None,
) -> list[DiagramCatalogEntry]:
    """Список схем как в UI (normalizeNodeMaterials / content.diagrams)."""
    entries: list[DiagramCatalogEntry] = []
    for i, diag in enumerate(
        _diagrams_from_block(content or NodeContentBlock()), start=1
    ):
        mermaid = (diag.mermaid or "").strip()
        if not mermaid:
            continue
        aid = (diag.id or f"diagram-{i}").strip() or f"diagram-{i}"
        title = (diag.title or "").strip()
        entries.append(DiagramCatalogEntry(index=i, asset_id=aid, title=title))
    return entries


def format_diagram_catalog_block(entries: list[DiagramCatalogEntry]) -> str:
    """Pinned-блок: пустой каталог = жёсткий запрет на [Diagram N]."""
    if not entries:
        return (
            "### DIAGRAM_CATALOG (панель Materials — единственный источник истины)\n"
            "Схем в интерфейсе: **0**.\n"
            "ЗАПРЕЩЕНО в tutor_message / lecture_body:\n"
            "- любые «[Diagram 1]», «[Diagram 2]», «[Diagram N: …]», diagram-N;\n"
            "- формулировки «как показано на схеме / диаграмме» с номером;\n"
            "- выдуманные названия и подписи схем;\n"
            "- сырой Mermaid в JSON (`referenced_diagram_id` = null).\n"
            "Объясняй текстом; при необходимости — короткий ASCII/markdown-tree без ссылок на Diagram N.\n"
        )
    lines = [
        "### DIAGRAM_CATALOG (панель Materials — цитируй ТОЛЬКО эти схемы)",
        f"Всего схем: {len(entries)}. Номера вне 1…{len(entries)} запрещены.",
    ]
    for e in entries:
        title = e.title if e.title else "(без подписи)"
        lines.append(f"- {e.asset_id} → [Diagram {e.index}]: «{title}»")
    lines.append(
        "Правила цитирования:\n"
        "- Допустимо: `[Diagram N]` или `[diagram:diagram-N]` только для N из списка выше.\n"
        "- JSON `referenced_diagram_id` must be one of the asset_id values below, or null.\n"
        "- ЗАПРЕЩЕНО: писать сырой Mermaid в любом JSON-поле (server резолвит код из каталога).\n"
        "- ЗАПРЕЩЕНО: придумывать подпись после номера (не пиши «[Diagram 1: длинный выдуманный заголовок]»).\n"
        "- ЗАПРЕЩЕНО: ссылаться на Diagram 2/3, если в каталоге только одна схема.\n"
        "- Анализируй mermaid из [AVAILABLE NODE MATERIALS] / PINNED_DIAGRAMS только если id совпадает с каталогом."
    )
    return "\n".join(lines)
