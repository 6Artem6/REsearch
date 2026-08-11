"""Модуль 1 — генератор учебного графа (Curriculum Generator).

Импорт `generate_curriculum_graph` — только из `knowledge_engine.src.curriculum.generator`
(не из пакета), чтобы не циклично тянуть node_deep_dive при загрузке schemas.
"""

from knowledge_engine.src.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
    CurriculumSourceRegistryEntry,
    LearningMaterials,
    NodeCurriculumBreakdown,
    NodeSourceRef,
    PrimaryWhitelistSource,
    RouteSourceEntry,
)

__all__ = [
    "CurriculumGenerateInput",
    "CurriculumGraph",
    "CurriculumNode",
    "CurriculumSourceRegistryEntry",
    "CurriculumSearchHit",
    "LearningMaterials",
    "NodeCurriculumBreakdown",
    "NodeSourceRef",
    "PrimaryWhitelistSource",
    "RouteSourceEntry",
]
