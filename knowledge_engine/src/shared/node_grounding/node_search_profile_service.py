"""NodeSearchProfileService — Node Grounding Gate, Этап 0: поисковый

профиль ОДНОЙ ноды. Add-only, параллельно ``taxonomy_service.py`` (тот —
для целого курса при создании через Штурвал; этот — для одной ноды при её
открытии). Не импортирует и не трогает ``targeted_node_search.py``.

ИСПРАВЛЕНО после ревью по docs/SOURCE_POOL.md ("Единая логика источников —
не дублировать в pipeline") и docs/ACADEMIC_AND_CONSENSUS.md: academic-часть
профиля больше НЕ генерируется отдельным Lite-промптом здесь — это был
дубль уже существующего Academic Query Architect
(``lite_search_pipeline.build_academic_search_plan``), который вдобавок
даёт структурированные ``arxiv_params`` (ti:/abs:/categories/годы), а мой
собственный промпт отдавал только голую строку. Теперь эта функция отвечает
ТОЛЬКО за то, для чего в проекте нет готового архитектора — Habr-хабы/теги/
ключевики под конкретную ноду (общего "Habr hub architect" в docs не
описано) — плюс Exa keywords как лёгкая, не завязанная на полный DEEP
multi-vector pipeline, подсказка для нашего облегчённого Gate 1.
Academic query + ``arxiv_params`` — через ``build_node_academic_plan``
ниже, тонкую обёртку над ``build_academic_search_plan``.
"""

from __future__ import annotations

import json

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.lite_search_pipeline import (
    AcademicSearchPlan,
    _lite_structured,
    build_academic_search_plan,
)
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NodeSearchProfile,
)

_PROFILE_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a Search Profile Architect for a single curriculum node (the "
    '"Node Grounding Gate" — a lighter, per-node analog of the Steering '
    "funnel). Given the node's title/summary/concepts, produce, in ONE "
    "response, seeds for two channels — Habr and general web (Exa). Do NOT "
    "produce academic queries here — that is handled separately by the "
    "project's existing Academic Query Architect.\n\n"
    "Return JSON matching NodeSearchProfile (leave academic_queries empty "
    "— it is filled in elsewhere):\n"
    "- habr_hubs: 2-6 Habr company-hub slugs plausibly covering this "
    'node (bare slugs, e.g. "yandex", "avito").\n'
    "- habr_tags: 2-5 Habr tags (Russian).\n"
    "- habr_keywords: 2-5 Russian search keywords for Habr full-text search.\n"
    "- exa_keywords: 2-5 English search keywords/phrases for general "
    "engineering-blog web search (Exa)."
)


def _anchor_profile(curriculum_id: str, node_id: str) -> str:
    return f"node_gate_profile:{curriculum_id}:{node_id}"[:500]


def _heuristic_profile(title: str, core_concepts: list[str]) -> NodeSearchProfile:
    words = [
        w.strip(".,:;!?()")
        for w in (title or "").split()
        if len(w.strip(".,:;!?()")) > 2
    ]
    return NodeSearchProfile(
        habr_hubs=[],
        habr_tags=words[:3],
        habr_keywords=words[:5],
        exa_keywords=list(core_concepts or [])[:5],
        academic_queries=[],
    )


async def generate_node_search_profile(
    *,
    node_title: str,
    node_summary: str,
    core_concepts: list[str],
    include_academic: bool,
    curriculum_id: str = "",
    node_id: str = "",
    anchor: str | None = None,
) -> NodeSearchProfile:
    """Контекст ноды → NodeSearchProfile (Habr + Exa seeds), один Flash Lite

    вызов. ``include_academic`` больше не влияет на промпт этой функции
    (academic_queries сюда не входит вообще, см. модуль docstring) —
    параметр сохранён для обратной совместимости вызова из эндпоинта, но
    сейчас не используется здесь; академия — через ``build_node_academic_plan``.
    Fail-open: при сбое — эвристический профиль из title/core_concepts."""
    title = (node_title or "").strip()
    if not title:
        return NodeSearchProfile()

    payload = {
        "node_title": title[:300],
        "node_summary": (node_summary or "").strip()[:800],
        "core_concepts": list(core_concepts or [])[:8],
    }
    trace(f"NODE_GATE profile ▶ | node={node_id}")
    try:
        out = await _lite_structured(
            _PROFILE_SYSTEM,
            json.dumps(payload, ensure_ascii=False),
            anchor or _anchor_profile(curriculum_id, node_id),
            NodeSearchProfile,
            "curriculum / node_gate_profile",
        )
    except Exception as exc:
        trace(f"NODE_GATE profile fallback | heuristic | {exc}")
        return _heuristic_profile(title, core_concepts)

    profile = (
        out
        if isinstance(out, NodeSearchProfile)
        else NodeSearchProfile.model_validate(out)
    )
    if profile.academic_queries:
        # Промпт явно просит не заполнять это поле — но вероятностно, не
        # доверяем инструкции без проверки (тот же принцип, что везде в
        # этом проекте).
        profile = profile.model_copy(update={"academic_queries": []})
    return profile


async def build_node_academic_plan(
    node_title: str,
    node_summary: str,
    *,
    anchor: str | None = None,
) -> AcademicSearchPlan:
    """Тонкая обёртка над УЖЕ СУЩЕСТВУЮЩИМ Academic Query Architect

    (``lite_search_pipeline.build_academic_search_plan`` — см.
    docs/ACADEMIC_AND_CONSENSUS.md, раздел 1) — переиспользуется как есть,
    не дублируется. Даёт и ``academic_query_en`` (для отображения на Gate 1
    и для Semantic Scholar/SearXNG science/Consensus), и структурированные
    ``arxiv_params`` (ti:/abs:/categories/годы), которых собственный Lite
    промпт этого модуля не производил."""
    goal = f"{(node_title or '').strip()}. {(node_summary or '').strip()}".strip(". ")
    return await build_academic_search_plan(goal, anchor=anchor)


__all__ = ["generate_node_search_profile", "build_node_academic_plan"]
