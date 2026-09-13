"""TaxonomyService — Этап 2 Штурвала, шаг 1: расширение target_goal в

поисковые сиды для Light Discovery. Один Flash Lite вызов с
Pydantic-валидированным выводом. Add-only: ничего из Autopilot-пайплайнов
(``generator.py``, ``targeted_node_search.py``) не трогается и не
импортируется отсюда. План/статус этапов — в
docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, контракты Gate 1/2 — в
``src.curriculum.steering_contracts``.

``tags``/``keywords`` попадают в ``TaxonomyDiscoveryResponse``, который
пользователь видит на Gate 1 (см. steering_contracts.py) — поэтому это
user-facing поля (RUSSIAN_OUTPUT_RULE). ``company_hubs`` — технические
идентификаторы (bare slug/hostname), по ним Light Discovery строит
``site:`` ограничения поиска, поэтому они остаются как есть, без перевода.

``source_policy`` (``practical_only``/``academic_only``/``hybrid``) управляет
и промптом (инструкция модели), и — на случай, если модель всё равно
предложит академический домен — детерминированным пост-фильтром
``company_hubs`` (``_filter_hubs_by_policy``): для ``practical_only`` это
zero-tolerance требование (см. отладку "утечка Consensus/arXiv/S2 в
Practical"), полагаться только на промпт недостаточно.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.lite_search_pipeline import (
    _lite_structured,
)

# Известные академические домены — используются и здесь (фильтр company_hubs),
# и в light_discovery_service.py (фильтр итоговых candidate_articles, на
# случай unrestricted-fallback Exa-поиска вне заданных хабов).
ACADEMIC_DOMAIN_MARKERS = ("arxiv.org", "semanticscholar.org", "consensus.app")


def is_academic_domain(value: str) -> bool:
    v = (value or "").strip().lower()
    return any(marker in v for marker in ACADEMIC_DOMAIN_MARKERS)


class TaxonomySeed(BaseModel):
    """Поисковые сиды под target_goal: теги/хабы компаний/ключевики для Gate 1 и Light Discovery."""

    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags derived from the target goal (Russian).",
    )
    company_hubs: list[str] = Field(
        default_factory=list,
        description=(
            "Company/vendor engineering-blog hubs relevant to the target goal, "
            "as bare slugs/hostnames (not translated, not full URLs)."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Search keywords/phrases to seed the Light Discovery pass (Russian).",
    )


_TAXONOMY_SYSTEM_BASE = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a Search Taxonomy Architect for a technical curriculum discovery "
    'pipeline (the "Steering" funnel). Given a learner\'s target goal, expand it '
    "into search seeds. tags/keywords are shown directly to the user before they "
    "approve the next step — company_hubs are technical identifiers consumed by "
    "the next search stage, not shown as prose.\n\n"
    "Return JSON matching TaxonomySeed:\n"
    "- tags: 3-8 short topic tags in Russian (technology/concept names).\n"
    "- company_hubs: 3-10 hubs whose content plausibly covers this goal. Each "
    "entry is a bare slug or hostname (not translated, no URL scheme, no free "
    "text).\n"
    "- keywords: 3-8 concrete search keywords/phrases in Russian to seed "
    "discovery queries."
)

_POLICY_HUB_RULES = {
    "practical_only": (
        "\n\ncompany_hubs SOURCE POLICY: practical_only — ONLY engineering "
        "blogs / vendor tech blogs / practitioner hub sites (e.g. "
        '"netflix", "habr.com/ru/companies/yandex"). NEVER suggest academic '
        "paper databases or research-index sites (arXiv, Semantic Scholar, "
        "Consensus, journal sites, university research pages) — this mode is "
        "explicitly practice-only, zero tolerance for academic sources."
    ),
    "academic_only": (
        "\n\ncompany_hubs SOURCE POLICY: academic_only — ONLY academic/"
        'research hubs (e.g. "arxiv.org", "semanticscholar.org", '
        '"consensus.app"). Do NOT suggest company engineering blogs.'
    ),
    "hybrid": (
        "\n\ncompany_hubs SOURCE POLICY: hybrid — mix engineering blogs AND "
        "academic/research hubs, both are welcome."
    ),
}


def _taxonomy_system_for_policy(source_policy: str) -> str:
    policy = (source_policy or "hybrid").strip().lower()
    return _TAXONOMY_SYSTEM_BASE + _POLICY_HUB_RULES.get(
        policy, _POLICY_HUB_RULES["hybrid"]
    )


def _filter_hubs_by_policy(hubs: list[str], source_policy: str) -> list[str]:
    """Детерминированный пост-фильтр поверх промпта — LLM иногда всё равно

    предлагает академический хаб даже при явном запрете в системном
    промпте; для ``practical_only`` это должно быть невозможно (zero
    tolerance), поэтому не полагаемся только на инструкцию."""
    policy = (source_policy or "hybrid").strip().lower()
    if policy == "practical_only":
        return [h for h in hubs if not is_academic_domain(h)]
    if policy == "academic_only":
        return [h for h in hubs if is_academic_domain(h)]
    return hubs


def _anchor_taxonomy(goal: str) -> str:
    return f"steering_taxonomy:{(goal or '').strip()[:500]}"


def _heuristic_taxonomy_seed(target_goal: str) -> TaxonomySeed:
    words = [
        w.strip(".,:;!?()")
        for w in (target_goal or "").split()
        if len(w.strip(".,:;!?()")) > 2
    ]
    return TaxonomySeed(tags=words[:5], company_hubs=[], keywords=words[:5])


async def generate_taxonomy_seed(
    target_goal: str,
    *,
    source_policy: str = "hybrid",
    anchor: str | None = None,
) -> TaxonomySeed:
    """target_goal → TaxonomySeed (шаг 1 Штурвала, перед Light Discovery).

    ``source_policy`` определяет, какие company_hubs допустимы (см. модуль
    docstring) — и в промпте, и повторно детерминированным фильтром на
    выходе. Fail-open: если Lite-вызов падает, возвращает эвристический сид
    из слов самой цели — воронка Gate 1 не должна рваться из-за сбоя одного
    вызова (эвристика не производит доменов, фильтровать нечего).
    """
    goal = (target_goal or "").strip()
    if not goal:
        return TaxonomySeed()

    trace(
        f"STEERING taxonomy ▶ | Lite expand goal={goal[:80]!r} "
        f"policy={source_policy}"
    )
    try:
        out = await _lite_structured(
            _taxonomy_system_for_policy(source_policy),
            json.dumps({"target_goal": goal[:1200]}, ensure_ascii=False),
            anchor or _anchor_taxonomy(goal),
            TaxonomySeed,
            "curriculum / steering_taxonomy",
        )
    except Exception as exc:
        trace(f"STEERING taxonomy fallback | heuristic seed | {exc}")
        return _heuristic_taxonomy_seed(goal)

    seed = out if isinstance(out, TaxonomySeed) else TaxonomySeed.model_validate(out)
    filtered_hubs = _filter_hubs_by_policy(seed.company_hubs, source_policy)
    if filtered_hubs != seed.company_hubs:
        trace(
            "STEERING taxonomy hub_filter | policy="
            f"{source_policy} dropped={len(seed.company_hubs) - len(filtered_hubs)}"
        )
        seed = seed.model_copy(update={"company_hubs": filtered_hubs})
    return seed


__all__ = ["TaxonomySeed", "generate_taxonomy_seed"]
