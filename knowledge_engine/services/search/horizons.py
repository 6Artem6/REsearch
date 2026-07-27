"""Три горизонты поиска: SOTA, Infra, Prod (не путать с категориями Trade-off матрицы)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import ROUTER_MODEL
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_ROUTER_RULE
from knowledge_engine.schemas import CSAbstraction, HorizonQuerySet
from knowledge_engine.ui.logger import set_status


class SearchHorizon(str, Enum):
    """Временной/контекстный горизонт источников для discovery."""

    SOTA = "sota"
    INFRA = "infra"
    PROD = "prod"


HORIZON_LABELS: Mapping[SearchHorizon, str] = {
    SearchHorizon.SOTA: "SOTA — papers, benchmarks, surveys",
    SearchHorizon.INFRA: "Infra — деплой, стек, observability",
    SearchHorizon.PROD: "Prod — инциденты, failure modes, postmortem",
}

HORIZON_PROVIDERS: Mapping[SearchHorizon, tuple[str, ...]] = {
    SearchHorizon.SOTA: ("arxiv", "semantic_scholar", "crossref", "consensus"),
    SearchHorizon.INFRA: ("google_meta",),
    SearchHorizon.PROD: ("habr", "google_meta"),
}

_QUERY_FOCUS: Mapping[SearchHorizon, str] = {
    SearchHorizon.SOTA: "survey benchmark arxiv",
    SearchHorizon.INFRA: "deployment observability docker k8s",
    SearchHorizon.PROD: "production incident failure postmortem",
}


def _normalize_abstractions(abstractions: Sequence[Any]) -> list[CSAbstraction]:
    out: list[CSAbstraction] = []
    for item in abstractions:
        if isinstance(item, CSAbstraction):
            out.append(item)
        else:
            out.append(CSAbstraction.model_validate(item))
    return out


def _template_horizon_queries(
    user_problem: str,
    context_constraints: str,
    abstractions: Sequence[Any],
) -> dict[SearchHorizon, str]:
    abs_models = _normalize_abstractions(abstractions)
    concepts = " ".join(dict.fromkeys(a.cs_concept for a in abs_models[:3])).strip()
    problem_short = " ".join(user_problem.strip().split())[:90]
    constraints = " ".join(context_constraints.strip().split())[:50]

    queries: dict[SearchHorizon, str] = {}
    queries[SearchHorizon.SOTA] = " ".join(
        p
        for p in [concepts or problem_short[:60], _QUERY_FOCUS[SearchHorizon.SOTA]]
        if p
    )
    queries[SearchHorizon.INFRA] = " ".join(
        p
        for p in [problem_short[:70], constraints, _QUERY_FOCUS[SearchHorizon.INFRA]]
        if p
    )
    queries[SearchHorizon.PROD] = " ".join(
        p
        for p in [
            problem_short[:70],
            constraints,
            _QUERY_FOCUS[SearchHorizon.PROD],
        ]
        if p
    )
    return queries


def _clip_words(text: str, max_words: int = 12) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])


def build_horizon_queries(
    user_problem: str,
    context_constraints: str,
    abstractions: Sequence[Any],
) -> dict[SearchHorizon, str]:
    """Лаконичные запросы: роутер 1.5B, fallback — короткий шаблон."""
    abs_models = _normalize_abstractions(abstractions)
    abs_text = "\n".join(f"- {a.cs_concept}: {a.title}" for a in abs_models[:5])
    fallback = _template_horizon_queries(
        user_problem, context_constraints, abstractions
    )

    set_status("[Router] короткие запросы SOTA / Infra / Prod…")
    structured = structured_chat(ROUTER_MODEL, HorizonQuerySet, temperature=0.05)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Сформируй три КОРОТКИЕ поисковые строки (не больше 12 слов каждая). "
            "Для papers допустимы английские термины (arxiv); для prod/infra — по контексту. "
            "Без перечисления десятков ключевых слов. Разные углы: "
            "sota=исследования, infra=деплой/стек, prod=продакшен/сбои."
        )
    )
    human = HumanMessage(
        content=(
            f"Задача: {user_problem}\n"
            f"Ограничения: {context_constraints or '(нет)'}\n"
            f"CS-абстракции:\n{abs_text or '(нет)'}\n"
            f"Шаблон для ориентира:\n"
            f"sota: {fallback[SearchHorizon.SOTA]}\n"
            f"infra: {fallback[SearchHorizon.INFRA]}\n"
            f"prod: {fallback[SearchHorizon.PROD]}"
        )
    )
    try:
        result: HorizonQuerySet | None = invoke_logged(
            structured, [system, human], "horizons / HorizonQuerySet"
        )
        if result is None:
            raise ValueError("None")
        queries = {
            SearchHorizon.SOTA: _clip_words(result.sota.strip()),
            SearchHorizon.INFRA: _clip_words(result.infra.strip()),
            SearchHorizon.PROD: _clip_words(result.prod.strip()),
        }
    except Exception:
        queries = fallback

    queries = {h: _clip_words(q, 12) for h, q in queries.items()}

    for h, q in queries.items():
        set_status(f"[Horizon {h.value}] {q[:100]}")
    return queries
