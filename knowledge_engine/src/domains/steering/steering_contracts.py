"""Add-only контракты «Штурвала» (Control Axis) и Topic Q&A (Interaction Axis).

Этап 1 мастер-плана (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md): только
данные/схемы, никакой логики.
Ничего из существующих модулей — Autopilot-пайплайны (`src/curriculum/generator.py`,
`targeted_node_search.py`), текущий `WorkJobStatus`, действующие эндпоинты
(`api/routes/curriculum.py`, `/node/init|chat|verify`) — этим файлом НЕ
изменяется и не импортирует его: все классы ниже параллельны существующим
и пока никем не используются. `CurriculumGenerateInput` импортируется только
на чтение (наследование), его собственные поля/поведение не трогаются.
Topic Q&A (`InteractionAxis`, `interaction_axis`) с этого шага уже живой —
поле теперь прямо на `NodeDeepDiveRequest` (node_deep_dive/schemas.py), а не
в отдельном `TopicQnaNodeDeepDiveRequest` здесь (см.
docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Interaction Axis подключён").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from knowledge_engine.src.domains.curriculum.schemas import CurriculumGenerateInput

# --- Оси управления -------------------------------------------------------

ControlAxis = Literal["autopilot", "steering"]
# RU: autopilot — текущий Map-Reduce без пауз; steering — Штурвал с Gate 1/2.

SteeringMode = Literal["per_node", "standalone_digest"]
# RU: два независимых сценария внутри control_axis="steering" (см.
# docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Разделение на Mode 1/Mode 2"):
# - per_node — граф строится Model-First (Flash Lite, БЕЗ предварительного
#   поиска статей), заземление DEEP-нод точечное/ленивое при их открытии
#   через уже существующий Node Grounding Gate (node_gate_contracts.py) —
#   до NODE_GATE1_MAX_APPROVED источников на ноду через реальный Map-Reduce,
#   ровно как у Autopilot по умолчанию (generate_curriculum_graph →
#   CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED). Никакого Gate 1/2 на уровне
#   курса — Штурвал управляет заземлением НОД, а не поиском для курса.
# - standalone_digest — тот же Taxonomy+Light Discovery → Gate 1 → Batch
#   Digest → Gate 2, но ПОСЛЕ Gate 2 — реальный тяжёлый Map-Reduce по
#   approved-статьям (тот же движок, что Autopilot/Node Grounding Gate),
#   итог — валидный CurriculumGraph (≥3 ноды, см.
#   services/steering_topic_node_service.py).

InteractionAxis = Literal["lecture_self_check", "topic_qna"]
# RU: lecture_self_check — текущая лекция+самопроверка; topic_qna — новый
# Socratic/Topic Q&A формат.


# --- Gate 1: Light Discovery & Hub Selection ------------------------------


class CandidateArticleMeta(BaseModel):
    """Карточка кандидата-статьи для решения пользователя на Gate 1."""

    url: str = Field(..., description="Source URL of the candidate article.")
    title: str = Field(..., description="Article title as discovered.")
    source_hub: str = Field(
        ...,
        description=(
            "Discovered hub/domain the article belongs to "
            "(company blog, docs hub, aggregator)."
        ),
    )
    lead_paragraph: str = Field(
        ...,
        description="Intro/lead paragraph or snippet used for the quick relevance check.",
    )


class TaxonomyDiscoveryResponse(BaseModel):
    """Результат шагов 1–2 Штурвала (TaxonomyService + Light Discovery) — то, что видит пользователь на Gate 1."""

    tags: list[str] = Field(
        default_factory=list, description="Topic tags generated for the target goal."
    )
    company_hubs: list[str] = Field(
        default_factory=list,
        description="Company/vendor hubs and blogs discovered for the topic.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Search keywords used to seed the discovery pass.",
    )
    candidate_articles: list[CandidateArticleMeta] = Field(
        default_factory=list,
        description="Candidate articles surfaced for Gate 1 review.",
    )


class Gate1ApprovePayload(BaseModel):
    """Решение пользователя на Gate 1 — какие URL допускаются к дайджесту (Gate 2)."""

    work_job_id: str = Field(
        ..., description="WorkJob id awaiting Gate 1 approval (AWAITING_GATE_1)."
    )
    approved_urls: list[str] = Field(
        ...,
        description="Subset of candidate_articles[].url approved by the user to proceed to Gate 2.",
    )


# --- Gate 2: Surface Digest ------------------------------------------------


class SurfaceDigestItem(BaseModel):
    """Поверхностный дайджест одной статьи (проблема/решение/стек), без запуска Map-Reduce."""

    url: str = Field(..., description="Source URL this digest was generated from.")
    title: str = Field(..., description="Article title.")
    company_or_author: str = Field(
        ..., description="Publishing company or author, as far as it can be inferred."
    )
    problem_solved: str = Field(
        ..., description="Problem the article addresses, one to two sentences."
    )
    main_tech_stack: list[str] = Field(
        default_factory=list,
        description="Main technologies/tools mentioned in the article.",
    )
    two_sentence_summary: str = Field(
        ..., description="Dry two-sentence summary of the article, no narrative filler."
    )


class SurfaceDigestResponse(BaseModel):
    """Результат шага 3 Штурвала (Batch Digest Generator) — то, что видит пользователь на Gate 2."""

    digests: list[SurfaceDigestItem] = Field(
        default_factory=list,
        description="Surface digests for the Gate 1-approved articles.",
    )


class Gate2ApprovePayload(BaseModel):
    """Финальное решение пользователя на Gate 2 — какие URL идут в реальный

    Map-Reduce (Режим 2, см. services/steering_topic_node_service.py)."""

    work_job_id: str = Field(
        ..., description="WorkJob id awaiting Gate 2 approval (AWAITING_GATE_2)."
    )
    final_approved_urls: list[str] = Field(
        ...,
        description="Subset of digests[].url approved by the user for the final Map-Reduce.",
    )


# --- Расширенные запросы генерации/сессий (add-only, backward-compatible) -


class SteeringCurriculumGenerateInput(CurriculumGenerateInput):
    """`CurriculumGenerateInput` + выбор Control Axis.

    Существующий `CurriculumGenerateInput` не меняется. Вызовы без нового
    поля (весь текущий Autopilot-код) получают ровно прежнее поведение —
    `control_axis` по умолчанию `"autopilot"`.
    """

    control_axis: ControlAxis = Field(
        default="autopilot",
        description="Autopilot (existing Map-Reduce) vs Steering (new gated funnel).",
    )
    steering_mode: SteeringMode = Field(
        default="per_node",
        description=(
            "Only meaningful when control_axis='steering': per_node "
            "(graph first, lazy per-node grounding via Node Grounding Gate) "
            "vs standalone_digest (Gate 1/2 funnel → one merged document, "
            "no graph)."
        ),
    )


__all__ = [
    "ControlAxis",
    "SteeringMode",
    "InteractionAxis",
    "CandidateArticleMeta",
    "TaxonomyDiscoveryResponse",
    "Gate1ApprovePayload",
    "SurfaceDigestItem",
    "SurfaceDigestResponse",
    "Gate2ApprovePayload",
    "SteeringCurriculumGenerateInput",
]
