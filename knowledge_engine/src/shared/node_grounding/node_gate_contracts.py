"""Node Grounding Gate — контракты Этапа 0 (профиль поиска) и Gate 1

(паспорта кандидатов) для гейта на открытие DEEP-ноды. Add-only,
параллельно ``steering_contracts.py`` (Штурвал — гейты на уровне ЦЕЛОГО
курса при создании; этот файл — гейт на уровне ОДНОЙ ноды при её
открытии/перегрунде, отдельный, более лёгкий поток). Не импортирует и не
трогает ``targeted_node_search.py``/``engine.py`` — см.
docs/STEERING_AND_TOPIC_QNA_ROADMAP.md (Node Grounding Gate).

Реализованы Этап 0 (единый поисковый профиль), Этап 1 → Gate 1 (сбор +
дедуп + паспорта) и Этап 2 → Gate 2 (BGE-M3+Cross-Encoder top-K чанков +
Gemma точечный дайджест). Этап 3 (полный Map-Reduce и запись в pgvector
после Gate 2) — следующий шаг, контрактов для него здесь ещё нет.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NODE_GATE1_POOL_MIN = 5
NODE_GATE1_POOL_MAX = 10
NODE_GATE1_MAX_APPROVED = NODE_GATE1_POOL_MAX
# RU: на Gate 1 в дайджест можно отправить хоть весь пул кандидатов.
NODE_GATE2_MAX_APPROVED = 4
# RU: из дайджестов Gate 2 в полный пайплайн (Map-Reduce ingest) проходит
# не больше NODE_GATE2_MAX_APPROVED статей — он тяжёлый (минуты на статью).


class NodeSearchProfile(BaseModel):
    """Единый поисковый профиль ОДНОЙ ноды — один Flash Lite вызов (Этап 0)."""

    habr_hubs: list[str] = Field(
        default_factory=list,
        description=(
            "Habr topic-hub slugs (habr.com/ru/hubs/<slug>/) plausibly "
            "covering this node, e.g. machine_learning, postgresql."
        ),
    )
    """ RU: slug-и тематических хабов Habr; лента хаба — RSS
    /ru/rss/hubs/<slug>/articles/ (slug компании принимается как запасной)."""
    habr_tags: list[str] = Field(
        default_factory=list, description="Habr tags to seed search."
    )
    habr_keywords: list[str] = Field(
        default_factory=list, description="Habr full-text search keywords."
    )
    exa_keywords: list[str] = Field(
        default_factory=list,
        description="English keywords/phrases for general engineering-blog web search (Exa).",
    )
    academic_queries: list[str] = Field(
        default_factory=list,
        description=(
            "English academic-style queries (Consensus/arXiv/Semantic Scholar). "
            "Empty when academic sources are not enabled for this node."
        ),
    )


class NodeCandidatePassport(BaseModel):
    """Сухой паспорт кандидата для Gate 1: тема | стек | 1-2 предложения сути."""

    url: str = Field(..., description="Candidate source URL.")
    title: str = Field(..., description="Article/paper title.")
    source_tier: str = Field(
        ...,
        description="habr | exa | consensus | arxiv | semantic_scholar.",
    )
    topic: str = Field(..., description="One short topic label (Russian).")
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Technologies/tools mentioned, if any (English names as-is).",
    )
    gist: str = Field(
        ...,
        description="1-2 dry sentences of substance from the intro/conclusion (Russian).",
    )


class NodeGate1DiscoveryResponse(BaseModel):
    """Результат Этапа 0+1 — то, что видит пользователь на Gate 1 (одна нода)."""

    curriculum_id: str
    node_id: str
    profile: NodeSearchProfile
    candidates: list[NodeCandidatePassport] = Field(default_factory=list)


class NodeGate1ApprovePayload(BaseModel):
    """Решение пользователя на Gate 1 для одной ноды: статьи для дайджеста
    (до всего пула кандидатов)."""

    curriculum_id: str = Field(..., min_length=3, max_length=80)
    node_id: str = Field(..., min_length=2, max_length=80)
    approved_urls: list[str] = Field(
        ...,
        max_length=NODE_GATE1_MAX_APPROVED,
        description=f"Subset of candidates[].url, at most {NODE_GATE1_MAX_APPROVED}.",
    )
    """ RU: сколько угодно кандидатов из пула — лимит полного пайплайна
    стоит на Gate 2 (NODE_GATE2_MAX_APPROVED)."""


NODE_GATE2_COARSE_K = 16
NODE_GATE2_TOP_K_CHUNKS = 6
# RU: BGE-M3 грубо отбирает NODE_GATE2_COARSE_K абзацев по косинусу к теме
# ноды, затем Cross-Encoder (score_relevance_pairs) уточняет до
# NODE_GATE2_TOP_K_CHUNKS самых полезных — см. docs/RAG_GATEWAY_MODULE_3.md.


class NodeDigestItem(BaseModel):
    """Точечный дайджест ОДНОЙ статьи (Gemma, Этап 2) — не сухой паспорт Gate 1."""

    url: str = Field(..., description="Copied verbatim from Gate 1 approved_urls.")
    title: str = Field(..., description="Article/paper title.")
    architecture: str = Field(
        ..., description="Архитектура/подход, описанные в статье (Russian)."
    )
    practical_case: str = Field(
        ..., description="Практический кейс/результат применения (Russian)."
    )
    limitations: str = Field(
        ..., description="Ограничения/недостатки/условия применимости (Russian)."
    )


class NodeGate2DigestResponse(BaseModel):
    """Результат Этапа 2 — то, что видит пользователь на Gate 2 (одна нода)."""

    curriculum_id: str
    node_id: str
    digests: list[NodeDigestItem] = Field(default_factory=list)


class NodeGate2ApprovePayload(BaseModel):
    """Финальное согласие пользователя на Gate 2 — инжест по этим URL (Этап 3)."""

    curriculum_id: str = Field(..., min_length=3, max_length=80)
    node_id: str = Field(..., min_length=2, max_length=80)
    approved_urls: list[str] = Field(
        ...,
        max_length=NODE_GATE2_MAX_APPROVED,
        description=(
            "Subset of digests[].url the user gives final ingest consent to, "
            f"at most {NODE_GATE2_MAX_APPROVED}."
        ),
    )


class NodeGateState(BaseModel):
    """Незавершённый гейт ноды — переживает перезагрузку страницы.

    Пишет worker после discover (Gate 1) и digest (Gate 2), удаляет
    finalize. ``/node/grounding-discover`` отдаёт сохранённое состояние
    вместо повторного поиска; поля Gate 1 совпадают с
    ``NodeGate1DiscoveryResponse``."""

    curriculum_id: str
    node_id: str
    status: Literal["awaiting_gate_1", "awaiting_gate_2"] = Field(
        ..., description="Which gate the user has to pass next."
    )
    profile: NodeSearchProfile = Field(default_factory=NodeSearchProfile)
    candidates: list[NodeCandidatePassport] = Field(default_factory=list)
    digests: list[NodeDigestItem] = Field(
        default_factory=list, description="Gate 2 digests (awaiting_gate_2 only)."
    )
    approved_urls: list[str] = Field(
        default_factory=list, description="URLs approved on Gate 1."
    )
    updated_at: str = ""


__all__ = [
    "NODE_GATE1_POOL_MIN",
    "NODE_GATE1_POOL_MAX",
    "NODE_GATE1_MAX_APPROVED",
    "NODE_GATE2_MAX_APPROVED",
    "NODE_GATE2_COARSE_K",
    "NODE_GATE2_TOP_K_CHUNKS",
    "NodeSearchProfile",
    "NodeCandidatePassport",
    "NodeGate1DiscoveryResponse",
    "NodeGateState",
    "NodeGate1ApprovePayload",
    "NodeDigestItem",
    "NodeGate2DigestResponse",
    "NodeGate2ApprovePayload",
]
