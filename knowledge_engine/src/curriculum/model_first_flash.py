"""Этап 1: Model-First DAG без веб-поиска и без внешних ссылок."""

from __future__ import annotations

import re
import uuid

from pydantic import BaseModel, Field

from knowledge_engine.config import (
    CURRICULUM_MODEL_FIRST_MIN_NODES,
    CURRICULUM_MODEL_FIRST_TARGET_NODES,
    GEMINI_FLASH_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.gemini_stateless import (
    gemini_reasoner_model_chain,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.curriculum.dag_validator import validate_curriculum_dag_full
from knowledge_engine.src.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumNode,
    LearningMaterials,
)
from knowledge_engine.src.curriculum.source_registry import sync_route_sources_from_registry
from knowledge_engine.ui.run_log import trace

_MODEL_FIRST_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — Senior IT-Методист. Построй учебный DAG (направленный ациклический граф) "
    "ИСКЛЮЧИТЕЛЬНО на своих знаниях.\n\n"
    "ТЕМА МАРШРУТА: {target_goal}\n\n"
    "ЖЁСТКИЕ ПРАВИЛА:\n"
    "1. Декомпозируй тему на **минимум 8–12** самостоятельных учебных нод (шагов/подтем).\n"
    "2. **ЗАПРЕЩЕНО** указывать URL, ссылки, learning_resources, whitelist-источники.\n"
    "3. Ноды без внешних материалов: только структура (title, layer, category, "
    "brief_summary, core_concepts, prerequisites).\n"
    "4. Слои: foundation → advanced → sota; DAG без циклов.\n"
    "5. **Топология DAG (критично):** НЕ строй простую линейную последовательность "
    "(A→B→C→D, одна prerequisite на каждую ноду). Построй **ветвящийся DAG**:\n"
    "   - Do NOT generate a simple linear sequence of nodes (A→B→C→D). Construct a "
    "true directed acyclic graph with branching.\n"
    "   - Early foundational concepts (e.g. embeddings and chunking) MUST be "
    "**independent parallel branches** from the root or a shared foundation parent.\n"
    "   - Advanced nodes should **merge** dependencies from multiple prior nodes "
    "(2+ prerequisites).\n"
    "   - На этапе foundation: минимум **2–3 параллельные ветки** (ноды без взаимных "
    "зависимостей между соседними ветками или общий родитель с 2+ детьми).\n"
    "   - Одна нода может иметь **несколько prerequisites** (несколько родителей); "
    "один prerequisite может быть родителем **нескольких** нод.\n"
    "6. node_id — snake_case латиница; текст нод — русский.\n"
    "7. Покрой путь от базовых концептов до сложных инженерных точек (edge cases, "
    "отказоустойчивость, распределённые паттерны).\n"
    "8. curriculum_id (slug), title, description — на русском.\n"
)


class _ModelFirstNode(BaseModel):
    node_id: str = ""
    title: str = ""
    layer: str = ""
    category: str = ""
    brief_summary: str = ""
    core_concepts: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)


class _ModelFirstPayload(BaseModel):
    curriculum_id: str = ""
    title: str = ""
    description: str = ""
    nodes: list[_ModelFirstNode] = Field(default_factory=list)


_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _normalize_slug(raw: str, fallback: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) < 3:
        return fallback
    if not s[0].isalpha():
        s = f"n_{s}"
    return s[:80]


def _layer_kind(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v == "sota":
        return "sota"
    if v == "advanced":
        return "advanced"
    return "foundation"


def _coerce_model_first_nodes(payload: _ModelFirstPayload) -> list[CurriculumNode]:
    nodes: list[CurriculumNode] = []
    for raw in payload.nodes or []:
        nid = (raw.node_id or "").strip()
        if len(nid) < 2:
            continue
        concepts = [c.strip() for c in (raw.core_concepts or []) if c and str(c).strip()][:8]
        if not concepts:
            concepts = ["ключевая тема"]
        brief = (raw.brief_summary or "").strip()
        if len(brief) < 10:
            brief = (raw.title or nid)[:1200]
        if len(brief) < 10:
            brief = f"Шаг маршрута: {raw.title or nid}"
        try:
            nodes.append(
                CurriculumNode(
                    node_id=nid,
                    title=(raw.title or nid)[:300],
                    layer=_layer_kind(raw.layer),
                    category=(raw.category or "Архитектура")[:200],
                    brief_summary=brief[:1200],
                    core_concepts=concepts,
                    prerequisites=list(raw.prerequisites or [])[:24],
                    mapped_source_ids=[],
                    primary_source_id="",
                    learning_goal=brief[:600],
                    learning_materials=LearningMaterials(),
                    learning_resources=[],
                    resource_urls=[],
                    source_ref=None,
                    node_curriculum_breakdown=None,
                    node_risk_kind="BASE",
                    grounding_status="model_only",
                )
            )
        except Exception:
            continue
    return nodes


def _graph_from_payload(
    inp: CurriculumGenerateInput,
    payload: _ModelFirstPayload,
) -> CurriculumGraph:
    cid = (payload.curriculum_id or "").strip()
    if not cid or not _SLUG_RE.match(cid):
        cid = _normalize_slug(cid, f"curriculum_{uuid.uuid4().hex[:12]}")
    nodes = _coerce_model_first_nodes(payload)
    return CurriculumGraph(
        curriculum_id=cid,
        title=(payload.title or "Учебный маршрут").strip()[:300],
        description=(payload.description or inp.target_goal).strip()[:4000],
        total_nodes=len(nodes),
        curriculum_sources_registry=[],
        route_sources=[],
        nodes=nodes,
    )


def _build_user_payload(inp: CurriculumGenerateInput, repair_hint: str = "") -> str:
    target = CURRICULUM_MODEL_FIRST_TARGET_NODES
    parts = [
        f"### user_level\n{inp.user_level.strip()}",
        f"### depth_level\n{inp.depth_level}",
        f"Создай **{target}** узлов (минимум {CURRICULUM_MODEL_FIRST_MIN_NODES}). "
        "Без URL и без источников. "
        "Топология: ветвящийся DAG (2+ параллельные foundation-ветки, merge-ноды с 2+ prerequisites), "
        "НЕ линейная цепочка.",
    ]
    if inp.depth_level == "Overview":
        parts.append("Overview: 8–10 узлов, шире обзор.")
    elif inp.depth_level == "Deep Mechanics":
        parts.append("Deep Mechanics: 10–14 узлов, больше advanced/sota.")
    if repair_hint:
        parts.append(f"### repair_feedback\n{repair_hint}")
    return "\n\n".join(parts)


def generate_model_first_graph(
    inp: CurriculumGenerateInput,
    anchor: str,
) -> CurriculumGraph:
    trace("CURRICULUM model_first ▶ | Flash DAG без веб-поиска")
    system = _MODEL_FIRST_SYSTEM.format(target_goal=inp.target_goal.strip())
    user = _build_user_payload(inp)

    payload = run_gemini_structured_with_chain(
        GEMINI_FLASH_MODEL,
        system,
        user,
        anchor,
        _ModelFirstPayload,
        "curriculum_generator / model_first",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=gemini_reasoner_model_chain(GEMINI_FLASH_MODEL),
    )
    graph = _graph_from_payload(inp, payload)
    errors = validate_curriculum_dag_full(graph)
    min_n = CURRICULUM_MODEL_FIRST_MIN_NODES

    if errors or len(graph.nodes) < min_n:
        hints: list[str] = list(errors)
        if len(graph.nodes) < min_n:
            hints.append(
                f"Слишком мало узлов ({len(graph.nodes)}); нужно ≥{min_n}."
            )
        hint = "\n".join(f"- {h}" for h in hints)
        trace(f"CURRICULUM model_first ▶ repair | issues={len(hints)}")
        payload = run_gemini_structured_with_chain(
            GEMINI_FLASH_MODEL,
            system,
            _build_user_payload(inp, repair_hint=hint),
            anchor,
            _ModelFirstPayload,
            "curriculum_generator / model_first_repair",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
            models=gemini_reasoner_model_chain(GEMINI_FLASH_MODEL),
        )
        graph = _graph_from_payload(inp, payload)
        errors = validate_curriculum_dag_full(graph)

    if errors:
        raise ValueError(
            "Model-First: невалидный DAG: " + "; ".join(errors[:5])
        )
    if len(graph.nodes) < min_n:
        raise ValueError(
            f"Model-First: после repair узлов {len(graph.nodes)} < {min_n}"
        )

    graph = graph.model_copy(update={"total_nodes": len(graph.nodes)})
    graph = sync_route_sources_from_registry(graph)
    trace(
        f"CURRICULUM model_first ✓ | nodes={graph.total_nodes} "
        f"registry=0 (источники на этапе grounding)"
    )
    return graph
