"""Генерация учебного DAG: Targeted Grounding (default), Search-First или legacy Reasoner."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from knowledge_engine.config import (
    CURRICULUM_SEARCH_FIRST_ENABLED,
    CURRICULUM_SEARCH_MIN_HITS,
    CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED,
    GEMINI_REASONER_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    gemini_reasoner_model_chain,
    is_gemini_available,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.curriculum.dag_validator import validate_curriculum_dag
from knowledge_engine.src.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumReasonerPayload,
)
from knowledge_engine.services.curriculum_whitelist_prompt import curriculum_whitelist_prompt_block
from knowledge_engine.src.curriculum.search_first_flash import generate_curriculum_search_first
from knowledge_engine.src.curriculum.search_prestep import (
    assign_source_ids,
    collect_curriculum_source_hits,
    search_hits_as_prompt_json,
)
from knowledge_engine.src.curriculum.source_material_pipeline import (
    enrich_search_hits_with_extracts,
    summarize_whitelist_blog_hits,
)
from knowledge_engine.src.curriculum.source_enrichment import enrich_curriculum_whitelist_sources
from knowledge_engine.src.curriculum.targeted_node_grounding import (
    generate_curriculum_targeted_grounding,
)
from knowledge_engine.ui.run_log import trace

_CURRICULUM_SYSTEM_BASE = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — Senior IT-Архитектор, формирующий учебный маршрут (DAG).\n"
    "Спроектируй направленный ациклический граф учебных узлов под цель пользователя.\n"
    "НЕ генерируй лекции и длинные объяснения — только структура графа и источники из Whitelist.\n\n"
    "{whitelist_block}\n"
    "ПРАВИЛА ПОСТРОЕНИЯ:\n"
    "1. **Топология (prerequisites):** если узел B опирается на A, node_id A в prerequisites B. "
    "НЕ линейная цепочка: ветвящийся DAG — несколько параллельных foundation-веток, "
    "merge-ноды с 2+ prerequisites, один родитель — несколько детей.\n"
    "2. **Слои (layer):** 'foundation' или 'sota'.\n"
    "3. **core_concepts:** 3–5 тезисов границ темы.\n"
    "4. **DAG:** без циклов; prerequisites только на существующие node_id.\n"
    "5. **Покрытие:** узлы foundation и sota.\n"
    "6. **node_id:** латиница snake_case.\n"
    "7. **learning_materials.primary_whitelist_source** — ОБЯЗАТЕЛЬНО на каждую ноду: "
    "source_name, chapter_or_article, core_concepts (из whitelist-источника).\n"
    "8. **learning_resources** / resource_urls — опционально, только whitelist URL.\n"
    "9. **Порядок:** от простого к сложному; без изолированных узлов.\n\n"
    "Формат: JSON по схеме. Текст — русский; node_id — латиница.\n"
    "Ориентир: 10–15 узлов.\n"
)


def _curriculum_system_instruction() -> str:
    return _CURRICULUM_SYSTEM_BASE.format(
        whitelist_block=curriculum_whitelist_prompt_block(),
    )


_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _anchor(goal: str) -> str:
    return f"curriculum:{goal.strip()[:500]}"


def _normalize_slug(raw: str, fallback: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) < 3:
        return fallback
    if not s[0].isalpha():
        s = f"n_{s}"
    return s[:80]


def _finalize_graph(payload: CurriculumReasonerPayload) -> CurriculumGraph:
    cid = (payload.curriculum_id or "").strip()
    if not cid or not _SLUG_RE.match(cid):
        cid = _normalize_slug(cid, f"curriculum_{uuid.uuid4().hex[:12]}")
    graph = CurriculumGraph(
        curriculum_id=cid,
        title=(payload.title or "Учебный маршрут").strip(),
        description=(payload.description or "").strip(),
        total_nodes=len(payload.nodes),
        nodes=payload.nodes,
    )
    return graph


def _build_user_payload(
    inp: CurriculumGenerateInput,
    repair_hint: str = "",
) -> str:
    parts = [
        f"### target_goal\n{inp.target_goal.strip()}",
        f"### user_level\n{inp.user_level.strip()}",
        f"### depth_level\n{inp.depth_level}",
        "Сформируй curriculum_id (короткий slug латиница), title, description и nodes.",
    ]
    if inp.depth_level == "Overview":
        parts.append("depth_level=Overview: меньше узлов (6–10), шире обзор.")
    elif inp.depth_level == "Standard":
        parts.append("depth_level=Standard: около 10–12 узлов.")
    else:
        parts.append("depth_level=Deep Mechanics: 12–15 узлов, глубже prerequisites.")
    if repair_hint:
        parts.append(f"### repair_feedback\n{repair_hint}")
    return "\n\n".join(parts)


def _invoke_reasoner(
    system_instruction: str,
    user_payload: str,
    anchor: str,
    label: str,
) -> CurriculumReasonerPayload:
    return run_gemini_structured_with_chain(
        GEMINI_REASONER_MODEL,
        system_instruction,
        user_payload,
        anchor,
        CurriculumReasonerPayload,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=gemini_reasoner_model_chain(),
    )


def _generate_legacy_reasoner(
    model_in: CurriculumGenerateInput,
    anchor: str,
) -> CurriculumGraph:
    trace(f"CURRICULUM ▶ legacy Reasoner DAG | goal={model_in.target_goal[:80]}…")
    payload = _invoke_reasoner(
        _curriculum_system_instruction(),
        _build_user_payload(model_in),
        anchor,
        "curriculum_generator / draft",
    )
    graph = _finalize_graph(payload)
    errors = validate_curriculum_dag(graph)

    if errors:
        hint = (
            "Предыдущий граф отклонён валидатором. Исправь:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\nСохрани curriculum_id и исправь только проблемные связи/слои."
        )
        trace(f"CURRICULUM ▶ repair | errors={len(errors)}")
        payload = _invoke_reasoner(
            _curriculum_system_instruction(),
            _build_user_payload(model_in, repair_hint=hint),
            anchor,
            "curriculum_generator / repair",
        )
        graph = _finalize_graph(payload)
        errors = validate_curriculum_dag(graph)

    if errors:
        trace(f"CURRICULUM ✗ DAG invalid | {errors[0]}")
        raise ValueError(
            "Не удалось построить валидный учебный граф: " + "; ".join(errors[:5])
        )

    graph = enrich_curriculum_whitelist_sources(model_in, graph, anchor)
    return graph


def generate_curriculum_graph(
    inp: CurriculumGenerateInput | dict[str, Any],
) -> CurriculumGraph:
    """
    Targeted Node Grounding (default): Model-First → Risk → Search → Grounding.
    Legacy Search-First: CURRICULUM_SEARCH_FIRST_ENABLED=true.
    Fallback: Reasoner DAG + Lite whitelist enrich.
    """
    if not is_gemini_available():
        raise GeminiUnavailableError(
            "Gemini недоступен для Curriculum Generator"
        )

    if isinstance(inp, dict):
        data = dict(inp)
        data.setdefault("user_level", "Intermediate/Advanced")
        data.setdefault("depth_level", "Standard")
        data.setdefault("generation_mode", "fast")
        model_in = CurriculumGenerateInput.model_validate(data)
    else:
        model_in = inp

    anchor = _anchor(model_in.target_goal)
    t_gen = time.monotonic()
    trace(
        f"CURRICULUM ▶ генерация | mode={model_in.generation_mode} "
        f"goal={model_in.target_goal[:80]}…"
    )

    graph: CurriculumGraph | None = None

    if CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED:
        trace(
            f"CURRICULUM pipeline | Targeted Node Grounding (lazy) "
            f"policy={model_in.source_policy}"
        )
        graph = generate_curriculum_targeted_grounding(model_in, anchor)
    elif CURRICULUM_SEARCH_FIRST_ENABLED:
        policy = model_in.source_policy
        hits = collect_curriculum_source_hits(
            model_in.target_goal,
            depth_level=model_in.depth_level,
            generation_mode=model_in.generation_mode,
            source_policy=policy,
        )
        if hits and len(hits) < CURRICULUM_SEARCH_MIN_HITS:
            trace(
                f"CURRICULUM search thin pool | hits={len(hits)} "
                f"< CURRICULUM_SEARCH_MIN_HITS={CURRICULUM_SEARCH_MIN_HITS} "
                "— мало материала для развёрнутого DAG"
            )
        if hits:
            hits = summarize_whitelist_blog_hits(hits, model_in.target_goal)
            hits = enrich_search_hits_with_extracts(hits, model_in.target_goal)
            hits = assign_source_ids(hits)
            parsed_json = search_hits_as_prompt_json(hits)
            graph = generate_curriculum_search_first(
                model_in,
                hits,
                parsed_json,
                anchor,
            )
        else:
            trace("CURRICULUM search prestep ⊘ | fallback legacy Reasoner")

    if graph is None:
        graph = _generate_legacy_reasoner(model_in, anchor)

    trace(
        f"CURRICULUM ✓ nodes={graph.total_nodes} id={graph.curriculum_id} "
        f"elapsed={time.monotonic() - t_gen:.1f}s"
    )
    return graph
