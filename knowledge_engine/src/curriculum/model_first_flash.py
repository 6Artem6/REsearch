"""Этап 1: Model-First DAG без веб-поиска и без внешних ссылок."""

from __future__ import annotations

import re
import uuid

from knowledge_engine.config import (
    CURRICULUM_MODEL_FIRST_MIN_NODES,
    CURRICULUM_MODEL_FIRST_TARGET_NODES,
    GEMINI_FLASH_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.llm_contracts.curriculum import (
    CurriculumDAGContract,
)
from knowledge_engine.schemas.llm_contracts.curriculum import (
    ModelFirstPayloadContract as _ModelFirstPayload,
)
from knowledge_engine.schemas.llm_contracts.curriculum import (
    NodeListContract,
    NodeListNodeContract,
)
from knowledge_engine.services.gemini_stateless import (
    gemini_reasoner_model_chain,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.curriculum.dag_validator import (
    CURRICULUM_DAG_REPAIR_PRESERVE_ANCHOR_TOPICS,
    validate_curriculum_dag_full,
)
from knowledge_engine.src.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumNode,
    LearningMaterials,
)
from knowledge_engine.src.curriculum.source_registry import (
    sync_route_sources_from_registry,
)
from knowledge_engine.ui.run_log import trace

_MODEL_FIRST_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a Senior IT Methodologist. Build a learning DAG (directed acyclic graph) "
    "EXCLUSIVELY from your own knowledge. The route topic (target_goal) is given in the "
    "user payload below.\n\n"
    "HARD RULES:\n"
    "1. Decompose the topic into **at least 8-12** independent learning nodes (steps/subtopics).\n"
    "2. **FORBIDDEN** to include URLs, links, learning_resources, whitelist sources.\n"
    "3. Nodes have no external materials: structure only (title, layer, category, "
    "brief_summary, core_concepts, prerequisites).\n"
    "4. Layers: foundation → advanced → sota; DAG with no cycles.\n"
    "5. **DAG topology (critical):** Do NOT build a simple linear sequence "
    "(A→B→C→D, one prerequisite per node). Build a **branching DAG**:\n"
    "   - Do NOT generate a simple linear sequence of nodes (A→B→C→D). Construct a "
    "true directed acyclic graph with branching.\n"
    "   - Early foundational concepts (e.g. embeddings and chunking) MUST be "
    "**independent parallel branches** from the root or a shared foundation parent.\n"
    "   - Advanced nodes should **merge** dependencies from multiple prior nodes "
    "(2+ prerequisites).\n"
    "   - At the foundation stage: at least **2-3 parallel branches** (nodes with no "
    "mutual dependency between sibling branches, or a shared parent with 2+ children).\n"
    "   - A node may have **multiple prerequisites** (multiple parents); one "
    "prerequisite may be the parent of **several** nodes.\n"
    "6. node_id — Latin snake_case; node text — Russian.\n"
    "7. Cover the path from basic concepts to advanced engineering points (edge cases, "
    "fault tolerance, distributed patterns).\n"
    "8. curriculum_id (slug), title, description — in Russian.\n"
    "9. **User anchor topics:** if target_goal lists specific topics "
    'in parentheses, comma-separated, or as a list (e.g. "Storage architecture '
    '(WAL, Ring Buffer, P99)"):\n'
    "   - You MUST weave each of these topics into the graph as separate nodes or key concepts.\n"
    "   - Build logical dependencies (prerequisites) around them: place basic topics earlier, "
    "advanced ones — in deeper layers of the graph.\n"
    "   - Preserve the substance and terminology of the suggested topics, adapting their names "
    "naturally to the course's engineering style.\n"
)
"""
RU (пояснение): legacy single-pass Model-First (без веб-поиска) — DAG
только по знаниям модели; target_goal передаётся в user payload
(_build_user_payload), не форматируется в system — кэш-friendly.
"""


# RU: Two-Pass Model-First (см. аудит изолированной ноды 'Хэш-индексы') —
# декомпозиция темы (Pass 1) и построение рёбер (Pass 2) разведены по двум
# отдельным авторегрессивным проходам вместо одного _MODEL_FIRST_SYSTEM
# вызова выше. Оба промпта на английском (модель точнее держит жёсткие
# контракты на английском, см. .cursor/rules/llm-system-prompts-english.mdc);
# текстовые поля самих нод (title/brief_summary/category/core_concepts)
# всё равно на русском по RUSSIAN_OUTPUT_RULE.
#
# Кэш-friendly структура (см. TUTOR_PROMPT_AND_UI_TEXT.md "BLOCK 1-3" /
# [[llm_contracts_and_prompts]]): system_instruction ниже — ЭТО BLOCK 1,
# роль+жёсткие правила, БАЙТ-В-БАЙТ одинаковый текст на каждый вызов, без
# .format()-плейсхолдеров внутри него. Раньше {target_goal}/{node_list}
# были вклеены прямо в system-строку — каждый вызов менял сам
# system_instruction, что убивало даже автоматическое префиксное
# кэширование Gemini (см. переменную часть теперь в user-payload функций
# _build_pass1_user_payload/_run_pass2 ниже — BLOCK 3, dynamic suffix).
#
# Явный explicit-cache API проекта (gemini_cache_manager.get_or_create_
# explicit_cache, layer1_context в run_gemini_structured_with_chain) здесь
# намеренно НЕ подключён: порог включения — GEMINI_CACHE_MIN_EST_TOKENS
# (32000 по умолчанию), а весь BLOCK 1 ниже — на два порядка меньше
# (~300-500 токенов). При текущем размере caches.create() был бы либо
# инертным skipped_below_threshold, либо пришлось бы переписывать сигнатуру
# общего _generate_once() (используется десятками других вызывающих) ради
# нулевого практического эффекта. Байт-стабильный BLOCK 1 сам по себе даёт
# автоматическое префиксное кэширование Gemini без единого дополнительного
# API-вызова — это и есть рабочий механизм на этом масштабе промпта.

MODEL_FIRST_PASS1_NODES_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a Senior IT Curriculum Architect. Decompose ONE topic into a flat "
    "list of learning nodes — content only, NO topology yet. Prerequisites are "
    "generated in a separate pass and must NOT appear in your output.\n\n"
    "HARD RULES:\n"
    "1. Decompose the topic into **8-12** independent, self-contained learning "
    "nodes (steps/subtopics).\n"
    "2. FORBIDDEN: URLs, links, learning_resources, whitelist sources, "
    "prerequisites, or any other topology field.\n"
    "3. Each node has ONLY: node_id, title, layer, category, brief_summary, "
    "core_concepts.\n"
    "4. Layers: foundation -> advanced -> sota. Cover the path from basic "
    "concepts to advanced engineering edge cases (failure modes, distributed "
    "patterns).\n"
    "5. node_id — snake_case Latin. ALL text fields (title, brief_summary, "
    "category, core_concepts) — Russian.\n"
    "6. curriculum_id (slug), title, description — Russian.\n"
    "7. User-pinned topics: if the topic (given below, in the user turn) names "
    "specific concepts in parentheses, comma-separated, or as a list (e.g. "
    '"Storage architecture (WAL, Ring Buffer, P99)"):\n'
    "   - Every one of them MUST become its own node or an explicit "
    "core_concept.\n"
    "   - Keep their terminology, adapted to engineering-course style.\n"
)

MODEL_FIRST_PASS2_EDGES_SYSTEM = (
    "You are a Senior IT Curriculum Architect. You receive a FIXED list of "
    "learning nodes (node_id + title, given below in the user turn) already "
    "decided in Pass 1 — do NOT rename, add, or drop any node. Your ONLY job "
    "is to build the prerequisite edges (topology) connecting them into a "
    "directed acyclic graph.\n\n"
    "HARD RULES:\n"
    "1. For EVERY node_id from the fixed list, output exactly one entry "
    "with its prerequisites (node_id list of direct parents; empty list "
    "ONLY for a genuine foundation root).\n"
    "2. Acyclic: prerequisites must never form a cycle, and a node can "
    "never be its own prerequisite.\n"
    "3. Branching & merging DAG — NOT a linear chain (A->B->C->D):\n"
    "   - Foundation layer: at least 2-3 parallel branches (independent "
    "foundation roots, or one foundation parent with 2+ children).\n"
    "   - Advanced/sota nodes should MERGE dependencies from multiple "
    "prior nodes (2+ prerequisites) where the topic genuinely builds on "
    "more than one prior concept.\n"
    "   - A node may have several prerequisites (several parents); a "
    "prerequisite may be the parent of several nodes.\n"
    "4. HARD CONNECTIVITY INVARIANT (highest priority — violating this "
    "fails the whole response): Every node MUST belong to a single "
    "connected graph. Every node MUST have in_degree + out_degree >= 1. "
    "Absolutely NO isolated/orphan nodes (degree == 0) are allowed. A "
    "foundation node without prerequisites MUST have at least one child "
    "node in advanced/sota layers — i.e. it MUST appear in at least one "
    "OTHER node's prerequisites list.\n"
    "5. Before finalizing, verify for EACH node_id from the fixed list: "
    "does it appear as a prerequisite of at least one other node, OR does "
    "it have at least one prerequisite of its own? If neither — fix it "
    "before responding.\n"
)


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
        concepts = [
            c.strip() for c in (raw.core_concepts or []) if c and str(c).strip()
        ][:8]
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
        f"### target_goal\n{inp.target_goal.strip()}",
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
    # Static system prompt (cache-friendly) — target_goal lives in the user
    # payload, not interpolated into system, so system_instruction is byte-
    # identical across every call regardless of the requested course topic.
    system = _MODEL_FIRST_SYSTEM
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
            hints.append(f"Слишком мало узлов ({len(graph.nodes)}); нужно ≥{min_n}.")
        hint = "\n".join(f"- {h}" for h in hints)
        hint += f"\n{CURRICULUM_DAG_REPAIR_PRESERVE_ANCHOR_TOPICS}"
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
        raise ValueError("Model-First: невалидный DAG: " + "; ".join(errors[:5]))
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


# ---------------------------------------------------------------------------
# Two-Pass Model-First (см. аудит изолированной ноды 'Хэш-индексы') —
# альтернативный вход, НЕ заменяет generate_model_first_graph выше и не
# подключён ни к одному вызывающему коду; переключение вызывающей стороны на
# этот путь — отдельный шаг после проверки на реальных прогонах.
# ---------------------------------------------------------------------------


def _build_pass1_user_payload(inp: CurriculumGenerateInput) -> str:
    # RU: BLOCK 3 (dynamic suffix) — вся переменная по вызову часть (тема,
    # уровень, глубина) живёт здесь, в user-payload, НЕ в system_instruction
    # (см. комментарий над MODEL_FIRST_PASS1_NODES_SYSTEM выше).
    target = CURRICULUM_MODEL_FIRST_TARGET_NODES
    parts = [
        f"### topic\n{inp.target_goal.strip()}",
        f"### user_level\n{inp.user_level.strip()}",
        f"### depth_level\n{inp.depth_level}",
        f"Создай **{target}** узлов (минимум {CURRICULUM_MODEL_FIRST_MIN_NODES}). "
        "Без URL и без источников. Топологию (prerequisites) НЕ указывай — "
        "это отдельный шаг.",
    ]
    if inp.depth_level == "Overview":
        parts.append("Overview: 8–10 узлов, шире обзор.")
    elif inp.depth_level == "Deep Mechanics":
        parts.append("Deep Mechanics: 10–14 узлов, больше advanced/sota.")
    return "\n\n".join(parts)


def _format_pass1_node_list(nodes: list[NodeListNodeContract]) -> str:
    return "\n".join(f"- {n.node_id} | layer={n.layer} | {n.title}" for n in nodes)


def _coerce_two_pass_nodes(
    node_list: NodeListContract,
    edges: CurriculumDAGContract,
) -> list[CurriculumNode]:
    edges_by_id: dict[str, list[str]] = {
        e.node_id: e.prerequisites for e in edges.nodes
    }
    nodes: list[CurriculumNode] = []
    for raw in node_list.nodes or []:
        nid = (raw.node_id or "").strip()
        if len(nid) < 2:
            continue
        concepts = [
            c.strip() for c in (raw.core_concepts or []) if c and str(c).strip()
        ][:8]
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
                    prerequisites=list(edges_by_id.get(nid, []))[:24],
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


def _graph_from_two_pass(
    inp: CurriculumGenerateInput,
    node_list: NodeListContract,
    edges: CurriculumDAGContract,
) -> CurriculumGraph:
    cid = (node_list.curriculum_id or "").strip()
    if not cid or not _SLUG_RE.match(cid):
        cid = _normalize_slug(cid, f"curriculum_{uuid.uuid4().hex[:12]}")
    nodes = _coerce_two_pass_nodes(node_list, edges)
    return CurriculumGraph(
        curriculum_id=cid,
        title=(node_list.title or "Учебный маршрут").strip()[:300],
        description=(node_list.description or inp.target_goal).strip()[:4000],
        total_nodes=len(nodes),
        curriculum_sources_registry=[],
        route_sources=[],
        nodes=nodes,
    )


def generate_model_first_graph_two_pass(
    inp: CurriculumGenerateInput,
    anchor: str,
) -> CurriculumGraph:
    """Two-Pass Model-First: Pass 1 — декомпозиция темы на ноды без
    топологии; Pass 2 — только prerequisites на уже зафиксированный список
    node_id из Pass 1. Разводит по разным авторегрессивным проходам две
    разные задачи (декомпозиция темы vs связность графа), вместо того чтобы
    модель изобретала рёбра параллельно с ещё не оконченной декомпозицией
    (корневая причина изолированных нод, см. аудит 'Хэш-индексы').

    Три независимых защитных слоя от orphan-нод/оторванных подграфов:
    1. Промпт Pass 2 — жёсткая текстовая инструкция про connectivity invariant.
    2. CurriculumDAGContract.model_validator — ловит нарушение сразу на
       парсинге ответа Pass 2 (до сборки графа), кидает ValueError, вызывающий
       код здесь ловит его и повторяет Pass 2 с текстом ошибки как
       repair_feedback.
    3. validate_curriculum_dag_full (через validate_curriculum_topology) —
       backstop уже после сборки полного CurriculumGraph, тот же repair-hint
       путь, что и для циклов/слоёв у generate_model_first_graph.
    """
    trace("CURRICULUM model_first_two_pass ▶ | Pass 1: nodes")
    # RU: system_instruction байт-в-байт статичен (BLOCK 1, см. комментарий
    # над MODEL_FIRST_PASS1_NODES_SYSTEM) — тема уходит в user-payload.
    node_list = run_gemini_structured_with_chain(
        GEMINI_FLASH_MODEL,
        MODEL_FIRST_PASS1_NODES_SYSTEM,
        _build_pass1_user_payload(inp),
        anchor,
        NodeListContract,
        "curriculum_generator / model_first_pass1_nodes",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=gemini_reasoner_model_chain(GEMINI_FLASH_MODEL),
    )

    min_n = CURRICULUM_MODEL_FIRST_MIN_NODES
    if len(node_list.nodes) < min_n:
        raise ValueError(f"Model-First Pass 1: узлов {len(node_list.nodes)} < {min_n}")
    trace(f"CURRICULUM model_first_two_pass ✓ | Pass 1: nodes={len(node_list.nodes)}")

    def _run_pass2(repair_hint: str = "") -> CurriculumDAGContract:
        # RU: system_instruction байт-в-байт статичен (BLOCK 1) для ЛЮБОЙ
        # темы — фиксированный список нод из Pass 1 уходит в user-payload
        # (BLOCK 3, dynamic suffix), не в system.
        user2 = (
            "### fixed_node_list\n"
            f"{_format_pass1_node_list(node_list.nodes)}\n\n"
            "Построй prerequisites для всех перечисленных узлов."
        )
        if repair_hint:
            user2 += f"\n\n### repair_feedback\n{repair_hint}"
        return run_gemini_structured_with_chain(
            GEMINI_FLASH_MODEL,
            MODEL_FIRST_PASS2_EDGES_SYSTEM,
            user2,
            anchor,
            CurriculumDAGContract,
            "curriculum_generator / model_first_pass2_edges",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
            models=gemini_reasoner_model_chain(GEMINI_FLASH_MODEL),
        )

    trace("CURRICULUM model_first_two_pass ▶ | Pass 2: edges")
    try:
        edges = _run_pass2()
    except Exception as exc:
        # Контрактный model_validator (orphan/weak-connectivity/referential
        # integrity в CurriculumDAGContract) кинул ValueError -> Pydantic
        # ValidationError -> _parse_structured (gemini_stateless.py)
        # заворачивает в RuntimeError. Ловим здесь и повторяем Pass 2 с
        # текстом ошибки как repair_feedback — тот же принцип, что и у
        # repair-цикла ниже, просто триггер раньше (на парсинге ответа).
        trace(f"CURRICULUM model_first_two_pass ▶ Pass 2 repair (contract) | {exc}")
        edges = _run_pass2(repair_hint=str(exc))

    graph = _graph_from_two_pass(inp, node_list, edges)
    errors = validate_curriculum_dag_full(graph)

    if errors:
        hint = "\n".join(f"- {h}" for h in errors)
        hint += f"\n{CURRICULUM_DAG_REPAIR_PRESERVE_ANCHOR_TOPICS}"
        trace(
            f"CURRICULUM model_first_two_pass ▶ Pass 2 repair (post-build) | "
            f"issues={len(errors)}"
        )
        edges = _run_pass2(repair_hint=hint)
        graph = _graph_from_two_pass(inp, node_list, edges)
        errors = validate_curriculum_dag_full(graph)

    if errors:
        raise ValueError(
            "Model-First Two-Pass: невалидный DAG: " + "; ".join(errors[:5])
        )

    graph = graph.model_copy(update={"total_nodes": len(graph.nodes)})
    graph = sync_route_sources_from_registry(graph)
    trace(
        f"CURRICULUM model_first_two_pass ✓ | nodes={graph.total_nodes} "
        f"registry=0 (источники на этапе grounding)"
    )
    return graph
