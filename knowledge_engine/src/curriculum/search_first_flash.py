"""Flash: маршрут ВОКРУГ собранных источников и выдержек."""

from __future__ import annotations

import re
import uuid

from knowledge_engine.config import GEMINI_FLASH_MODEL, GEMINI_RPM_PAUSE_SEC
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.llm_contracts.curriculum import (
    FlashCurriculumPayloadContract as _FlashCurriculumPayload,
)
from knowledge_engine.schemas.llm_contracts.curriculum import (
    FlashExpansionPatchContract as _FlashExpansionPatch,
)
from knowledge_engine.schemas.llm_contracts.curriculum import (
    FlashSourceRefContract as _FlashSourceRef,
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
    CurriculumExpansionPatch,
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
    CurriculumSourceRegistryEntry,
    LearningMaterials,
    NodeCurriculumBreakdown,
    NodeSourceRef,
)
from knowledge_engine.src.curriculum.search_prestep import (
    _normalize_url_key,
    search_hit_index,
)
from knowledge_engine.src.curriculum.source_registry import (
    sync_route_sources_from_registry,
    validate_curriculum_source_links,
)
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    resolve_source_provenance,
)
from knowledge_engine.ui.run_log import trace

_METHODIST_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — Senior IT-Методист и Главный Архитектор Обучения.\n\n"
    "ТЕМА МАРШРУТА: {target_goal}\n\n"
    "ВХОДНЫЕ МАТЕРИАЛЫ И ВЫДЕРЖКИ (Собранный пайплайн Consensus / Playwright):\n"
    "=== НАЧАЛО МАТЕРИАЛОВ ===\n"
    "{parsed_sources_with_extracts_json}\n"
    "=== КОНЕЦ МАТЕРИАЛОВ ===\n\n"
    "ИНСТРУКЦИЯ ПО ПРОЕКТИРОВАНИЮ МАРШРУТА:\n"
    "1. Построй ВЗАИМОСВЯЗАННЫЙ логический маршрут (ветвящийся DAG, не линейная цепочка "
    "A→B→C), взяв за ОСНОВУ материалы выше. Foundation: 2+ параллельные ветки; advanced/sota: "
    "merge с 2+ prerequisites где уместно.\n"
    "2. Каждая нода — конкретный шаг обучения, опирающийся на выдержки из источников.\n"
    "3. Разбей материалы на этапы: foundation → advanced → sota.\n"
    "4. Для каждой ноды заполни source_ref (source_id, url из входа, relevant_extracts — цитаты "
    "из key_extracts) и node_curriculum_breakdown (key_concepts, architectural_focus).\n"
    "5. Источники с source_tier=consensus — академические статьи (приоритет); "
    "gemini_grounding / gemini_web / whitelist_blog / archive — инженерные блоги.\n"
    "6. curriculum_id (slug), title, description — русский; node_id — snake_case латиница.\n"
    "7. Если в реестре ≥5 источников — проектируй **8–15 нод** (foundation + advanced + sota), "
    "раскрывая обширную цель по шагам; не схлопывай маршрут в 2–3 ноды.\n"
    "ЗАПРЕЩЕНО: абстрактные темы без выдержек; сторонние URL; при богатом пуле — "
    "искусственно мало нод.\n"
    "8. **Опорные темы пользователя:** если в цели (target_goal) указаны конкретные темы "
    "в скобках, через запятую или списком (например, «Архитектура хранилищ (WAL, Ring Buffer, P99)»):\n"
    "   - Обязательно вплети каждую из этих тем в граф в виде отдельных нод или ключевых concepts.\n"
    "   - Выстрой вокруг них логичные зависимости (prerequisites): базовые темы размещай раньше, "
    "продвинутые — в глубоких слоях графа.\n"
    "   - Сохраняй суть и терминологию предложенных тем, органично адаптируя их названия "
    "под инженерный стиль курса.\n"
)


def coerce_expansion_patch_from_flash(
    raw: _FlashExpansionPatch,
    hits: list[CurriculumSearchHit],
    registry: list[CurriculumSourceRegistryEntry],
) -> CurriculumExpansionPatch:
    from knowledge_engine.src.curriculum.schemas import CurriculumExpansionEdge

    nodes = _coerce_nodes(
        _FlashCurriculumPayload(nodes=list(raw.new_nodes or [])),
        hits,
        registry,
    )
    edges: list[CurriculumExpansionEdge] = []
    for e in raw.new_edges or []:
        fr = (e.from_node_id or "").strip()
        to = (e.to_node_id or "").strip()
        if len(fr) < 2 or len(to) < 2:
            continue
        try:
            edges.append(
                CurriculumExpansionEdge(from_node_id=fr[:80], to_node_id=to[:80])
            )
        except Exception:
            continue
    return CurriculumExpansionPatch(new_nodes=nodes, new_edges=edges)


_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _norm_src_id(raw: str, index: int) -> str:
    s = (raw or "").strip()
    if re.match(r"^src_\d+$", s, re.I):
        return s.lower().replace("SRC_", "src_")
    if re.match(r"^S\d+$", s, re.I):
        return f"src_{s[1:]}"
    return f"src_{index}"


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


def _build_user_payload(inp: CurriculumGenerateInput, repair_hint: str = "") -> str:
    parts = [
        f"### user_level\n{inp.user_level.strip()}",
        f"### depth_level\n{inp.depth_level}",
        "Покрой все значимые выдержки из входных материалов; не оставляй источник без ноды.",
    ]
    if repair_hint:
        parts.append(f"### repair_feedback\n{repair_hint}")
    return "\n\n".join(parts)


def _registry_from_hits(
    hits: list[CurriculumSearchHit],
) -> list[CurriculumSourceRegistryEntry]:
    registry: list[CurriculumSourceRegistryEntry] = []
    for i, hit in enumerate(hits, start=1):
        sid = _norm_src_id(hit.source_id, i)
        matched, cat = resolve_source_provenance(hit.url)
        from urllib.parse import urlparse

        domain = (
            cat if cat != "open_candidate" else (urlparse(hit.url).netloc or "").lower()
        )
        extracts = list(hit.key_extracts or [])
        why = (hit.snippet or "")[:800]
        if extracts and not why:
            why = extracts[0][:800]
        registry.append(
            CurriculumSourceRegistryEntry(
                source_id=sid[:16],
                title=(hit.title or domain or sid)[:400],
                whitelist_domain=domain[:200],
                source_type="Article",
                url=hit.url[:2000],
                why_read=why,
                snippet=(hit.snippet or "")[:1200],
                key_extracts=extracts[:12],
                source_tier=(hit.source_tier or "")[:24],
            )
        )
    return registry[:20]


def _resolve_source_ref(
    raw: _FlashSourceRef,
    hits: list[CurriculumSearchHit],
    registry: list[CurriculumSourceRegistryEntry],
) -> NodeSourceRef | None:
    idx = search_hit_index(hits)
    reg_by_id = {e.source_id: e for e in registry}
    sid = _norm_src_id(raw.source_id, 1)
    url = (raw.url or "").strip()
    key = _normalize_url_key(url)
    hit = idx.get(key)
    entry = reg_by_id.get(sid)

    if not entry and sid in reg_by_id:
        entry = reg_by_id[sid]
    if not entry and hit:
        entry = next(
            (e for e in registry if _normalize_url_key(e.url) == key),
            None,
        )
    if not entry and registry:
        entry = registry[0]

    if not entry:
        return None

    extracts = [e.strip() for e in raw.relevant_extracts if e and str(e).strip()][:12]
    if not extracts:
        extracts = list(entry.key_extracts or [])[:8]
    if not extracts and entry.snippet:
        extracts = [entry.snippet[:800]]

    return NodeSourceRef(
        source_id=entry.source_id[:16],
        url=entry.url[:2000],
        relevant_extracts=extracts[:12],
    )


def _coerce_nodes(
    payload: _FlashCurriculumPayload,
    hits: list[CurriculumSearchHit],
    registry: list[CurriculumSourceRegistryEntry],
) -> list[CurriculumNode]:
    nodes: list[CurriculumNode] = []

    for raw in payload.nodes or []:
        nid = (raw.node_id or "").strip()
        if len(nid) < 2:
            continue

        source_ref = _resolve_source_ref(raw.source_ref, hits, registry)
        bd_raw = raw.node_curriculum_breakdown
        key_concepts = [
            c.strip() for c in (bd_raw.key_concepts or []) if c and str(c).strip()
        ][:24]
        arch_focus = (bd_raw.architectural_focus or "").strip()[:800]

        breakdown = None
        if key_concepts or arch_focus:
            breakdown = NodeCurriculumBreakdown(
                key_concepts=key_concepts or ["ключевая тема"],
                architectural_focus=arch_focus,
            )

        core = key_concepts[:8] if key_concepts else ["ключевая тема"]
        mapped: list[str] = []
        if source_ref and source_ref.source_id:
            mapped = [source_ref.source_id[:16]]

        brief = (raw.brief_summary or "").strip()
        if len(brief) < 10:
            brief = (arch_focus or raw.title or nid)[:1200]
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
                    core_concepts=core,
                    prerequisites=list(raw.prerequisites or [])[:24],
                    mapped_source_ids=mapped,
                    primary_source_id=mapped[0] if mapped else "",
                    learning_goal=arch_focus[:600],
                    learning_materials=LearningMaterials(),
                    learning_resources=[],
                    resource_urls=(
                        [source_ref.url] if source_ref and source_ref.url else []
                    ),
                    source_ref=source_ref,
                    node_curriculum_breakdown=breakdown,
                )
            )
        except Exception:
            continue

    return nodes


def _graph_from_flash(
    inp: CurriculumGenerateInput,
    payload: _FlashCurriculumPayload,
    hits: list[CurriculumSearchHit],
) -> CurriculumGraph:
    cid = (payload.curriculum_id or "").strip()
    if not cid or not _SLUG_RE.match(cid):
        cid = _normalize_slug(cid, f"curriculum_{uuid.uuid4().hex[:12]}")

    registry = _registry_from_hits(hits)
    nodes = _coerce_nodes(payload, hits, registry)
    if len(nodes) < 3:
        raise ValueError("Search-First Flash: меньше 3 валидных узлов")

    graph = CurriculumGraph(
        curriculum_id=cid,
        title=(payload.title or "Учебный маршрут").strip()[:300],
        description=(payload.description or inp.target_goal).strip()[:4000],
        total_nodes=len(nodes),
        curriculum_sources_registry=registry,
        nodes=nodes,
    )
    return sync_route_sources_from_registry(graph)


def generate_curriculum_search_first(
    inp: CurriculumGenerateInput,
    hits: list[CurriculumSearchHit],
    parsed_sources_json: str,
    anchor: str,
) -> CurriculumGraph:
    system = _METHODIST_SYSTEM.format(
        target_goal=inp.target_goal.strip(),
        parsed_sources_with_extracts_json=parsed_sources_json,
    )
    user = _build_user_payload(inp)

    payload = run_gemini_structured_with_chain(
        GEMINI_FLASH_MODEL,
        system,
        user,
        anchor,
        _FlashCurriculumPayload,
        "curriculum_generator / search_first",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=gemini_reasoner_model_chain(GEMINI_FLASH_MODEL),
    )
    graph = _graph_from_flash(inp, payload, hits)
    errors = validate_curriculum_dag_full(graph)
    if errors:
        hint = (
            "Граф отклонён валидатором. Исправь только prerequisites/слои/DAG:\n"
            + "\n".join(f"- {e}" for e in errors)
            + f"\n{CURRICULUM_DAG_REPAIR_PRESERVE_ANCHOR_TOPICS}"
        )
        trace(f"CURRICULUM search_first ▶ repair | errors={len(errors)}")
        payload = run_gemini_structured_with_chain(
            GEMINI_FLASH_MODEL,
            system,
            _build_user_payload(inp, repair_hint=hint),
            anchor,
            _FlashCurriculumPayload,
            "curriculum_generator / search_first_repair",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
            models=gemini_reasoner_model_chain(GEMINI_FLASH_MODEL),
        )
        graph = _graph_from_flash(inp, payload, hits)
        errors = validate_curriculum_dag_full(graph)

    if errors:
        raise ValueError("Search-First: невалидный DAG: " + "; ".join(errors[:5]))

    link_errors = validate_curriculum_source_links(graph)
    if link_errors:
        trace(f"CURRICULUM search_first warn | {link_errors[0]}")

    trace(
        f"CURRICULUM search_first ✓ | registry={len(graph.curriculum_sources_registry)} "
        f"nodes={graph.total_nodes}"
    )
    return graph
