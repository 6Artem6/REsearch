"""SteeringLazyIngestService — при первом открытии Штурвал-сгенерированной

ноды реально доингестить примапленные источники (тот же тяжёлый ingest,
что ``node_grounding_finalize_service.py`` уже использует для Node
Grounding Gate), а не оставлять узел на дешёвом Gate-2 дайджесте
(2-3 предложения Flash Lite). См. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md —
там же разбор находки: ``WorkJobKind.STEERING_MAP_REDUCE``/докстринги
называют финальный шаг Штурвала "тяжёлым Map-Reduce", но
``generate_curriculum_from_steering_digests`` фактически строит курс ОДНИМ
Flash-вызовом прямо из дайджестов — реального Map-Reduce/ingest там нет и
не было. Проверено ``scripts/log_profiler.py`` на реальном прогоне: за
весь ``steering_map_reduce`` job — ровно один ``GEMINI HTTP`` вызов
(``curriculum_generator / search_first``), ни одного MAP/REDUCE события.

По прямому указанию пользователя: не эйджерно на генерации курса (замедлило
бы Gate 2 → курс с ~6s до многих секунд на статью), а ЛЕНИВО, при открытии
конкретной ноды — с дедупом между нодами: одна и та же approved-статья
нередко примаплена на несколько нод (см. живую отладку — вырожденный
случай, когда ``primary_source_id``/``mapped_source_ids`` у всех нод курса
совпадали), поэтому статья доингещивается РОВНО ОДИН РАЗ: как только её
``source_tier`` в ``curriculum_sources_registry`` перестаёт быть
``steering_approved``, следующие открытия других нод с тем же source_id
это видят и ничего не переделывают.

Переиспользуется как есть (ничего не дублируется):
- ``source_material_pipeline.summarize_whitelist_blog_hits_async`` — тот
  же унифицированный ingest, что и ``node_grounding_finalize_service.py``/
  штатный per-node lazy grounding.
- ``node_grounding_finalize_service._infer_source_tier`` — та же эвристика
  URL → канонический тир (``whitelist_blog``/``arxiv``/... ), нужна по той
  же причине: Gate 1/2 хранят человекочитаемый тир, ingest-пайплайну нужен
  канонический.

``targeted_node_search.py``/``engine.py`` не импортируются и не меняются —
эндпоинт (``api/routes/node_skill.py``) вызывается ФРОНТЕНДОМ ДО
``/node/init``, тот же паттерн, что уже используется для Node Grounding
Gate."""

from __future__ import annotations

from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
    NodeSourceRef,
)
from knowledge_engine.src.domains.curriculum.source_material_pipeline import (
    summarize_whitelist_blog_hits_async,
)
from knowledge_engine.src.shared.node_grounding.node_grounding_finalize_service import (
    _infer_source_tier,
)

_PENDING_TIER = "steering_approved"


def _extracts_from_hit(hit: CurriculumSearchHit) -> list[str]:
    extracts = [e.strip() for e in (hit.key_extracts or []) if e and str(e).strip()][
        :12
    ]
    if not extracts and hit.snippet:
        extracts = [hit.snippet[:800]]
    return extracts


def _url_key(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


async def ensure_node_steering_sources_ingested(
    graph: CurriculumGraph,
    node: CurriculumNode,
    *,
    target_goal: str = "",
) -> tuple[CurriculumGraph, CurriculumNode]:
    """Если у ноды есть ``mapped_source_ids`` с ``source_tier ==

    "steering_approved"`` (Gate-2-дайджест, ещё не прошёл реальный ingest)
    — доингестить их сейчас, обновить и реестр курса, и саму ноду
    (``source_ref``/``grounding_status``). Instant no-op (возвращает
    ``graph``/``node`` без изменений — ``updated_node is node``), если
    таких источников нет — то есть для подавляющего большинства нод
    (Autopilot, уже доингещенные Штурвал-ноды). Fail-open: сбой ingest —
    нода и реестр без изменений, следующее открытие попробует снова."""
    registry = list(graph.curriculum_sources_registry)
    by_id = {e.source_id: e for e in registry}
    pending_ids = [
        sid
        for sid in (node.mapped_source_ids or [])
        if sid in by_id and (by_id[sid].source_tier or "").strip() == _PENDING_TIER
    ]
    if not pending_ids:
        return graph, node

    stub_hits = [
        CurriculumSearchHit(
            url=by_id[sid].url,
            title=by_id[sid].title,
            source_tier=_infer_source_tier(by_id[sid].url),
        )
        for sid in pending_ids
    ]
    trace(
        f"STEERING lazy_ingest ▶ | node={node.node_id} pending_sources={len(pending_ids)}"
    )
    try:
        hits = await summarize_whitelist_blog_hits_async(stub_hits, target_goal)
    except Exception as exc:
        trace(f"STEERING lazy_ingest ✗ | node={node.node_id} | {exc}")
        return graph, node
    if not hits:
        trace(f"STEERING lazy_ingest ⊘ | node={node.node_id} | ingest returned nothing")
        return graph, node

    # Сопоставляем результат с исходными записями реестра по URL —
    # summarize_whitelist_blog_hits_async не обязан сохранять порядок, а
    # упавшие фетчи просто выпадают (fail-open внутри самой функции).
    hits_by_url = {_url_key(h.url): h for h in hits}

    new_registry: list = []
    updated_sids: set[str] = set()
    for e in registry:
        if e.source_id not in pending_ids:
            new_registry.append(e)
            continue
        h = hits_by_url.get(_url_key(e.url))
        if h is None:
            new_registry.append(e)
            continue
        extracts = _extracts_from_hit(h)
        new_registry.append(
            e.model_copy(
                update={
                    "key_extracts": extracts or e.key_extracts,
                    "snippet": (h.snippet or e.snippet)[:1200],
                    "source_tier": (h.source_tier or e.source_tier)[:24],
                }
            )
        )
        updated_sids.add(e.source_id)

    if not updated_sids:
        trace(f"STEERING lazy_ingest ⊘ | node={node.node_id} | no URL matched registry")
        return graph, node

    graph = graph.model_copy(update={"curriculum_sources_registry": new_registry})
    updated_by_id = {e.source_id: e for e in new_registry}

    primary_sid = (
        node.primary_source_id
        if node.primary_source_id in updated_sids
        else next(iter(updated_sids))
    )
    primary_entry = updated_by_id.get(primary_sid)
    updated_node = node
    if primary_entry is not None:
        updated_node = node.model_copy(
            update={
                "source_ref": NodeSourceRef(
                    source_id=primary_sid[:16],
                    url=primary_entry.url[:2000],
                    relevant_extracts=list(primary_entry.key_extracts or [])[:12],
                ),
                "grounding_status": "grounded",
            }
        )
        new_nodes = [
            updated_node if n.node_id == node.node_id else n for n in graph.nodes
        ]
        graph = graph.model_copy(update={"nodes": new_nodes})

    trace(
        f"STEERING lazy_ingest ✓ | node={node.node_id} updated_sources={len(updated_sids)}"
    )
    return graph, updated_node


__all__ = ["ensure_node_steering_sources_ingested"]
