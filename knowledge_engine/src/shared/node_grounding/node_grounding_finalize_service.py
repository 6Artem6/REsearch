"""NodeGroundingFinalizeService — Node Grounding Gate, Этап 3: финальный

инжест по Gate-2-одобренным URL. Add-only, параллельно остальным сервисам
Node Grounding Gate — см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md.

ВАЖНО: здесь НЕТ нового Map-Reduce — пользователь прав, тяжёлый
многопроходный Gemma-инжест УЖЕ реализован и используется существующим
per-node lazy-grounding путём. Единственное, что нужно было НОВОГО —
это откуда берутся ``hits`` перед этим инжестом: вместо собственного
``search_sources_for_deep_node_async`` (с его SOTA-override, из-за
которого и был нужен весь этот гейт) — Gate 2 ``approved_urls``.

Переиспользуется как есть (ничего не дублируется):
- ``source_material_pipeline.summarize_whitelist_blog_hits_async`` — ТОТ ЖЕ
  унифицированный ingest, что использует ``search_sources_for_deep_node_async``
  для любого хита: blog-тиры идут пуловым MAP+REDUCE
  (``_ingest_blog_hits_batch_async``), academic-тиры — через
  ``_ingest_academic_hit_async``. Принимает "голые" ``CurriculumSearchHit``
  (только url/title/source_tier) — ей всё равно, откуда взялся список,
  найден ли он поиском или утверждён на Gate 2.
- ``source_material_pipeline.enrich_search_hits_with_extracts_async`` — тот
  же дешёвый backfill-проход, что в ``lazy_ground_deep_node_on_demand``.
- ``curriculum_lancedb_persist.persist_approved_curriculum_hits_to_lancedb_async``,
  ``targeted_node_search.hit_to_registry_entry``,
  ``targeted_node_grounding._attach_hits_to_node``,
  ``source_registry.cap_curriculum_sources_registry``/``sync_route_sources_from_registry``,
  ``diagram_session.refresh_node_session_diagrams_from_articles`` — БУКВАЛЬНО
  тот же "хвост" (после получения ``hits``), что и в
  ``targeted_node_grounding.lazy_ground_deep_node_on_demand`` — не
  переписан, а импортирован как есть. По аналогии с тем, как
  ``steering_generator_bridge.py`` строит ``CurriculumSearchHit`` прямо из
  готовых дайджестов Штурвала, минуя Discovery Phase, но переиспользуя
  дальнейший ``generator.py``-путь без изменений.
- ``targeted_node_search.py``/``engine.py`` НЕ импортируются целиком и не
  модифицируются — только их уже публичные/переиспользуемые куски (как
  договорено при проектировании гейта).

``save_curriculum_record`` НЕ вызывается здесь — как и
``lazy_ground_deep_node_on_demand``, эта функция только возвращает
обновлённые ``(graph, node)``; персист — ответственность вызывающего
эндпоинта (``api/routes/node_skill.py``), тот же паттерн, что
``_apply_lazy_grounding_for_init`` в ``engine.py``.
"""

from __future__ import annotations

from typing import Callable
from urllib.parse import urlparse

from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.curriculum_lancedb_persist import (
    persist_approved_curriculum_hits_to_lancedb_async,
)
from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
)
from knowledge_engine.src.domains.curriculum.source_material_pipeline import (
    _ACADEMIC_SOURCE_TIERS,
    _BLOG_SOURCE_TIERS,
    enrich_search_hits_with_extracts_async,
    summarize_whitelist_blog_hits_async,
)
from knowledge_engine.src.domains.curriculum.source_registry import (
    cap_curriculum_sources_registry,
    sync_route_sources_from_registry,
)
from knowledge_engine.src.domains.curriculum.targeted_node_grounding import (
    _attach_hits_to_node,
)
from knowledge_engine.src.domains.curriculum.targeted_node_search import (
    hit_to_registry_entry,
)
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NODE_GATE1_MAX_APPROVED,
)

# RU: у Gate 1/Gate 2 свой человекочитаемый source_tier ("habr") для
# паспортов — на Этапе 3 нужен КАНОНИЧЕСКИЙ тир (см. source_material_pipeline
# ._BLOG_SOURCE_TIERS/_ACADEMIC_SOURCE_TIERS), иначе summarize_whitelist_blog_hits_async
# молча пропустит хит без ingest (не найдёт тир ни в одном из двух множеств).
_TIER_DOMAIN_MAP: tuple[tuple[str, str], ...] = (
    ("arxiv.org", "arxiv"),
    ("semanticscholar.org", "semantic_scholar"),
    ("consensus.app", "consensus"),
    ("habr.com", "whitelist_blog"),
)


def _infer_source_tier(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    for domain, tier in _TIER_DOMAIN_MAP:
        if domain in host:
            return tier
    return "exa"


async def finalize_node_grounding(
    graph: CurriculumGraph,
    node: CurriculumNode,
    approved_urls: list[str],
    *,
    target_goal: str = "",
    on_progress: Callable[[str], None] | None = None,
) -> tuple[CurriculumGraph, CurriculumNode]:
    """Gate 2 финальное согласие → полный ingest → ``(graph, updated_node)``.

    Fail-open: если инжест не вернул ни одного хита (все URL сломаны/
    Gemma не смогла обработать) — узел возвращается БЕЗ изменений
    (остаётся в прежнем ``grounding_status``, не переходит в "grounded"
    без реального источника)."""
    urls = [u.strip() for u in approved_urls if (u or "").strip()][
        :NODE_GATE1_MAX_APPROVED
    ]
    if not urls:
        trace(f"NODE_GATE finalize ⊘ | node={node.node_id} | no approved urls")
        return graph, node

    stub_hits = [
        CurriculumSearchHit(url=u, title=u, source_tier=_infer_source_tier(u))
        for u in urls
    ]
    if on_progress:
        on_progress("Загружаем и обрабатываем материалы (Map-Reduce)…")
    trace(f"NODE_GATE finalize ▶ | node={node.node_id} urls={len(urls)}")
    hits = await summarize_whitelist_blog_hits_async(stub_hits, target_goal)
    hits = await enrich_search_hits_with_extracts_async(hits, target_goal)

    if not hits:
        trace(f"NODE_GATE finalize ⊘ | node={node.node_id} | ingest produced no hits")
        return graph, node

    already_persisted_tiers = _BLOG_SOURCE_TIERS | _ACADEMIC_SOURCE_TIERS
    unpersisted_hits = [
        h for h in hits if (h.source_tier or "").strip() not in already_persisted_tiers
    ]
    if unpersisted_hits:
        if on_progress:
            on_progress("Сохраняем материалы в базу знаний…")
        await persist_approved_curriculum_hits_to_lancedb_async(
            unpersisted_hits, label=f"node_gate_finalize:{node.node_id}"
        )

    registry = list(graph.curriculum_sources_registry)
    next_src_idx = len(registry) + 1
    sids: list[str] = []
    registry_hits: list[CurriculumSearchHit] = []
    for i, raw_hit in enumerate(hits):
        sid = f"src_{next_src_idx + i}"
        hit, entry = hit_to_registry_entry(raw_hit, sid)
        registry.append(entry)
        sids.append(sid)
        registry_hits.append(hit)

    updated_node = _attach_hits_to_node(node, registry_hits, sids)
    new_nodes = [updated_node if n.node_id == node.node_id else n for n in graph.nodes]
    graph = graph.model_copy(update={"nodes": new_nodes})
    registry = cap_curriculum_sources_registry(registry, graph=graph)
    graph = graph.model_copy(update={"curriculum_sources_registry": registry})
    graph = sync_route_sources_from_registry(graph)

    ingest_urls = [h.url for h in registry_hits if h.url.startswith("http")]
    from knowledge_engine.src.domains.grounding.diagram_session import (
        refresh_node_session_diagrams_from_articles,
    )

    if on_progress:
        on_progress("Готовим диаграммы по материалам…")
    n = refresh_node_session_diagrams_from_articles(
        graph.curriculum_id, updated_node, extra_urls=ingest_urls, rebuild=True
    )
    tiers = ", ".join(sorted({(h.source_tier or "?")[:12] for h in registry_hits}))
    trace(
        f"NODE_GATE finalize ✓ | node={node.node_id} status=grounded "
        f"registry+={len(registry_hits)} tiers={tiers} content_diagrams={n}"
    )
    return graph, updated_node


__all__ = ["finalize_node_grounding"]
