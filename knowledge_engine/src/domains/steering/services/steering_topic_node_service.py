"""Standalone Topic Digest — Режим 2 Штурвала, финал (см.

docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Разделение на Mode 1/Mode 2").

Итерация 2 (по прямому запросу пользователя, prompt.log "КРИТИЧЕСКОЕ
ИСПРАВЛЕНИЕ"): убрано искусственное деление approved-статей на ≥3
ноды-"бакета" ради валидатора ``CurriculumGraph.nodes``. Теперь результат —
РОВНО ОДНА сфокусированная нода (``node_kind="steering_standalone"``),
заземлённая через реальный тяжёлый Map-Reduce ПО ВСЕМ Gate-2-approved
статьям сразу (без капа ``CURRICULUM_DEEP_NODE_MAX_HITS`` — см. условную
валидацию в ``schemas.py``: ``node_kind``/``graph_kind`` снимают капы
``mapped_source_ids``/``nodes`` ИМЕННО для Штурвала, Autopilot продолжает
валидироваться как раньше без исключений). Нода получает явную структуру
глубокого разбора (4-12 подтем, ``NodeCurriculumBreakdown.subtopics``) —
дальше материал открывается ровно как любая другая grounded-нода: обычный
``node_deep_dive`` (Лекция/Topic Q&A) строит контент по существующему RAG
поверх заземления, никакой отдельной логики генерации лекции здесь нет.

Переиспользуется как есть (ничего не дублируется):
- ``source_material_pipeline.summarize_whitelist_blog_hits_async``/
  ``enrich_search_hits_with_extracts_async`` — тот же ingest, что
  ``node_grounding_finalize_service.py``/``targeted_node_grounding.py``.
- ``targeted_node_search.hit_to_registry_entry``,
  ``source_registry.cap_curriculum_sources_registry``/
  ``sync_route_sources_from_registry``,
  ``curriculum_lancedb_persist.persist_approved_curriculum_hits_to_lancedb_async``
  — тот же "хвост", что ``node_grounding_finalize_service.finalize_node_
  grounding``.
- ``node_grounding_finalize_service._infer_source_tier`` — импортирован как
  есть (тот же паттерн, что ``steering_lazy_ingest_service.py``).
"""

from __future__ import annotations

import json
import re
from itertools import count

from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.curriculum_lancedb_persist import (
    persist_approved_curriculum_hits_to_lancedb_async,
)
from knowledge_engine.src.domains.curriculum.lite_search_pipeline import (
    _lite_structured,
)
from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
    NodeCurriculumBreakdown,
    NodeSourceRef,
    NodeSubtopic,
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
from knowledge_engine.src.domains.curriculum.targeted_node_search import (
    hit_to_registry_entry,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    SurfaceDigestItem,
    SurfaceDigestResponse,
)
from knowledge_engine.src.shared.node_grounding.node_grounding_finalize_service import (
    _infer_source_tier,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_EXTRACTS_IN_SOURCE_REF = 12


def _slugify(text: str, *, fallback: str = "topic") -> str:
    s = _SLUG_RE.sub("_", (text or "").strip().lower()).strip("_")
    if not s or not s[0].isalpha():
        s = f"{fallback}_{s}" if s else fallback
    return s[:60]


def _unique_id(base: str, taken: set[str]) -> str:
    candidate = base
    for i in count(2):
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        candidate = f"{base}_{i}"[:80]


class _StandaloneNodeContent(BaseModel):
    title: str = Field(..., min_length=2, max_length=300)
    category: str = Field(..., min_length=2, max_length=200)
    brief_summary: str = Field(..., min_length=10, max_length=1200)
    core_concepts: list[str] = Field(..., min_length=1, max_length=8)
    subtopics: list[NodeSubtopic] = Field(
        ...,
        min_length=1,
        max_length=12,
        description=(
            "4-12 в норме (см. промпт) — min_length=1, а не 4, чтобы не "
            "заставлять LLM выдумывать подтемы, когда approved-статей "
            "заведомо мало (fallback-путь строит по одной подтеме на "
            "статью, тем же контрактом)."
        ),
    )


_CONTENT_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are structuring ONE deep-dive curriculum node from several "
    "already-approved technical articles on one topic — this is NOT a "
    "multi-node course, it is a single focused node covering the topic "
    "holistically. Base everything ONLY on the given articles, do not "
    "invent facts.\n\n"
    "Return JSON matching the required schema:\n"
    "- title: short node title for the whole topic (Russian).\n"
    "- category: short topical category label (Russian).\n"
    "- brief_summary: 2-3 dry technical sentences framing the whole node "
    "(Russian).\n"
    "- core_concepts: 1-8 short key-concept phrases spanning the material "
    "(Russian).\n"
    "- subtopics: 4 to 12 subtopics that together organize ALL the given "
    "articles' substance into a deep breakdown — each with a short title "
    "and a 2-3 sentence dry technical summary (Russian). Group related "
    "articles into the same subtopic where they overlap; split a single "
    "rich article into multiple subtopics if it covers distinct aspects. "
    "Do not just produce one subtopic per article mechanically — organize "
    "by SUBSTANCE."
)


def _fallback_content(
    target_goal: str, articles: list[SurfaceDigestItem]
) -> _StandaloneNodeContent:
    goal = (target_goal or "тема").strip()[:200]
    tech: list[str] = []
    for a in articles:
        for t in a.main_tech_stack or []:
            if t not in tech:
                tech.append(t)
    subtopics = [
        NodeSubtopic(
            title=(a.title or goal)[:200],
            summary=(a.two_sentence_summary or a.problem_solved or goal)[:800],
        )
        for a in articles[:12]
    ] or [
        NodeSubtopic(
            title=goal[:200],
            summary=f"Общий контекст и вводные понятия по теме «{goal}».",
        )
    ]
    return _StandaloneNodeContent(
        title=goal[:300] or "Обзор темы",
        category=goal[:200] or "Общее",
        brief_summary=(
            f"Материалы по теме «{goal}», утверждённые пользователем в Штурвале."
        )[:1200],
        core_concepts=(tech[:8] or [goal[:80]] or ["тема"]),
        subtopics=subtopics,
    )


async def _generate_content(
    target_goal: str,
    articles: list[SurfaceDigestItem],
    *,
    anchor: str,
) -> _StandaloneNodeContent:
    payload_articles = [
        {
            "title": a.title,
            "company_or_author": a.company_or_author,
            "problem_solved": a.problem_solved,
            "tech_stack": a.main_tech_stack,
            "summary": a.two_sentence_summary,
        }
        for a in articles
    ]
    try:
        out = await _lite_structured(
            _CONTENT_SYSTEM,
            json.dumps(
                {"target_goal": target_goal, "articles": payload_articles},
                ensure_ascii=False,
            ),
            f"{anchor}:content",
            _StandaloneNodeContent,
            "curriculum/steering_topic_node_content",
        )
    except Exception as exc:
        trace(f"STEERING topic_node content fallback | skip LLM | {exc}")
        return _fallback_content(target_goal, articles)
    if not isinstance(out, _StandaloneNodeContent):
        return _fallback_content(target_goal, articles)
    return out


async def generate_standalone_topic_graph(
    target_goal: str,
    digests: SurfaceDigestResponse,
    final_approved_urls: list[str],
    *,
    anchor: str | None = None,
    curriculum_id_hint: str = "",
) -> CurriculumGraph:
    """Gate 2 ``final_approved_urls`` → реальный Map-Reduce ПО ВСЕМ статьям →

    валидный ``CurriculumGraph`` из РОВНО ОДНОЙ ноды
    (``node_kind=graph_kind="steering_standalone"`` — без Autopilot-капов,
    см. schemas.py). Fail-open только на уровне content-generation LLM-вызова
    (``_fallback_content``) — сам ingest не fail-open: пусто на входе/выходе
    Map-Reduce — ошибка входных данных, поднимается наружу."""
    by_url = {(d.url or "").strip(): d for d in digests.digests}
    ordered_urls = [
        u.strip() for u in final_approved_urls if (u or "").strip() and u in by_url
    ]
    if not ordered_urls:
        raise ValueError(
            "Steering standalone_digest: ни для одного final_approved_urls "
            "нет готового SurfaceDigestItem — нечего инжестить"
        )

    stub_hits = [
        CurriculumSearchHit(
            url=u,
            title=by_url[u].title,
            snippet=(by_url[u].two_sentence_summary or "")[:1200],
            source_tier=_infer_source_tier(u),
        )
        for u in ordered_urls
    ]
    anchor_ = anchor or f"steering_topic_node:{target_goal.strip()[:200]}"
    trace(f"STEERING topic_node ingest ▶ | approved={len(ordered_urls)}")
    hits = await summarize_whitelist_blog_hits_async(stub_hits, target_goal)
    hits = await enrich_search_hits_with_extracts_async(hits, target_goal)
    if not hits:
        raise ValueError(
            "Steering standalone_digest: Map-Reduce не дал ни одного хита "
            "по approved URL — нечего оформлять в ноду"
        )
    trace(f"STEERING topic_node ingest ✓ | hits={len(hits)}")

    already_persisted_tiers = _BLOG_SOURCE_TIERS | _ACADEMIC_SOURCE_TIERS
    unpersisted_hits = [
        h for h in hits if (h.source_tier or "").strip() not in already_persisted_tiers
    ]
    if unpersisted_hits:
        await persist_approved_curriculum_hits_to_lancedb_async(
            unpersisted_hits, label=f"steering_topic_node:{anchor_}"
        )

    registry_hits: list[CurriculumSearchHit] = []
    registry = []
    sids: list[str] = []
    for i, raw_hit in enumerate(hits, start=1):
        sid = f"src_{i}"
        hit, entry = hit_to_registry_entry(raw_hit, sid)
        registry.append(entry)
        registry_hits.append(hit)
        sids.append(sid)

    articles = [by_url[u] for u in ordered_urls if u in by_url]
    content = await _generate_content(target_goal, articles, anchor=anchor_)

    extracts: list[str] = []
    for hit in registry_hits:
        for e in hit.key_extracts or []:
            e = e.strip()
            if e and e not in extracts:
                extracts.append(e)
        if len(extracts) >= _MAX_EXTRACTS_IN_SOURCE_REF:
            break
    if not extracts:
        extracts = [h.snippet[:800] for h in registry_hits if h.snippet][:8]
    primary = registry_hits[0]
    source_ref = NodeSourceRef(
        source_id=sids[0],
        url=primary.url[:2000],
        relevant_extracts=extracts[:_MAX_EXTRACTS_IN_SOURCE_REF],
    )

    node_id = _unique_id(_slugify(content.title), set())
    node = CurriculumNode(
        node_id=node_id,
        title=content.title,
        layer="foundation",
        category=content.category,
        brief_summary=content.brief_summary,
        core_concepts=content.core_concepts,
        resource_urls=[h.url[:2000] for h in registry_hits],
        primary_source_id=sids[0],
        mapped_source_ids=list(sids),
        source_ref=source_ref,
        node_curriculum_breakdown=NodeCurriculumBreakdown(
            key_concepts=content.core_concepts,
            architectural_focus=content.brief_summary,
            subtopics=content.subtopics,
        ),
        node_risk_kind="DEEP",
        grounding_status="grounded",
        node_kind="steering_standalone",
    )

    curriculum_id = _unique_id(
        _slugify(curriculum_id_hint or target_goal, fallback="steering_topic"),
        set(),
    )
    graph = CurriculumGraph(
        curriculum_id=curriculum_id,
        title=content.title,
        description=(
            f"Обзор темы «{target_goal.strip()}» по материалам, утверждённым "
            "в Штурвале (Standalone Topic Digest)."
        )[:4000],
        total_nodes=1,
        curriculum_sources_registry=registry,
        nodes=[node],
        graph_kind="steering_standalone",
    )
    registry_capped = cap_curriculum_sources_registry(registry, graph=graph)
    graph = graph.model_copy(update={"curriculum_sources_registry": registry_capped})
    graph = sync_route_sources_from_registry(graph)
    trace(
        f"STEERING topic_node ✓ | curriculum={graph.curriculum_id} "
        f"sources={len(registry_capped)} subtopics={len(content.subtopics)}"
    )
    return graph


__all__ = ["generate_standalone_topic_graph"]
