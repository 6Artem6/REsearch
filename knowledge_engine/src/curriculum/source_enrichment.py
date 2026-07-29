"""Второй этап: curriculum_sources_registry + mapped_source_ids (Lite)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from knowledge_engine.config import GEMINI_LITE_MODEL, GEMINI_RPM_PAUSE_SEC
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.curriculum_whitelist_prompt import curriculum_whitelist_prompt_block
from knowledge_engine.services.gemini_stateless import (
    gemini_lite_model_chain,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumNode,
    CurriculumResourceRef,
    CurriculumSourceRegistryEntry,
    LearningMaterials,
    PrimaryWhitelistSource,
)
from knowledge_engine.src.curriculum.source_registry import (
    sync_route_sources_from_registry,
    validate_curriculum_source_links,
)
from knowledge_engine.ui.run_log import trace

_SOURCES_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты составляешь библиотеку источников курса и точечную адресацию к нодам DAG.\n\n"
    "{whitelist_block}\n"
    "СТРУКТУРА ОТВЕТА (ОБЯЗАТЕЛЬНО ДВА БЛОКА):\n"
    "1) curriculum_sources_registry — 8–15 ресурсов из Whitelist по ВСЕЙ теме курса.\n"
    "   source_id: src_1, src_2, … (уникальные). title, whitelist_domain, source_type, url, why_read.\n"
    "2) nodes[] — для КАЖДОЙ ноды из nodes_outline:\n"
    "   - mapped_source_ids: 1–3 source_id ИЗ registry (не пустой)\n"
    "   - learning_goal: цель ноды в 1–2 предложения\n"
    "   - primary_whitelist_source: source_name, chapter_or_article, core_concepts\n"
    "   - learning_resources / resource_urls — URL только из registry\n"
    "ЗАПРЕЩЕНО: 3–4 общие ссылки на все ноды; каждая нода — свои 1–3 src_id.\n"
)


class _GeminiRegistryEntry(BaseModel):
    source_id: str = ""
    title: str = ""
    whitelist_domain: str = ""
    source_type: str = ""
    url: str = ""
    why_read: str = ""


class _GeminiPrimarySource(BaseModel):
    source_name: str = ""
    chapter_or_article: str = ""
    core_concepts: list[str] = Field(default_factory=list)


class _GeminiLearningResource(BaseModel):
    title: str = ""
    url: str = ""
    why_read: str = ""


class _GeminiNodePatch(BaseModel):
    node_id: str = ""
    mapped_source_ids: list[str] = Field(default_factory=list)
    learning_goal: str = ""
    primary_source_id: str = ""
    primary_whitelist_source: _GeminiPrimarySource = Field(
        default_factory=_GeminiPrimarySource
    )
    learning_resources: list[_GeminiLearningResource] = Field(default_factory=list)
    resource_urls: list[str] = Field(default_factory=list)


class _GeminiSourcesEnrichment(BaseModel):
    curriculum_sources_registry: list[_GeminiRegistryEntry] = Field(default_factory=list)
    nodes: list[_GeminiNodePatch] = Field(default_factory=list)


def _norm_src_id(raw: str, index: int) -> str:
    s = (raw or "").strip()
    if re.match(r"^src_\d+$", s, re.I):
        return s.lower().replace("SRC_", "src_")
    if re.match(r"^S\d+$", s, re.I):
        return f"src_{s[1:]}"
    return f"src_{index}"


def _to_registry_entry(raw: _GeminiRegistryEntry, index: int) -> CurriculumSourceRegistryEntry | None:
    title = (raw.title or "").strip()
    url = (raw.url or "").strip()
    domain = (raw.whitelist_domain or "").strip()
    if not title and not domain:
        return None
    sid = _norm_src_id(raw.source_id, index)
    if url and not url.startswith("http"):
        url = ""
    return CurriculumSourceRegistryEntry(
        source_id=sid[:16],
        title=title[:400] or domain[:400] or sid,
        whitelist_domain=domain[:200],
        source_type=(raw.source_type or "").strip()[:120],
        url=url[:2000],
        why_read=(raw.why_read or "").strip()[:800],
    )


def _to_primary(raw: _GeminiPrimarySource) -> PrimaryWhitelistSource | None:
    name = (raw.source_name or "").strip()
    chapter = (raw.chapter_or_article or "").strip()
    concepts = [c.strip() for c in raw.core_concepts if c and str(c).strip()][:12]
    if not name or not chapter or not concepts:
        return None
    return PrimaryWhitelistSource(
        source_name=name[:400],
        chapter_or_article=chapter[:800],
        core_concepts=concepts,
    )


def _nodes_need_sources(graph: CurriculumGraph) -> bool:
    if len(graph.curriculum_sources_registry) < 8:
        return True
    for n in graph.nodes:
        if not n.mapped_source_ids:
            return True
        p = n.learning_materials.primary_whitelist_source
        if p is None:
            return True
    return False


def _build_enrichment_payload(inp: CurriculumGenerateInput, graph: CurriculumGraph) -> str:
    lines = [
        f"### target_goal\n{inp.target_goal.strip()}",
        f"### curriculum_title\n{graph.title}",
        "### nodes_outline",
    ]
    for n in graph.nodes:
        concepts = ", ".join(n.core_concepts[:6])
        lines.append(
            f"- {n.node_id} | {n.layer} | {n.title} | {n.category} | concepts: {concepts}"
        )
    lines.append(
        "\nСначала curriculum_sources_registry (8–15), затем nodes[] с mapped_source_ids "
        "для ВСЕХ node_id (1–3 src на ноду)."
    )
    return "\n".join(lines)


def _apply_enrichment(
    graph: CurriculumGraph,
    enrich: _GeminiSourcesEnrichment,
) -> CurriculumGraph:
    registry: list[CurriculumSourceRegistryEntry] = []
    for i, rs in enumerate(enrich.curriculum_sources_registry or [], start=1):
        entry = _to_registry_entry(rs, i)
        if entry:
            registry.append(entry)

    reg_ids = {e.source_id for e in registry}
    patch_by_id: dict[str, _GeminiNodePatch] = {}
    for p in enrich.nodes or []:
        nid = (p.node_id or "").strip()
        if nid:
            patch_by_id[nid] = p

    new_nodes: list[CurriculumNode] = []
    for n in graph.nodes:
        patch = patch_by_id.get(n.node_id)
        if not patch:
            new_nodes.append(n)
            continue
        mapped = []
        for raw_id in patch.mapped_source_ids or []:
            sid = _norm_src_id(raw_id, len(mapped) + 1)
            if sid in reg_ids and sid not in mapped:
                mapped.append(sid)
        if not mapped and registry:
            mapped = [registry[0].source_id]

        primary = _to_primary(patch.primary_whitelist_source)
        lm = LearningMaterials(primary_whitelist_source=primary) if primary else n.learning_materials

        refs: list[CurriculumResourceRef] = []
        for lr in patch.learning_resources or []:
            u = (lr.url or "").strip()
            if len(u) < 8:
                continue
            refs.append(
                CurriculumResourceRef(
                    title=(lr.title or "").strip()[:400],
                    url=u[:2000],
                    why_read=(lr.why_read or "").strip()[:800],
                )
            )
        for sid in mapped:
            for e in registry:
                if e.source_id == sid and e.url.startswith("http"):
                    if e.url not in [r.url for r in refs]:
                        refs.append(
                            CurriculumResourceRef(
                                title=e.title[:400],
                                url=e.url[:2000],
                                why_read=(e.why_read or "")[:800],
                            )
                        )
                    break

        urls = [u.strip() for u in (patch.resource_urls or []) if u.strip().startswith("http")]
        for lr in refs:
            if lr.url not in urls:
                urls.append(lr.url)

        primary_sid = (patch.primary_source_id or "").strip()
        if not primary_sid and mapped:
            primary_sid = mapped[0]
        primary_sid = _norm_src_id(primary_sid, 1) if primary_sid else (mapped[0] if mapped else "")

        new_nodes.append(
            n.model_copy(
                update={
                    "mapped_source_ids": mapped[:3],
                    "primary_source_id": primary_sid[:16],
                    "learning_goal": (patch.learning_goal or n.learning_goal or "")[:600],
                    "learning_materials": lm,
                    "learning_resources": refs[:8],
                    "resource_urls": urls[:12],
                }
            )
        )

    out = graph.model_copy(
        update={
            "nodes": new_nodes,
            "curriculum_sources_registry": registry[:20],
        }
    )
    return sync_route_sources_from_registry(out)


def enrich_curriculum_whitelist_sources(
    inp: CurriculumGenerateInput,
    graph: CurriculumGraph,
    anchor: str,
) -> CurriculumGraph:
    if not _nodes_need_sources(graph):
        trace(
            f"CURRICULUM sources skip | registry={len(graph.curriculum_sources_registry)} "
            "(уже заполнены)"
        )
        return graph

    trace(f"CURRICULUM sources enrich ▶ Lite | nodes={graph.total_nodes}")
    system = _SOURCES_SYSTEM.format(whitelist_block=curriculum_whitelist_prompt_block())
    payload = _build_enrichment_payload(inp, graph)
    try:
        enrich = run_gemini_structured_with_chain(
            GEMINI_LITE_MODEL,
            system,
            payload,
            anchor,
            _GeminiSourcesEnrichment,
            "curriculum_generator / whitelist_sources",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
            models=gemini_lite_model_chain(),
        )
    except Exception as exc:
        trace(f"CURRICULUM sources enrich ✗ | {exc}")
        return graph

    out = _apply_enrichment(graph, enrich)
    link_errors = validate_curriculum_source_links(out)
    if link_errors:
        trace(f"CURRICULUM sources warn | {link_errors[0]}")
    filled = sum(1 for n in out.nodes if n.mapped_source_ids)
    trace(
        f"CURRICULUM sources enrich ✓ | registry={len(out.curriculum_sources_registry)} "
        f"nodes_mapped={filled}/{len(out.nodes)}"
    )
    return out
