"""Локальный LanceDB контекст для генерации лекций (Node Deep-Dive)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import LECTURE_RAG_TOP_K, LIGHT_RAG_MIN_COSINE_SIM
from knowledge_engine.db.source_links import get_source_link_archive
from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
    format_primary_whitelist_foundation,
    primary_whitelist_from_graph_node,
)
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.skill_tree_store import get_curriculum_graph, get_curriculum_meta
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.locks import run_under_uma_lock
from knowledge_engine.src.memory.light_rag import LightRAG
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.session_store import get_session
from knowledge_engine.ui.run_log import trace

LECTURE_RAG_FALLBACK = (
    "Локальный конспект не найден. Сгенерируй лекцию на основе фундаментальных "
    "Best Practices архитектуры."
)


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _build_search_query(
    node: NodeDataInput,
    user_query: str,
    curriculum_goal: str = "",
) -> str:
    concepts = ", ".join(node.core_concepts[:8])
    parts = [
        curriculum_goal,
        node.title,
        node.brief_summary,
        node.category,
        concepts,
        f"node_id:{node.node_id}",
        (user_query or "").strip(),
    ]
    return "\n".join(p for p in parts if (p or "").strip())[:4000]


def _format_document_summary(ds: DocumentSummary, index: int, tag: str = "Конспект") -> str:
    lines = [
        f"### {tag} {index}: {ds.title}",
        f"URL: {ds.url}",
    ]
    if ds.cs_concepts:
        lines.append("Концепты: " + ", ".join(ds.cs_concepts[:16]))
    if ds.key_takeaways:
        lines.append(
            "Выжимка:\n" + "\n".join(f"- {t}" for t in ds.key_takeaways[:14])
        )
    if ds.failure_modes:
        lines.append(
            "Failure modes:\n" + "\n".join(f"- {t}" for t in ds.failure_modes[:8])
        )
    if ds.diagram_descriptions:
        lines.append(
            "Схемы:\n" + "\n".join(f"- {t}" for t in ds.diagram_descriptions[:6])
        )
    return "\n".join(lines)


def _format_knowledge_node(index: int, content: str, source_url: str | None, level: str) -> str:
    head = f"### Knowledge graph {index} [{level}]"
    if source_url:
        head += f" ({source_url})"
    return f"{head}\n{content[:6000]}"


def _format_registry_stub(
    index: int,
    url: str,
    title: str,
    snippet: str,
) -> str:
    lines = [f"### Ссылка маршрута/registry {index}: {title or url}", f"URL: {url}"]
    if snippet:
        lines.append(f"Контекст: {snippet[:1200]}")
    return "\n".join(lines)


def _hybrid_document_summaries(query: str, limit: int) -> list[DocumentSummary]:
    store = VectorStore()
    try:
        return store.hybrid_search(query, limit=limit)
    except RuntimeError:
        return []
    except Exception as exc:
        trace(f"LECTURE_RAG hybrid_search skip | {exc}")
        return []


def _summaries_for_urls(urls: list[str], limit: int) -> list[DocumentSummary]:
    store = VectorStore()
    try:
        return store.fetch_summaries_by_urls(urls, limit=limit)
    except Exception as exc:
        trace(f"LECTURE_RAG fetch_by_url skip | {exc}")
        return []


def _hybrid_knowledge_nodes(query: str, limit: int) -> list[tuple[str, str | None, str]]:
    store = VectorStore()
    try:
        nodes = store.hybrid_search_nodes(query, limit=limit)
        return [(n.content, n.source_url, n.level) for n in nodes if (n.content or "").strip()]
    except Exception as exc:
        trace(f"LECTURE_RAG knowledge_nodes skip | {exc}")
        return []


def _collect_route_link_candidates(
    curriculum_id: str,
    node: NodeDataInput,
) -> list[tuple[str, str, str]]:
    """URL + title + snippet из маршрута, сессии ноды и архива ссылок."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(url: str, title: str = "", snippet: str = "") -> None:
        u = (url or "").strip()
        if not u.startswith("http"):
            return
        key = _normalize_url(u)
        if key in seen:
            return
        seen.add(key)
        out.append((u, (title or "").strip(), (snippet or "").strip()))

    cid = (curriculum_id or "").strip()
    nid = node.node_id

    graph = get_curriculum_graph(cid) if cid else None
    route_by_id: dict[str, dict] = {}
    if graph:
        for rs in graph.get("route_sources") or []:
            if isinstance(rs, dict) and rs.get("source_id"):
                route_by_id[str(rs["source_id"])] = rs
        for raw in graph.get("nodes") or []:
            if str(raw.get("node_id") or "") != nid:
                continue
            pid = str(raw.get("primary_source_id") or "").strip()
            if pid and pid in route_by_id:
                rs = route_by_id[pid]
                add(
                    str(rs.get("url") or ""),
                    str(rs.get("source_name") or ""),
                    str(rs.get("why_read") or ""),
                )
            for u in raw.get("resource_urls") or []:
                add(str(u))
            for lr in raw.get("learning_resources") or []:
                if isinstance(lr, dict):
                    add(
                        str(lr.get("url") or ""),
                        str(lr.get("title") or ""),
                        str(lr.get("why_read") or ""),
                    )

    if cid:
        session = get_session(cid, nid)
        for ref in session.content.references or []:
            add(
                ref.url,
                ref.title or ref.source_name,
                "\n".join(
                    x for x in [ref.why_read, ref.key_focus] if x
                ),
            )
        from knowledge_engine.src.node_deep_dive.session_store import (
            get_all_sessions_for_curriculum,
        )

        blob = get_all_sessions_for_curriculum(cid).get(nid) or {}
        for reg in blob.get("source_registry") or []:
            if isinstance(reg, dict):
                add(
                    str(reg.get("url") or ""),
                    str(reg.get("title") or ""),
                    str(reg.get("snippet") or ""),
                )

    meta = get_curriculum_meta(cid) if cid else None
    goal = (meta.get("target_goal") or "") if meta else ""
    if goal:
        try:
            archive = get_source_link_archive()
            for u in archive.get_reusable_urls(
                f"{goal} {node.title}",
                explored=set(),
                limit=6,
                min_trust=0.35,
            ):
                add(u, "archive", goal[:200])
        except Exception:
            pass

    return out[:16]


async def retrieve_lecture_rag_context(
    node: NodeDataInput,
    user_query: str,
    curriculum_id: str = "",
) -> str:
    """
    TOP-K из LanceDB + конспекты по URL маршрута (learning_resources / registry) +
    light_rag_facts + knowledge_nodes.
    """
    node = enrich_node_learning_materials_from_graph(node, curriculum_id)
    meta = get_curriculum_meta(curriculum_id.strip()) if curriculum_id else None
    curriculum_goal = (meta.get("target_goal") or "") if meta else ""
    query = _build_search_query(node, user_query, curriculum_goal)
    if not query:
        return LECTURE_RAG_FALLBACK

    top_k = max(2, min(LECTURE_RAG_TOP_K, 5))
    chunks: list[str] = []
    seen_urls: set[str] = set()
    url_docs: list[DocumentSummary] = []

    foundation = format_primary_whitelist_foundation(node, curriculum_id)
    if foundation.strip():
        chunks.append(foundation.strip())

    route_links = _collect_route_link_candidates(curriculum_id, node)
    route_urls = [u for u, _, _ in route_links]
    if route_urls:
        url_docs = await run_under_uma_lock(
            _summaries_for_urls, route_urls, min(len(route_urls), top_k + 2)
        )
        for i, ds in enumerate(url_docs, 1):
            seen_urls.add(_normalize_url(ds.url))
            chunks.append(_format_document_summary(ds, i, tag="Конспект по ссылке маршрута"))

    for i, (url, title, snippet) in enumerate(route_links, 1):
        if _normalize_url(url) in seen_urls:
            continue
        chunks.append(_format_registry_stub(i, url, title, snippet))

    docs = await run_under_uma_lock(_hybrid_document_summaries, query, top_k)
    doc_idx = 0
    for ds in docs:
        key = _normalize_url(ds.url)
        if key in seen_urls:
            continue
        seen_urls.add(key)
        doc_idx += 1
        chunks.append(_format_document_summary(ds, doc_idx, tag="Семантический конспект"))

    knodes = await run_under_uma_lock(_hybrid_knowledge_nodes, query, 2)
    for i, (content, source_url, level) in enumerate(knodes, 1):
        chunks.append(_format_knowledge_node(i, content, source_url, level))

    rag = LightRAG()
    hits = await rag.vector_search(
        query,
        top_k,
        kinds=frozenset({"profile", "fact"}),
        min_cosine=LIGHT_RAG_MIN_COSINE_SIM,
    )
    for sim, text, meta_h in hits:
        node_tag = (meta_h.get("node_id") or "").strip()
        category = (meta_h.get("category") or "").strip()
        header = f"[LightRAG score={sim:.2f}"
        if node_tag:
            header += f" node={node_tag}"
        if category:
            header += f" category={category}"
        header += "]"
        chunks.append(f"{header}\n{text[:6000]}")

    if not chunks:
        trace(f"LECTURE_RAG ⊘ empty | query_len={len(query)} route_urls={len(route_urls)}")
        return LECTURE_RAG_FALLBACK

    trace(
        f"LECTURE_RAG ✓ | route_urls={len(route_urls)} url_docs={len(url_docs) if route_urls else 0} "
        f"hybrid_docs={len(docs)} light_hits={len(hits)} chunks={len(chunks)}"
    )
    return "\n\n---\n\n".join(chunks)


def build_lecture_generation_payload(
    node: NodeDataInput,
    memory_rag_profile: str,
    user_query: str,
    rag_context: str,
    concepts_matrix: str,
    rolling_summary: str,
    curriculum_id: str = "",
) -> str:
    topic = (user_query or "").strip() or node.title
    body = (rag_context or "").strip() or LECTURE_RAG_FALLBACK
    concepts = "\n".join(f"- {c}" for c in node.core_concepts)
    foundation = format_primary_whitelist_foundation(node, curriculum_id)
    foundation_block = f"{foundation}\n\n" if foundation.strip() else ""
    goal_line = ""
    lg = (getattr(node, "learning_goal", None) or "").strip()
    if lg:
        goal_line = f"Цель ноды: {lg}\n\n"
    return (
        "Ты — IT-Тьютор.\n\n"
        f"Тема ноды: {node.title}\n\n"
        f"{goal_line}"
        f"{foundation_block}"
        "ЗАДАЧА: сгенерируй ПОДРОБНУЮ лекцию (от 400 слов), опираясь строго на "
        "привязанные источники из реестра курса (если указаны) и блок НАЧАЛО МАТЕРИАЛА.\n\n"
        "Входной материал из локальной базы знаний:\n"
        "=== НАЧАЛО МАТЕРИАЛА ===\n"
        f"{body}\n"
        "=== КОНЕЦ МАТЕРИАЛА ===\n\n"
        f"Тема/Запрос пользователя: {topic}\n\n"
        "ИНСТРУКЦИЯ ПО ГЕНЕРАЦИИ ЛЕКЦИИ:\n"
        "1. Напиши ПОДРОБНУЮ лекцию (от 400 слов), опираясь на ФУНДАМЕНТАЛЬНЫЙ ИСТОЧНИК "
        "(если указан) и блок НАЧАЛО МАТЕРИАЛА как главный авторитетный базис.\n"
        "2. Подробно разбери каждую ключевую концепцию из источника с архитектурными примерами.\n"
        "3. Разбей лекцию: Проблема → Причины → Архитектурное решение → Примеры/Код.\n"
        "4. Запрещено краткое резюме и «Материал перед вами». Полный текст — в lecture_body.\n\n"
        "### node_metadata (для панели UI, JSON-поля ответа)\n"
        f"title: {node.title}\n"
        f"layer: {node.layer}\n"
        f"category: {node.category}\n"
        f"brief_summary: {node.brief_summary}\n"
        f"core_concepts:\n{concepts}\n"
        f"directional_rag_profile:\n{memory_rag_profile}\n"
        f"concepts_matrix:\n{concepts_matrix}\n"
        f"rolling_summary:\n{rolling_summary or '(пусто)'}\n"
    )
