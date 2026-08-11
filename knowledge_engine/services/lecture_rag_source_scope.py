"""Allowed LanceDB sources for lecture RAG (node primary + curriculum library)."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.services.skill_tree_store import get_curriculum_graph
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def normalize_lecture_source_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


@dataclass(frozen=True)
class LectureRagSourceScope:
    node_id: str
    curriculum_id: str
    primary_urls: tuple[str, ...]
    library_urls: tuple[str, ...]
    primary_doc_ids: tuple[str, ...]
    library_doc_ids: tuple[str, ...]

    @property
    def allowed_doc_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for did in (*self.primary_doc_ids, *self.library_doc_ids):
            if did and did not in seen:
                seen.add(did)
                out.append(did)
        return tuple(out)


def _doc_ids_for_urls(urls: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        u = normalize_lecture_source_url(raw)
        if not u.startswith("http"):
            continue
        did = VectorStore.doc_id_for_url(u)
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


def collect_curriculum_library_urls(curriculum_id: str) -> list[str]:
    cid = (curriculum_id or "").strip()
    if not cid:
        return []
    graph = get_curriculum_graph(cid) or {}
    seen: set[str] = set()
    urls: list[str] = []

    def add(raw: str) -> None:
        u = (raw or "").strip()
        if not u.startswith("http"):
            return
        key = normalize_lecture_source_url(u)
        if key in seen:
            return
        seen.add(key)
        urls.append(u)

    for rs in graph.get("route_sources") or []:
        if isinstance(rs, dict):
            add(str(rs.get("url") or ""))
    for entry in graph.get("curriculum_sources_registry") or []:
        if isinstance(entry, dict):
            add(str(entry.get("url") or ""))
    return urls


def collect_mapped_source_urls(curriculum_id: str, node: NodeDataInput) -> list[str]:
    """URL из curriculum_sources_registry по mapped_source_ids ноды."""
    cid = (curriculum_id or "").strip()
    if not cid:
        return []
    from knowledge_engine.src.curriculum.source_registry import resolve_sources_for_node

    graph = get_curriculum_graph(cid) or {}
    if not graph:
        return []
    mapped = [str(x).strip() for x in (node.mapped_source_ids or []) if str(x).strip()]
    rows = resolve_sources_for_node(graph, (node.node_id or "").strip(), mapped)
    urls: list[str] = []
    seen: set[str] = set()
    for row in rows:
        u = (row.get("url") or "").strip()
        if not u.startswith("http"):
            continue
        key = normalize_lecture_source_url(u)
        if key in seen:
            continue
        seen.add(key)
        urls.append(u)
    return urls


def mapped_doc_ids_for_node(curriculum_id: str, node: NodeDataInput) -> list[str]:
    return _doc_ids_for_urls(collect_mapped_source_urls(curriculum_id, node))


def build_lecture_rag_source_scope(
    curriculum_id: str,
    node: NodeDataInput,
    primary_urls: list[str],
) -> LectureRagSourceScope:
    """Primary = mapped node articles + route URLs; library = course registry minus primary."""
    mapped_urls = collect_mapped_source_urls(curriculum_id, node)
    primary_ordered: list[str] = []
    prim_norm: set[str] = set()
    for raw in (*mapped_urls, *primary_urls):
        u = (raw or "").strip()
        if not u.startswith("http"):
            continue
        key = normalize_lecture_source_url(u)
        if key in prim_norm:
            continue
        prim_norm.add(key)
        primary_ordered.append(u)
    library: list[str] = []
    for u in collect_curriculum_library_urls(curriculum_id):
        if normalize_lecture_source_url(u) not in prim_norm:
            library.append(u)

    primary_doc_ids = _doc_ids_for_urls(primary_ordered)
    library_doc_ids = _doc_ids_for_urls(library)

    return LectureRagSourceScope(
        node_id=(node.node_id or "").strip(),
        curriculum_id=(curriculum_id or "").strip(),
        primary_urls=tuple(primary_ordered),
        library_urls=tuple(library),
        primary_doc_ids=tuple(primary_doc_ids),
        library_doc_ids=tuple(library_doc_ids),
    )
