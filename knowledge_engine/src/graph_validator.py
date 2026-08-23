"""Knowledge-graph integrity: DAG cycles, orphan ids, overlay-layer refs.

Independent of the v0.7 LangGraph orchestrator in ``knowledge_engine.src.graph``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_VALID_OVERLAY_TYPES = frozenset({"ADVANCED_ASTERISK", "DEEP_ASTERISK"})
_VALID_OVERLAY_KINDS = frozenset(
    {"advanced_analysis", "deep_design", "deep_analysis"}
)
_SUBCONCEPT_REF_KEYS = (
    "pending_evaluation_concept_id",
    "asked_question_sub_concept_id",
    "next_question_concept_id",
    "last_tutor_sub_concept_id",
    "focus_sub_concept_id",
)


@dataclass(frozen=True)
class GraphIntegrityReport:
    """Structured result of ``validate_knowledge_graph_integrity``."""

    ok: bool
    errors: tuple[str, ...] = ()
    cycles: tuple[tuple[str, ...], ...] = ()
    orphan_node_ids: tuple[str, ...] = ()
    orphan_sub_concept_ids: tuple[str, ...] = ()
    overlay_errors: tuple[str, ...] = ()
    node_count: int = 0
    sub_concept_count: int = 0


def _as_mapping(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        data = dump()
        return dict(data) if isinstance(data, Mapping) else {}
    return {}


def _as_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _node_id_of(node: Any) -> str:
    data = _as_mapping(node)
    return str(data.get("node_id") or data.get("id") or "").strip()


def _sub_id_of(row: Any) -> str:
    data = _as_mapping(row)
    return str(data.get("id") or data.get("sub_concept_id") or "").strip()


def _extract_nodes(graph_data: Any) -> list[Any]:
    data = _as_mapping(graph_data)
    nodes = _as_list(data.get("nodes"))
    if nodes:
        return nodes
    inner = data.get("graph")
    if inner is not None:
        return _extract_nodes(inner)
    return []


def _collect_sub_concepts(graph_data: Any, nodes: list[Any]) -> list[dict[str, Any]]:
    data = _as_mapping(graph_data)
    rows: list[dict[str, Any]] = []
    for raw in _as_list(data.get("sub_concepts")):
        item = _as_mapping(raw)
        if _sub_id_of(item):
            rows.append(item)
    memory = data.get("memory")
    if memory is not None:
        mem = _as_mapping(memory)
        for raw in _as_list(mem.get("sub_concepts")):
            item = _as_mapping(raw)
            if _sub_id_of(item):
                rows.append(item)
    for node in nodes:
        nd = _as_mapping(node)
        for raw in _as_list(nd.get("sub_concepts")):
            item = _as_mapping(raw)
            if _sub_id_of(item):
                item.setdefault("node_id", _node_id_of(nd))
                rows.append(item)
    # Deduplicate by id, first wins.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in rows:
        sid = _sub_id_of(item)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(item)
    return out


def _collect_overlays(graph_data: Any, nodes: list[Any]) -> list[dict[str, Any]]:
    data = _as_mapping(graph_data)
    rows: list[dict[str, Any]] = []

    def _push(raw: Any) -> None:
        item = _as_mapping(raw)
        if not item:
            if isinstance(raw, str) and raw.strip():
                rows.append(
                    {"concept_id": raw.strip(), "overlay_type": "DEEP_ASTERISK"}
                )
            return
        cid = str(
            item.get("concept_id")
            or item.get("sub_concept_id")
            or item.get("id")
            or ""
        ).strip()
        if cid:
            rows.append(item if "concept_id" in item else {**item, "concept_id": cid})

    for raw in _as_list(data.get("overlays")):
        _push(raw)
    for raw in _as_list(data.get("deep_mastery_concepts")):
        _push(raw)
    memory = data.get("memory")
    if memory is not None:
        mem = _as_mapping(memory)
        for raw in _as_list(mem.get("deep_mastery_concepts")):
            _push(raw)
        for raw in _as_list(mem.get("overlays")):
            _push(raw)
    for node in nodes:
        nd = _as_mapping(node)
        for key in ("overlays", "overlay_refs", "deep_mastery_concepts"):
            for raw in _as_list(nd.get(key)):
                _push(raw)
        for sc in _as_list(nd.get("sub_concepts")):
            scd = _as_mapping(sc)
            kind = str(scd.get("overlay_kind") or scd.get("overlay_type") or "").strip()
            if kind and (
                bool(scd.get("is_extension"))
                or kind.upper() in _VALID_OVERLAY_TYPES
                or kind.lower() in _VALID_OVERLAY_KINDS
            ):
                cid = _sub_id_of(scd)
                if cid:
                    rows.append(
                        {
                            "concept_id": cid,
                            "overlay_kind": kind,
                            "overlay_type": scd.get("overlay_type") or "",
                            "is_extension": bool(scd.get("is_extension")),
                            "parent_id": str(scd.get("parent_id") or "").strip(),
                        }
                    )
    return rows


def _tarjan_cycles(adj: dict[str, list[str]]) -> list[tuple[str, ...]]:
    """Tarjan SCCs: components with a cycle (size > 1 or a self-loop)."""
    index_counter = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index_counter
        index[v] = index_counter
        lowlink[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, ()):
            if w not in adj:
                continue
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in adj:
        if v not in index:
            strongconnect(v)

    cycles: list[tuple[str, ...]] = []
    for scc in sccs:
        if len(scc) > 1:
            cycles.append(tuple(reversed(scc)))
            continue
        v = scc[0]
        if v in adj.get(v, ()):
            cycles.append((v,))
    return cycles


def _dangling_subconcept_refs(
    graph_data: Any,
    known: set[str],
) -> list[str]:
    data = _as_mapping(graph_data)
    dangling: list[str] = []
    seen: set[str] = set()

    def check(cid: str, *, ctx: str) -> None:
        sid = (cid or "").strip()
        if not sid or sid in known or sid in seen:
            return
        seen.add(sid)
        dangling.append(f"{ctx}:{sid}")

    for key in _SUBCONCEPT_REF_KEYS:
        check(str(data.get(key) or ""), ctx=key)
    memory = data.get("memory")
    if memory is not None:
        mem = _as_mapping(memory)
        for key in _SUBCONCEPT_REF_KEYS:
            check(str(mem.get(key) or ""), ctx=f"memory.{key}")
    return dangling


def validate_knowledge_graph_integrity(graph_data: Any) -> GraphIntegrityReport:
    """
    Validate a curriculum / session knowledge graph.

    Checks:
      1. Prerequisite DAG has no cycles (Tarjan SCCs).
      2. Orphan / broken ``node_id`` and ``sub_concept_id`` references.
      3. Overlay-layer awards and extension rows point at real sub-concepts
         with a valid overlay type/kind.
    """
    errors: list[str] = []
    overlay_errors: list[str] = []
    nodes = _extract_nodes(graph_data)
    by_id: dict[str, dict[str, Any]] = {}
    dupes: list[str] = []
    for node in nodes:
        nid = _node_id_of(node)
        if not nid:
            errors.append("node missing node_id")
            continue
        if nid in by_id:
            dupes.append(nid)
            continue
        by_id[nid] = _as_mapping(node)

    if dupes:
        errors.append("duplicate node_id: " + ", ".join(sorted(set(dupes))))

    adj: dict[str, list[str]] = {nid: [] for nid in by_id}
    orphan_nodes: list[str] = []
    seen_orphans: set[str] = set()
    for nid, nd in by_id.items():
        prereqs = [
            str(p).strip()
            for p in _as_list(nd.get("prerequisites"))
            if str(p).strip()
        ]
        for p in prereqs:
            if p == nid:
                errors.append(f"node '{nid}': self-reference in prerequisites")
            if p not in by_id:
                if p not in seen_orphans:
                    seen_orphans.add(p)
                    orphan_nodes.append(p)
                    errors.append(
                        f"node '{nid}': prerequisite '{p}' is not in the graph"
                    )
                continue
            adj[nid].append(p)

    cycles = tuple(_tarjan_cycles(adj))
    if cycles:
        for cyc in cycles:
            errors.append("cycle: " + " → ".join(cyc) + f" → {cyc[0]}")

    sub_concepts = _collect_sub_concepts(graph_data, nodes)
    known_subs = {_sub_id_of(sc) for sc in sub_concepts}
    orphan_subs: list[str] = []
    seen_sub_orphans: set[str] = set()

    def note_sub_orphan(sid: str, *, ctx: str) -> None:
        if not sid or sid in known_subs or sid in seen_sub_orphans:
            return
        seen_sub_orphans.add(sid)
        orphan_subs.append(sid)
        errors.append(f"broken sub_concept_id '{sid}' ({ctx})")

    for sc in sub_concepts:
        parent = str(sc.get("parent_id") or sc.get("core_id") or "").strip()
        if parent:
            note_sub_orphan(parent, ctx=f"parent of {_sub_id_of(sc)}")
        owner = str(sc.get("node_id") or "").strip()
        if owner and by_id and owner not in by_id:
            if owner not in seen_orphans:
                seen_orphans.add(owner)
                orphan_nodes.append(owner)
                errors.append(
                    f"sub_concept '{_sub_id_of(sc)}' references missing node '{owner}'"
                )

    for ref in _dangling_subconcept_refs(graph_data, known_subs):
        sid = ref.split(":", 1)[-1]
        note_sub_orphan(sid, ctx=ref.rsplit(":", 1)[0])

    overlays = _collect_overlays(graph_data, nodes)
    for ov in overlays:
        cid = str(ov.get("concept_id") or "").strip()
        if not cid:
            overlay_errors.append("overlay row missing concept_id")
            continue
        if known_subs and cid not in known_subs:
            note_sub_orphan(cid, ctx="overlay.concept_id")
            overlay_errors.append(f"overlay concept_id '{cid}' is not in sub_concepts")
        otype = str(ov.get("overlay_type") or "").strip().upper()
        kind = str(ov.get("overlay_kind") or "").strip().lower()
        if otype and otype not in _VALID_OVERLAY_TYPES:
            overlay_errors.append(
                f"overlay '{cid}': invalid overlay_type {otype!r}"
            )
        if kind and kind not in _VALID_OVERLAY_KINDS and kind not in {
            t.lower() for t in _VALID_OVERLAY_TYPES
        }:
            overlay_errors.append(
                f"overlay '{cid}': invalid overlay_kind {kind!r}"
            )
        parent = str(ov.get("parent_id") or "").strip()
        if bool(ov.get("is_extension")) and parent:
            if known_subs and parent not in known_subs:
                overlay_errors.append(
                    f"overlay '{cid}': parent_id '{parent}' is not in sub_concepts"
                )
        elif bool(ov.get("is_extension")) and not parent and known_subs:
            # Extension without a parent is allowed only when the id itself is known.
            pass

    errors.extend(overlay_errors)
    ok = not errors and not cycles
    return GraphIntegrityReport(
        ok=ok,
        errors=tuple(errors),
        cycles=cycles,
        orphan_node_ids=tuple(orphan_nodes),
        orphan_sub_concept_ids=tuple(orphan_subs),
        overlay_errors=tuple(overlay_errors),
        node_count=len(by_id),
        sub_concept_count=len(known_subs),
    )
