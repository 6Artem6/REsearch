"""Проверка DAG: существующие prerequisites, отсутствие циклов, слои foundation/sota."""

from __future__ import annotations

from knowledge_engine.src.curriculum.schemas import CurriculumGraph, CurriculumNode


def validate_curriculum_dag(graph: CurriculumGraph) -> list[str]:
    """
    Критерии приёмки: валидные prerequisites, DAG без циклов, foundation + sota.
    Пустой список — граф корректен.
    """
    errors: list[str] = []
    nodes = graph.nodes
    if not nodes:
        errors.append("Граф не содержит узлов.")
        return errors

    by_id: dict[str, CurriculumNode] = {n.node_id: n for n in nodes}
    ids = set(by_id.keys())

    has_foundation = False
    has_sota = False
    for n in nodes:
        if n.layer == "foundation":
            has_foundation = True
        elif n.layer == "sota":
            has_sota = True
        for p in n.prerequisites:
            if p not in ids:
                errors.append(
                    f"Узел '{n.node_id}': prerequisite '{p}' не существует в графе."
                )
            if p == n.node_id:
                errors.append(f"Узел '{n.node_id}': self-reference в prerequisites.")

    if not has_foundation:
        errors.append("В графе нет узлов слоя foundation.")
    if not has_sota:
        errors.append("В графе нет узлов слоя sota.")

    cycle = _find_cycle(by_id)
    if cycle:
        errors.append(
            "Циклическая зависимость: "
            + " → ".join(cycle)
            + (" → " + cycle[0] if cycle else "")
        )

    if graph.total_nodes != len(nodes):
        errors.append(
            f"total_nodes ({graph.total_nodes}) не совпадает с числом узлов ({len(nodes)})."
        )

    return errors


def validate_dag_branching(graph: CurriculumGraph) -> list[str]:
    """
    Эвристика: не линейная цепочка 1→2→3… — параллельные foundation-ветки и слияния.
    """
    nodes = graph.nodes
    if len(nodes) < 6:
        return []

    errors: list[str] = []
    foundation = [n for n in nodes if n.layer == "foundation"]
    children: dict[str, int] = {}
    for n in nodes:
        for p in n.prerequisites:
            children[p] = children.get(p, 0) + 1

    roots_foundation = [n for n in foundation if not n.prerequisites]
    foundation_with_children = sum(
        1 for n in foundation if children.get(n.node_id, 0) >= 2
    )
    if len(roots_foundation) < 2 and foundation_with_children < 1:
        errors.append(
            "FOUNDATION: нужны минимум 2–3 параллельные ветки — несколько "
            "foundation-нод без prerequisites ИЛИ одна foundation с 2+ дочерних нод."
        )

    branch_points = sum(1 for c in children.values() if c >= 2)
    if branch_points < 2:
        errors.append(
            "Топология: минимум 2 точки ветвления (один prerequisite у 2+ разных нод)."
        )

    merge_points = sum(1 for n in nodes if len(n.prerequisites) >= 2)
    if merge_points < 1:
        errors.append(
            "Топология: хотя бы одна advanced/sota нода с 2+ prerequisites (слияние веток)."
        )

    # Почти линейная цепочка: >75% нод с ровно одним prerequisite
    single_prereq = sum(1 for n in nodes if len(n.prerequisites) == 1)
    if single_prereq >= max(6, int(0.75 * len(nodes))):
        errors.append(
            "Граф слишком линейный (A→B→C→…). Разветви foundation и добавь merge-ноды."
        )

    return errors


def validate_curriculum_dag_full(graph: CurriculumGraph) -> list[str]:
    """Структурная валидность + ветвление."""
    return validate_curriculum_dag(graph) + validate_dag_branching(graph)


def _find_cycle(by_id: dict[str, CurriculumNode]) -> list[str] | None:
    """DFS: вернуть путь цикла или None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in by_id}
    parent: dict[str, str | None] = {nid: None for nid in by_id}

    def dfs(u: str) -> list[str] | None:
        color[u] = GRAY
        node = by_id[u]
        for v in node.prerequisites:
            if v not in by_id:
                continue
            if color[v] == GRAY:
                path = [v, u]
                cur = u
                while parent[cur] and parent[cur] != v:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            if color[v] == WHITE:
                parent[v] = u
                found = dfs(v)
                if found:
                    return found
        color[u] = BLACK
        return None

    for nid in by_id:
        if color[nid] == WHITE:
            found = dfs(nid)
            if found:
                return found
    return None


def repair_curriculum_dag_cycles(
    graph: CurriculumGraph,
    *,
    prefer_remove_node_ids: set[str] | None = None,
    max_steps: int = 16,
) -> tuple[CurriculumGraph, int]:
    """Снимает prerequisite-рёбра, образующие цикл (при expand — в приоритете new_nodes)."""
    prefer = prefer_remove_node_ids or set()
    removed = 0
    nodes = list(graph.nodes)
    for _ in range(max_steps):
        by_id: dict[str, CurriculumNode] = {n.node_id: n for n in nodes}
        cycle = _find_cycle(by_id)
        if not cycle:
            break
        cycle_set = set(cycle)
        victim: tuple[str, str] | None = None
        for nid in cycle:
            node = by_id[nid]
            for p in node.prerequisites:
                if p not in cycle_set:
                    continue
                if nid in prefer or p in prefer:
                    victim = (nid, p)
                    break
            if victim:
                break
        if not victim:
            nid = cycle[-1]
            node = by_id[nid]
            for p in node.prerequisites:
                if p in cycle_set:
                    victim = (nid, p)
                    break
        if not victim:
            break
        to_id, from_id = victim
        node = by_id[to_id]
        kept = [p for p in node.prerequisites if p != from_id]
        by_id[to_id] = node.model_copy(update={"prerequisites": kept})
        nodes = [by_id[n.node_id] for n in graph.nodes if n.node_id in by_id]
        removed += 1
    if removed:
        out = graph.model_copy(update={"nodes": nodes, "total_nodes": len(nodes)})
        return out, removed
    return graph, 0
