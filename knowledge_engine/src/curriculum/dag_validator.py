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
