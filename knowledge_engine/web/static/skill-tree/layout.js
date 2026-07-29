import Dagre from "@dagrejs/dagre";

const NODE_W = 200;
const NODE_H = 72;

export function layoutFlowNodes(curriculumNodes, flowNodes, flowEdges) {
  const g = new Dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 48, ranksep: 64 });

  flowNodes.forEach((n) => {
    g.setNode(n.id, { width: NODE_W, height: NODE_H });
  });
  flowEdges.forEach((e) => {
    g.setEdge(e.source, e.target);
  });

  Dagre.layout(g);

  return flowNodes.map((n) => {
    const pos = g.node(n.id);
    return {
      ...n,
      position: {
        x: pos.x - NODE_W / 2,
        y: pos.y - NODE_H / 2,
      },
    };
  });
}

export function curriculumToFlow(curriculum, statuses, selectedId) {
  const nodes = (curriculum.nodes || []).map((n) => {
    const st = statuses[n.node_id] || "unexplored";
    return {
      id: n.node_id,
      type: "skillNode",
      position: { x: 0, y: 0 },
      data: {
        label: n.title,
        category: n.category,
        layer: n.layer,
        status: st,
        selected: n.node_id === selectedId,
        raw: n,
      },
    };
  });

  const edges = [];
  for (const n of curriculum.nodes || []) {
    for (const p of n.prerequisites || []) {
      edges.push({
        id: `${p}->${n.node_id}`,
        source: p,
        target: n.node_id,
        animated: statuses[n.node_id] === "in_progress",
      });
    }
  }
  return { nodes, edges };
}
