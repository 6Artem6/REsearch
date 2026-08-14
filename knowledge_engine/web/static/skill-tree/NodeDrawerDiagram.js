import React from "react";
import { MermaidDiagramView } from "./MermaidDiagramView.js";

/** Упрощённый блок схемы внутри списка материалов. */
export function DiagramBlock({ diagram, nodeId, compact }) {
  const text = String(diagram || "").trim();
  if (!text) return null;
  return React.createElement(
    "div",
    { className: compact ? "diagram-embed compact" : "diagram-embed" },
    React.createElement(MermaidDiagramView, {
      diagram: text,
      nodeId,
      compact,
      hideTitle: compact,
    }),
  );
}
