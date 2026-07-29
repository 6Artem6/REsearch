import React from "react";
import { Handle, Position } from "@xyflow/react";

export function SkillNode({ data }) {
  const layer = data.layer || "foundation";
  const status = data.status || "unexplored";
  const selected = data.selected;
  const classes = [
    "skill-node-card",
    layer === "sota" ? "layer-sota" : "layer-foundation",
    `status-${status}`,
    selected ? "selected" : "",
  ].join(" ");

  const layerLabel =
    layer === "sota" ? "⚡ SOTA / Advanced" : "Foundation";

  return React.createElement(
    "div",
    { className: classes },
    React.createElement(Handle, {
      type: "target",
      position: Position.Top,
      style: { opacity: 0.4 },
    }),
    React.createElement("div", { className: "layer-badge" }, layerLabel),
    React.createElement("div", { className: "node-title" }, data.label),
    React.createElement("div", { className: "node-cat" }, data.category || ""),
    React.createElement(Handle, {
      type: "source",
      position: Position.Bottom,
      style: { opacity: 0.4 },
    }),
  );
}

export const skillNodeTypes = { skillNode: SkillNode };
