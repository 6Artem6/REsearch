import React from "react";
import { Handle, Position } from "@xyflow/react";

function layerClass(layer) {
  if (layer === "sota") return "layer-sota";
  if (layer === "advanced") return "layer-advanced";
  return "layer-foundation";
}

function layerLabel(layer) {
  if (layer === "sota") return "SOTA";
  if (layer === "advanced") return "Advanced";
  return "Foundation";
}

export function SkillNode({ data }) {
  const layer = data.layer || "foundation";
  const status = data.status || "unexplored";
  const selected = data.selected;
  const masteryPct = Math.min(
    100,
    Math.max(0, Number(data.masteryPct) || 0),
  );
  const barPct = masteryPct > 0 ? Math.max(masteryPct, 4) : 0;
  const classes = [
    "skill-node-card",
    layerClass(layer),
    `status-${status}`,
    selected ? "selected" : "",
  ].join(" ");

  const layerLabelText = layerLabel(layer);

  return React.createElement(
    "div",
    { className: classes },
    React.createElement(Handle, {
      type: "target",
      position: Position.Top,
      style: { opacity: 0.4 },
    }),
    React.createElement("div", { className: "layer-badge" }, layerLabelText),
    React.createElement("div", { className: "node-title" }, data.label),
    React.createElement("div", { className: "node-cat" }, data.category || ""),
    masteryPct > 0 &&
      React.createElement(
        "div",
        { className: "node-mastery-row" },
        React.createElement(
          "div",
          { className: "node-mastery-bar" },
          React.createElement("div", {
            className: "node-mastery-fill",
            style: { width: `${barPct}%` },
          }),
        ),
        React.createElement("span", { className: "node-mastery-pct" }, `${masteryPct}%`),
      ),
    React.createElement(Handle, {
      type: "source",
      position: Position.Bottom,
      style: { opacity: 0.4 },
    }),
  );
}

export const skillNodeTypes = { skillNode: SkillNode };