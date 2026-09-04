/**
 * Depth & Overlay — Core WHY/HOW/MECHANICS bars + expert overlay badges.
 * Progress bars use only `sub_concepts` with `is_extension === false`.
 */

import React from "react";
import {
  coreSubConcepts,
  itemFlag,
  overlayAwardsFromCoverage,
  overlayTypeOf,
} from "./nodeProgressTypes.js";

const LAYER_META = [
  { id: "why", key: "WHY", title: "WHY", subtitle: "Зачем / концепция", flag: "why_passed" },
  { id: "how", key: "HOW", title: "HOW", subtitle: "Как / архитектура", flag: "how_passed" },
  {
    id: "mechanic",
    key: "MECHANICS",
    title: "MECHANICS",
    subtitle: "Механики / реализация",
    flag: "mechanic_passed",
  },
];

const LAYER_STATUS_CLS = {
  verified: "depth-layer-passed",
  passed: "depth-layer-passed",
  in_progress: "depth-layer-active",
  pending: "depth-layer-locked",
  locked: "depth-layer-locked",
  failed: "depth-layer-failed",
  gloss: "depth-layer-gloss",
};

const LAYER_STATUS_LABEL = {
  verified: "Зачтено",
  passed: "Зачтено",
  in_progress: "Сейчас",
  pending: "Ещё не пройден",
  locked: "Ещё не пройден",
  failed: "Не зачтено",
  gloss: "Gloss",
};

function normalizeBackendLayers(coverage) {
  const raw = coverage?.layers;
  if (!raw || typeof raw !== "object") return null;
  const pick = (k) => {
    const row = raw[k] || raw[k === "mechanic" ? "MECHANICS" : k] || {};
    return {
      status: String(row.status || "locked").toLowerCase(),
      score: Math.min(1, Math.max(0, Number(row.score) || 0)),
    };
  };
  return { why: pick("why"), how: pick("how"), mechanic: pick("mechanic") };
}

function layersFromCoreItems(items, backendLayers, activeLayer) {
  const core = coreSubConcepts(items);
  if (!core.length) return null;
  const n = core.length;
  const frac = (flag) => core.filter((i) => itemFlag(i, flag)).length / n;
  const active = String(activeLayer || "").trim().toUpperCase();

  const build = (id, flag, key) => {
    const score = frac(flag);
    const be = backendLayers?.[id];
    let status = "pending";
    if (score >= 1) status = "verified";
    else if (
      active === key ||
      (active === "MECHANIC" && key === "MECHANICS") ||
      score > 0
    ) {
      status = "in_progress";
    }
    if (
      id === "mechanic" &&
      score < 1 &&
      (be?.status === "gloss" ||
        (frac("why_passed") >= 1 && frac("how_passed") >= 1))
    ) {
      status = "gloss";
    } else if (be?.status === "passed" && score >= 1) {
      status = "verified";
    }
    return { status, score };
  };

  return {
    why: build("why", "why_passed", "WHY"),
    how: build("how", "how_passed", "HOW"),
    mechanic: build("mechanic", "mechanic_passed", "MECHANICS"),
  };
}

function computeCoreDepthLayers(coverage, items) {
  const active = String(
    coverage?.active_layer || coverage?.activeLayer || "",
  )
    .trim()
    .toUpperCase();
  return layersFromCoreItems(items, normalizeBackendLayers(coverage), active);
}

function DepthLayersStrip({ layers, active }) {
  if (!layers) return null;
  const focus = String(active || "").toUpperCase();
  return React.createElement(
    "div",
    { className: "depth-layers" },
    LAYER_META.map((meta) => {
      const row = layers[meta.id];
      let status = row.status;
      if (
        (focus === meta.key || (focus === "MECHANIC" && meta.key === "MECHANICS")) &&
        (status === "pending" || status === "locked")
      ) {
        status = "in_progress";
      }
      const cls = LAYER_STATUS_CLS[status] || LAYER_STATUS_CLS.pending;
      return React.createElement(
        "div",
        {
          key: meta.id,
          className: `depth-layer ${cls}`,
          title: `${meta.subtitle}: ${LAYER_STATUS_LABEL[status] || status}`,
        },
        React.createElement(
          "div",
          { className: "depth-layer-head" },
          React.createElement("span", { className: "depth-layer-title" }, meta.title),
          React.createElement(
            "span",
            { className: "depth-layer-badge" },
            LAYER_STATUS_LABEL[status] || status,
          ),
        ),
        React.createElement("div", { className: "depth-layer-sub" }, meta.subtitle),
        React.createElement("div", {
          className: "depth-layer-meter",
          "aria-hidden": true,
          children: React.createElement("div", {
            className: "depth-layer-meter-fill",
            style: { width: `${Math.round(row.score * 100)}%` },
          }),
        }),
      );
    }),
  );
}

function OverlayExpertLayers({ score, coverage }) {
  if (Number(score) !== 100) return null;
  const awards = overlayAwardsFromCoverage(coverage);
  const types = new Set(awards.map(overlayTypeOf));
  const hasAdv = types.has("ADVANCED_ASTERISK");
  const hasDeep = types.has("DEEP_ASTERISK");
  return React.createElement(
    "div",
    { className: "overlay-expert-layers", "aria-label": "Дополнительные экспертные слои" },
    React.createElement(
      "div",
      { className: "coverage-widget-head" },
      React.createElement(
        "span",
        { className: "coverage-label" },
        "Дополнительные экспертные слои",
      ),
    ),
    React.createElement(
      "div",
      { className: "overlay-expert-badges" },
      hasAdv
        ? React.createElement(
            "span",
            { className: "overlay-expert-badge overlay-expert-advanced" },
            "ADVANCED",
          )
        : null,
      hasDeep
        ? React.createElement(
            "span",
            { className: "overlay-expert-badge overlay-expert-deep" },
            "DEEP DESIGN",
          )
        : null,
      !hasAdv &&
        !hasDeep &&
        React.createElement(
          "span",
          { className: "overlay-expert-empty muted" },
          "Доступны задачки со звёздочкой",
        ),
    ),
  );
}

export function DepthSection({ coverage, items, score }) {
  const rows = items || coverage?.items || [];
  const core = coreSubConcepts(rows);
  if (!core.length && Number(score) !== 100) return null;
  const active = String(
    coverage?.active_layer || coverage?.activeLayer || "",
  )
    .trim()
    .toUpperCase();
  const layers = computeCoreDepthLayers(coverage, rows);
  return React.createElement(
    "div",
    { className: "depth-section" },
    layers
      ? React.createElement(DepthLayersStrip, { layers, active })
      : null,
    React.createElement(OverlayExpertLayers, { score, coverage }),
  );
}
