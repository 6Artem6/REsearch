/**
 * Contextual Quick Reply chips under the last tutor message
 * (only when ready_for_transition). Host `quick_replies` is the SSOT;
 * client reconstruction is a legacy fallback.
 */

import React, { useMemo } from "react";
import {
  ActionChips,
  HOST_CHIP_LABELS,
  QUICK_REPLY_INTENTS,
  chipsFromHostLabels,
  hostChipLabelsFromSession,
} from "./ActionChips.js";

export { HOST_CHIP_LABELS, QUICK_REPLY_INTENTS, chipsFromHostLabels };

/** Nodes that list `nodeId` as a prerequisite (forward edges). */
export function listSuccessorNodes(curriculum, nodeId) {
  const nid = String(nodeId || "").trim();
  if (!nid || !curriculum?.nodes?.length) return [];
  return (curriculum.nodes || []).filter((n) => {
    const id = String(n.node_id || "").trim();
    if (!id || id === nid) return false;
    const prereqs = n.prerequisites || [];
    return prereqs.some((p) => String(p || "").trim() === nid);
  });
}

export function normalizeNodeLayer(layer) {
  const raw = String(layer || "foundation").trim().toLowerCase();
  if (["fundamental", "base", "intro", "foundation"].includes(raw)) {
    return "foundation";
  }
  if (["adv", "advanced"].includes(raw)) return "advanced";
  if (
    ["sota", "state_of_the_art", "state-of-the-art", "deep_mastery"].includes(raw)
  ) {
    return "sota";
  }
  return "foundation";
}

/** Optional depth layers that may remain open after threshold. */
export function optionalDepthLayers(nodeLayer) {
  const ly = normalizeNodeLayer(nodeLayer);
  if (ly === "foundation") return ["HOW", "MECHANIC"];
  if (ly === "advanced") return ["MECHANIC"];
  return [];
}

function aggregateFlags(session) {
  const items = session?.coverageSummary?.items || [];
  if (!items.length) {
    const layers = session?.coverageSummary?.layers;
    if (!layers) return { why: false, how: false, mech: false };
    return {
      why: Number(layers.why?.score ?? 0) >= 1,
      how: Number(layers.how?.score ?? 0) >= 1,
      mech: Number(layers.mechanic?.score ?? 0) >= 1,
    };
  }
  return {
    why: items.every((it) => Boolean(it.why_passed)),
    how: items.every((it) => Boolean(it.how_passed)),
    mech: items.every((it) => Boolean(it.mechanic_passed)),
  };
}

/**
 * Open optional layers for this node difficulty (empty for SotA).
 * @returns {string[]} e.g. ["HOW"], ["MECHANIC"], ["HOW","MECHANIC"]
 */
export function openOptionalLayers(session, nodeLayer) {
  const ly = normalizeNodeLayer(nodeLayer);
  const opts = optionalDepthLayers(ly);
  if (!opts.length) return [];
  const { how, mech } = aggregateFlags(session);
  const open = [];
  if (opts.includes("HOW") && !how) open.push("HOW");
  if (opts.includes("MECHANIC") && !mech) open.push("MECHANIC");
  if (!open.length) {
    const items = session?.coverageSummary?.items || [];
    for (const it of items) {
      if (opts.includes("HOW") && it.why_passed && !it.how_passed) open.push("HOW");
      if (
        opts.includes("MECHANIC") &&
        it.why_passed &&
        (it.how_passed || ly === "foundation") &&
        !it.mechanic_passed
      ) {
        open.push("MECHANIC");
      }
    }
    return [...new Set(open)];
  }
  return open;
}

export function isFullDepthClosure(session, nodeLayer) {
  const { why, how, mech } = aggregateFlags(session);
  return Boolean(why && how && mech);
}

/**
 * @param {object} session
 * @param {string} [nodeLayer]
 * @returns {{ id: string, label: string, intent: string }[]}
 */
export function buildTransitionChips(session, nodeLayer) {
  const host = chipsFromHostLabels(hostChipLabelsFromSession(session));
  if (!session?.readyForTransition) {
    return host;
  }
  if (host.length) return host;

  const ly = normalizeNodeLayer(nodeLayer || session?.nodeLayer);
  const open = openOptionalLayers(session, ly);
  const full = isFullDepthClosure(session, ly) || ly === "sota";
  const score = Number(session?.topicMasteryScore ?? 0);

  if (full || open.length === 0) {
    const next = {
      id: "next",
      label: HOST_CHIP_LABELS.next,
      intent: QUICK_REPLY_INTENTS.nextNode,
    };
    if (score < 100 && ly !== "sota") return [next];
    return [
      {
        id: "deep_design",
        label: "Архитектурный дизайн",
        intent: QUICK_REPLY_INTENTS.deepDesign,
      },
      next,
    ];
  }

  const pushLayer = open[0];
  const pushHow = pushLayer === "HOW";
  return [
    {
      id: "gloss",
      label: HOST_CHIP_LABELS.gloss,
      intent: QUICK_REPLY_INTENTS.gloss,
    },
    {
      id: "push",
      label: pushHow ? HOST_CHIP_LABELS.how : HOST_CHIP_LABELS.mech,
      intent: pushHow ? QUICK_REPLY_INTENTS.how : QUICK_REPLY_INTENTS.mech,
    },
    {
      id: "next",
      label: HOST_CHIP_LABELS.next,
      intent: QUICK_REPLY_INTENTS.nextNode,
    },
  ];
}

export function QuickReplyChips({
  visible,
  session,
  nodeLayer,
  disabled,
  onChip,
}) {
  const chips = useMemo(
    () => (visible ? buildTransitionChips(session, nodeLayer) : []),
    [visible, session, nodeLayer, session?.quickReplies, session?.suggestedChips, session?.topicMasteryScore],
  );
  return React.createElement(ActionChips, {
    visible,
    chips,
    disabled,
    onChip,
  });
}

export function NextNodeSelector({ open, nodes, onSelect, onClose }) {
  if (!open) return null;
  const list = nodes || [];
  return React.createElement(
    "div",
    {
      className: "next-node-selector-backdrop",
      role: "presentation",
      onClick: (e) => {
        if (e.target === e.currentTarget) onClose?.();
      },
    },
    React.createElement(
      "div",
      {
        className: "next-node-selector",
        role: "dialog",
        "aria-modal": "true",
        "aria-label": "Выбор следующей ноды",
      },
      React.createElement(
        "div",
        { className: "next-node-selector-head" },
        React.createElement("strong", null, "Следующая нода"),
        React.createElement(
          "button",
          {
            type: "button",
            className: "next-node-selector-close",
            onClick: () => onClose?.(),
            "aria-label": "Закрыть",
          },
          "×",
        ),
      ),
      list.length === 0
        ? React.createElement(
            "p",
            { className: "muted next-node-selector-empty" },
            "Нет смежных нод в графе. Выберите следующую тему на карте навыков.",
          )
        : React.createElement(
            "ul",
            { className: "next-node-selector-list" },
            list.map((n) =>
              React.createElement(
                "li",
                { key: n.node_id },
                React.createElement(
                  "button",
                  {
                    type: "button",
                    className: "next-node-selector-item",
                    onClick: () => onSelect?.(n),
                  },
                  React.createElement(
                    "span",
                    { className: "next-node-selector-title" },
                    n.title || n.node_id,
                  ),
                  n.layer
                    ? React.createElement(
                        "span",
                        { className: "muted next-node-selector-layer" },
                        n.layer,
                      )
                    : null,
                ),
              ),
            ),
          ),
    ),
  );
}
