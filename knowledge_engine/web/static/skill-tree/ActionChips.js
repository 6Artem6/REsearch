/**
 * Bottom action chips. Host `quick_replies` / `suggested_chips` is SSOT.
 * Overlay chips send `[mode:advanced_analysis] Анализ уязвимостей` /
 * `[mode:deep_design] Архитектурный дизайн` so the UI can strip the tag
 * and keep the visible user bubble.
 */

import React, { useMemo } from "react";

/** Short UI labels for overlay chips (no asterisk glyph). Also the visible body after `[mode:…]`. */
export const OVERLAY_CHIP_DISPLAY = {
  advanced_analysis: "Анализ уязвимостей",
  deep_design: "Архитектурный дизайн",
};

/** Messages sent to the tutor — Prompt Factory `[mode:…]` prefixes plus visible body. */
export const QUICK_REPLY_INTENTS = {
  gloss:
    "[mode:gloss] Сформируй сжатую выжимку (Glossary) по оставшимся слоям.",
  how: "[mode:deep_dive_how] Разбери архитектуру темы.",
  mech: "[mode:deep_dive_mech] Разбери механики и код темы.",
  advancedAnalysis: `[mode:advanced_analysis] ${OVERLAY_CHIP_DISPLAY.advanced_analysis}`,
  deepDesign: `[mode:deep_design] ${OVERLAY_CHIP_DISPLAY.deep_design}`,
  nextNode: "next_node_ui",
};

export const HOST_CHIP_LABELS = {
  gloss: "Хочу Gloss",
  how: "Дожать HOW",
  mech: "Дожать MECH",
  advancedAnalysis: "Анализ уязвимостей (задачка со звёздочкой)",
  deepDesign: "Архитектурный дизайн (сложная звёздочка)",
  next: "Идем дальше",
};

const HOST_CHIP_BY_LABEL = {
  [HOST_CHIP_LABELS.gloss]: {
    id: "gloss",
    intent: QUICK_REPLY_INTENTS.gloss,
  },
  [HOST_CHIP_LABELS.how]: { id: "push", intent: QUICK_REPLY_INTENTS.how },
  [HOST_CHIP_LABELS.mech]: { id: "push", intent: QUICK_REPLY_INTENTS.mech },
  [HOST_CHIP_LABELS.advancedAnalysis]: {
    id: "advanced_analysis",
    intent: QUICK_REPLY_INTENTS.advancedAnalysis,
  },
  [OVERLAY_CHIP_DISPLAY.advanced_analysis]: {
    id: "advanced_analysis",
    intent: QUICK_REPLY_INTENTS.advancedAnalysis,
  },
  [HOST_CHIP_LABELS.deepDesign]: {
    id: "deep_design",
    intent: QUICK_REPLY_INTENTS.deepDesign,
  },
  [OVERLAY_CHIP_DISPLAY.deep_design]: {
    id: "deep_design",
    intent: QUICK_REPLY_INTENTS.deepDesign,
  },
  "Задачка со звёздочкой": {
    id: "deep_design",
    intent: QUICK_REPLY_INTENTS.deepDesign,
  },
  "Задачка со звездочкой": {
    id: "deep_design",
    intent: QUICK_REPLY_INTENTS.deepDesign,
  },
  [HOST_CHIP_LABELS.next]: {
    id: "next",
    intent: QUICK_REPLY_INTENTS.nextNode,
  },
  "Идём дальше": { id: "next", intent: QUICK_REPLY_INTENTS.nextNode },
  практика: { id: "practice", intent: "практика" },
  проверка: { id: "check", intent: "проверка" },
  пропустить: { id: "skip", intent: "пропустить" },
};

/**
 * Map Host `quick_replies` / `suggested_chips` strings to UI chips.
 */
export function chipsFromHostLabels(labels) {
  const out = [];
  const seen = new Set();
  for (const raw of labels || []) {
    const label = String(raw || "").trim();
    if (!label || seen.has(label)) continue;
    seen.add(label);
    const mapped = HOST_CHIP_BY_LABEL[label];
    if (mapped) {
      const display =
        OVERLAY_CHIP_DISPLAY[mapped.id] || label;
      out.push({ ...mapped, label: display, hostLabel: label });
      continue;
    }
    if (/advanced_analysis/i.test(label) || /анализ уязвимостей/i.test(label)) {
      out.push({
        id: "advanced_analysis",
        label: OVERLAY_CHIP_DISPLAY.advanced_analysis,
        intent: QUICK_REPLY_INTENTS.advancedAnalysis,
        hostLabel: label,
      });
      continue;
    }
    if (/deep_design/i.test(label) || /архитектурный дизайн/i.test(label)) {
      out.push({
        id: "deep_design",
        label: OVERLAY_CHIP_DISPLAY.deep_design,
        intent: QUICK_REPLY_INTENTS.deepDesign,
        hostLabel: label,
      });
      continue;
    }
    out.push({ id: `host-${out.length}`, label, intent: label, hostLabel: label });
  }
  return out;
}

export function hostChipLabelsFromSession(session) {
  if (!session) return [];
  const a = session.quickReplies || session.quick_replies || [];
  const b = session.suggestedChips || session.suggested_chips || [];
  return [...(Array.isArray(a) ? a : []), ...(Array.isArray(b) ? b : [])];
}

export function ActionChips({ chips, disabled, onChip, visible = true }) {
  const list = useMemo(() => chips || [], [chips]);
  if (!visible || !list.length) return null;
  return React.createElement(
    "div",
    {
      className: "tutor-quick-replies",
      role: "group",
      "aria-label": "Дальнейшие шаги",
    },
    list.map((chip) =>
      React.createElement(
        "button",
        {
          key: chip.id + (chip.label || ""),
          type: "button",
          className: `tutor-quick-reply-chip tutor-quick-reply-${chip.id}`,
          disabled: Boolean(disabled),
          onClick: () => onChip?.(chip),
        },
        chip.label,
      ),
    ),
  );
}
