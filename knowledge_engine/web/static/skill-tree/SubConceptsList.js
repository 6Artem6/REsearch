/**
 * Sub-concept list for the right drawer: Core + dynamic extensions.
 */

import React, { useEffect, useRef, useState } from "react";
import {
  isExtensionSubConcept,
  itemFlag,
  layerAccuracyGrade,
  layerBadgeIcon,
  layerBadgeTitle,
  resolveProbeLayer,
  subConceptTitle,
} from "./nodeProgressTypes.js";

const GRADE_CHIP_CLASS = {
  EXACT_AND_CORRECT: "is-exact",
  PARTIAL: "is-partial",
  NEEDS_CORRECTION: "is-correction",
  MISUNDERSTANDING: "is-misunderstanding",
};

function layerBadgeClass(label, grade) {
  const base = `subtopic-layer-chip layer-${String(label).toLowerCase()}`;
  if (grade === "EXACT_AND_CORRECT") return `${base} is-passed is-exact`;
  return `${base} ${GRADE_CHIP_CLASS[grade] || "is-unevaluated"}`;
}

function itemAccuracyGrade(item) {
  return String(
    item?.last_accuracy_grade || item?.lastAccuracyGrade || "",
  ).trim();
}

function LayerBadge({ passed, label, probeLayer, lastAccuracyGrade }) {
  const short = label === "MECHANIC" || label === "MECHANICS" ? "MECH" : label;
  const gradeOpts = { passed, label, probeLayer, lastAccuracyGrade };
  const grade = layerAccuracyGrade(gradeOpts);
  const icon = layerBadgeIcon(gradeOpts);
  return React.createElement(
    "span",
    {
      className: layerBadgeClass(label, grade),
      title: layerBadgeTitle(short, icon, grade),
    },
    React.createElement(
      "span",
      { className: "subtopic-layer-chip-mark", "aria-hidden": true },
      icon,
    ),
    React.createElement("span", { className: "subtopic-layer-chip-text" }, short),
  );
}

function subtopicState(item) {
  return String(item?.state || item?.status || "unchecked")
    .trim()
    .toLowerCase();
}

function subtopicHint(item) {
  const fromBackend = (item.status_hint || item.statusHint || "").trim();
  if (fromBackend) return fromBackend;
  const why = itemFlag(item, "why_passed");
  const how = itemFlag(item, "how_passed");
  const mech = itemFlag(item, "mechanic_passed");
  if (why && how && mech) return null;
  if (why && how && !mech) return "Не хватает механик реализации";
  if (why && !how) return "Не хватает архитектуры (HOW)";
  if (!why && (how || mech)) return "Концепция (WHY) не раскрыта";
  if (!why && !how && !mech) return "Ещё не затронута";
  return null;
}

export function SubConceptsList({
  items,
  sub_concepts: subConceptsProp,
  lastEvalDirective,
  activeLayer,
}) {
  const rows = subConceptsProp || items || [];
  const prevIdsRef = useRef(new Set());
  const [entering, setEntering] = useState(() => new Set());
  const probeLayer = resolveProbeLayer({ lastEvalDirective, activeLayer });

  useEffect(() => {
    const ids = rows
      .map((row) => String(row?.id || "").trim())
      .filter(Boolean);
    const prev = prevIdsRef.current;
    const fresh = ids.filter((id) => prev.size > 0 && !prev.has(id));
    prevIdsRef.current = new Set(ids);
    if (!fresh.length) return undefined;
    setEntering(new Set(fresh));
    const timer = window.setTimeout(() => setEntering(new Set()), 400);
    return () => window.clearTimeout(timer);
  }, [rows]);

  if (!rows.length) return null;

  return React.createElement(
    "div",
    { className: "coverage-subtopics" },
    React.createElement(
      "div",
      { className: "coverage-widget-head" },
      React.createElement("span", { className: "coverage-label" }, "Подтемы"),
      React.createElement(
        "span",
        { className: "coverage-ratio muted" },
        `${rows.length}`,
      ),
    ),
    React.createElement(
      "ul",
      { className: "coverage-subtopic-list" },
      rows.map((item) => {
        const why = itemFlag(item, "why_passed");
        const how = itemFlag(item, "how_passed");
        const mech = itemFlag(item, "mechanic_passed");
        const hint = subtopicHint(item);
        const st = subtopicState(item);
        const id = String(item.id || subConceptTitle(item));
        const extension = isExtensionSubConcept(item);
        const enteringCls = entering.has(id) ? " is-entering" : "";
        const extCls = extension ? " is-extension" : "";
        const lastAccuracyGrade = itemAccuracyGrade(item);
        const badgeProps = { probeLayer, lastAccuracyGrade };
        return React.createElement(
          "li",
          {
            key: id,
            className: `coverage-subtopic coverage-subtopic-${st}${extCls}${enteringCls}`,
          },
          React.createElement(
            "div",
            { className: "coverage-subtopic-row" },
            React.createElement(
              "span",
              { className: "coverage-subtopic-label" },
              subConceptTitle(item),
              extension
                ? React.createElement(
                    "span",
                    { className: "subtopic-extension-badge" },
                    "[Расширение]",
                  )
                : null,
            ),
            React.createElement(
              "span",
              { className: "coverage-subtopic-badges" },
              React.createElement(LayerBadge, {
                passed: why,
                label: "WHY",
                ...badgeProps,
              }),
              React.createElement(LayerBadge, {
                passed: how,
                label: "HOW",
                ...badgeProps,
              }),
              React.createElement(LayerBadge, {
                passed: mech,
                label: "MECHANIC",
                ...badgeProps,
              }),
            ),
          ),
          hint &&
            React.createElement(
              "p",
              { className: "coverage-subtopic-hint" },
              hint,
            ),
        );
      }),
    ),
  );
}
