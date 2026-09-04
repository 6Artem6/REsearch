import React from "react";
import { SubConceptsList } from "./SubConceptsList.js";
import { coreSubConcepts, itemFlag } from "./nodeProgressTypes.js";

const PHASE_LABELS = {
  intro_assessment: "Экспресс-срез",
  dense_material: "Плотный материал",
  checkpoint: "Проверка",
  pathway_decision: "Выбор пути",
  socratic_focus: "Точечный Сократ",
};

const MODE_LABELS = {
  lecture: "Лекция",
  express_blitz: "Экспресс-блиц",
  socratic_point: "Сократ (точечно)",
};

function layerOverallScore(layers) {
  if (!layers) return null;
  return Math.round(
    (100 * (layers.why.score + layers.how.score + layers.mechanic.score)) / 3.0,
  );
}

function layersFromCoreItems(items, activeLayer) {
  const core = coreSubConcepts(items);
  if (!core.length) return null;
  const n = core.length;
  const frac = (flag) => core.filter((i) => itemFlag(i, flag)).length / n;
  const active = String(activeLayer || "").trim().toUpperCase();
  const build = (flag, key) => {
    const score = frac(flag);
    let status = "pending";
    if (score >= 1) status = "verified";
    else if (active === key || score > 0) status = "in_progress";
    return { status, score };
  };
  return {
    why: build("why_passed", "WHY"),
    how: build("how_passed", "HOW"),
    mechanic: build("mechanic_passed", "MECHANICS"),
  };
}

/**
 * Core mastery % for the node panel and map.
 * Denominator is Core sub-concepts only (`is_extension === false`).
 */
export function resolveMasteryScore(masteryDashboard, topicMasteryScore) {
  const dash = masteryDashboard || {};
  const coverage = dash.coverage_summary || dash.coverageSummary || null;
  const items = coverage?.items || [];
  const active = String(
    coverage?.active_layer || coverage?.activeLayer || "",
  )
    .trim()
    .toUpperCase();
  const layers = items.length ? layersFromCoreItems(items, active) : null;
  const layerScore = layerOverallScore(layers);
  const backendOverall =
    coverage && coverage.overall_score != null
      ? Number(coverage.overall_score)
      : coverage && coverage.overallScore != null
        ? Number(coverage.overallScore)
        : null;
  const score = layers
    ? Number.isFinite(backendOverall) && backendOverall >= 0
      ? backendOverall
      : Number.isFinite(layerScore)
        ? layerScore
        : 0
    : Math.max(
        Number(topicMasteryScore) || 0,
        Number(dash.topic_mastery_score) || 0,
      );

  return {
    score: Math.min(100, Math.max(0, score)),
    coverage,
    items,
    active,
    layers,
  };
}

function FactsPreview({ coverage, items }) {
  const flat = coverage?.facts_breakdown || coverage?.factsBreakdown || [];
  const fromItems = (items || []).flatMap((it) => it.facts || []);
  const facts = [...flat, ...fromItems];
  if (!facts.length) return null;
  return React.createElement(
    "div",
    { className: "coverage-facts-preview" },
    React.createElement(
      "div",
      { className: "coverage-widget-head" },
      React.createElement("span", { className: "coverage-label" }, "Факты"),
      React.createElement(
        "span",
        { className: "coverage-ratio muted" },
        `${facts.length}`,
      ),
    ),
    React.createElement(
      "ul",
      { className: "coverage-facts-list" },
      facts.slice(0, 12).map((f) =>
        React.createElement(
          "li",
          {
            key: f.fact_id || f.factId || f.statement,
            className: `coverage-fact status-${f.status || "pending"}`,
          },
          React.createElement(
            "span",
            { className: "coverage-fact-layer" },
            f.layer || "WHY",
          ),
          " ",
          (f.statement || "").slice(0, 120),
        ),
      ),
    ),
  );
}

function CoverageWidget({ coverage, score, lastEvalDirective }) {
  const items = coverage?.items || [];
  if (!items.length && Number(score) !== 100) return null;
  const glossHint = (coverage?.gloss_hint || coverage?.glossHint || "").trim();
  const activeLayer = String(
    coverage?.active_layer || coverage?.activeLayer || "",
  )
    .trim()
    .toUpperCase();
  const layers = layersFromCoreItems(items, activeLayer);
  const showGloss =
    Boolean(glossHint) ||
    (layers &&
      layers.why.status === "verified" &&
      layers.how.status === "verified" &&
      layers.mechanic.status !== "verified");

  return React.createElement(
    "div",
    { className: "coverage-widget coverage-widget-depth" },
    React.createElement(SubConceptsList, {
      items,
      lastEvalDirective,
      activeLayer,
    }),
    showGloss &&
      React.createElement(
        "p",
        { className: "depth-gloss-hint" },
        glossHint ||
          "Концепция зачтена. Дополните механики реализации или изучите Gloss",
      ),
    React.createElement(FactsPreview, { coverage, items }),
  );
}

export function NodeMasteryPanel({
  status,
  masteryDashboard,
  topicMasteryScore,
  lastEvalDirective,
  onModeSelect,
  disabled,
}) {
  const dash = masteryDashboard || {};
  const { coverage, score } = resolveMasteryScore(
    masteryDashboard,
    topicMasteryScore,
  );
  const barPct = score > 0 ? Math.max(score, 4) : 0;
  const phase = dash.learning_phase || "intro_assessment";
  const mode = dash.learning_mode || "lecture";
  const st = dash.node_status || status || "unexplored";

  function modeBtn(id, label, prefix) {
    return React.createElement(
      "button",
      {
        type: "button",
        className: `mastery-mode-btn${mode === id ? " active" : ""}`,
        disabled,
        title: disabled ? "Дождитесь загрузки ноды" : undefined,
        onClick: () => {
          if (disabled) return;
          onModeSelect?.(prefix);
        },
      },
      label,
    );
  }

  return React.createElement(
    "div",
    { className: "mastery-panel" },
    React.createElement(
      "div",
      { className: "mastery-progress-row" },
      React.createElement("span", { className: "mastery-pct" }, `${score}%`),
      React.createElement(
        "div",
        {
          className: "mastery-bar",
          title: "Core: WHY · HOW · MECHANICS",
        },
        React.createElement("div", {
          className: "mastery-bar-fill",
          style: { width: `${barPct}%` },
        }),
      ),
      React.createElement("span", { className: "chip mastery-status" }, st),
    ),
    React.createElement(
      "p",
      { className: "mastery-phase" },
      PHASE_LABELS[phase] || phase,
      " · ",
      MODE_LABELS[mode] || mode,
    ),
    React.createElement(CoverageWidget, {
      coverage,
      score,
      lastEvalDirective,
    }),
    (dash.strengths || []).length > 0 &&
      React.createElement(
        "div",
        { className: "mastery-zone mastery-zone-ok" },
        React.createElement("h4", null, "Сильные стороны"),
        React.createElement(
          "ul",
          null,
          dash.strengths.map((s, i) => React.createElement("li", { key: i }, s)),
        ),
      ),
    (dash.polish_zones || dash.weaknesses || []).length > 0 &&
      React.createElement(
        "div",
        { className: "mastery-zone mastery-zone-warn" },
        React.createElement("h4", null, "Слабые стороны"),
        React.createElement(
          "ul",
          null,
          (dash.weaknesses || dash.polish_zones || []).map((s, i) =>
            React.createElement("li", { key: i }, s),
          ),
        ),
      ),
    (dash.critical_gaps || []).length > 0 &&
      React.createElement(
        "div",
        { className: "mastery-zone mastery-zone-gap" },
        React.createElement("h4", null, "Критические пробелы"),
        React.createElement(
          "ul",
          null,
          dash.critical_gaps.map((s, i) => React.createElement("li", { key: i }, s)),
        ),
      ),
    (dash.pathway_bridge || "").trim() &&
      React.createElement(
        "p",
        { className: "mastery-bridge muted" },
        dash.pathway_bridge,
      ),
    React.createElement("h4", { className: "mastery-modes-label" }, "Режим работы"),
    React.createElement(
      "div",
      { className: "mastery-modes" },
      modeBtn("lecture", "Лекция", "[mode:lecture] Дай плотный материал по теме."),
      modeBtn("express_blitz", "Блиц", "[mode:blitz] Один экспресс-вопрос."),
      modeBtn(
        "socratic_point",
        "Сократ",
        "[mode:socratic] Точечный разбор моего пробела.",
      ),
    ),
  );
}
