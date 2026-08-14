import React from "react";

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

const LAYER_META = [
  { id: "why", key: "WHY", title: "WHY", subtitle: "Зачем / концепция", flag: "why_passed" },
  { id: "how", key: "HOW", title: "HOW", subtitle: "Как / архитектура", flag: "how_passed" },
  {
    id: "mechanic",
    key: "MECHANIC",
    title: "MECHANICS",
    subtitle: "Механики / реализация",
    flag: "mechanic_passed",
  },
];

const LAYER_STATUS_CLS = {
  passed: "depth-layer-passed",
  in_progress: "depth-layer-active",
  locked: "depth-layer-locked",
  failed: "depth-layer-failed",
  gloss: "depth-layer-gloss",
};

const LAYER_STATUS_LABEL = {
  passed: "Зачтено",
  in_progress: "Сейчас",
  locked: "Ещё не пройден",
  failed: "Не зачтено",
  gloss: "Gloss",
};

function itemFlag(item, flag) {
  if (typeof item?.[flag] === "boolean") return item[flag];
  // snake / camel
  const camel = flag.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
  if (typeof item?.[camel] === "boolean") return item[camel];
  return false;
}

/**
 * Aggregate WHY/HOW/MECHANIC from coverage.items[] (source of truth for UI).
 * Backend `layers` used only as status hint when present.
 */
function layersFromItems(items, backendLayers, activeLayer) {
  if (!items?.length) return null;
  const n = items.length;
  const frac = (flag) => items.filter((i) => itemFlag(i, flag)).length / n;
  const active = String(activeLayer || "").trim().toUpperCase();

  const build = (id, flag, key) => {
    const score = frac(flag);
    const be = backendLayers?.[id];
    let status = "locked";
    if (score >= 1) status = "passed";
    else if (active === key || score > 0) status = "in_progress";
    // Prefer backend gloss when WHY+HOW closed and mechanic incomplete
    if (
      id === "mechanic" &&
      score < 1 &&
      (be?.status === "gloss" ||
        (frac("why_passed") >= 1 && frac("how_passed") >= 1))
    ) {
      status = "gloss";
    } else if (be?.status === "passed" && score >= 1) {
      status = "passed";
    }
    return { status, score };
  };

  return {
    why: build("why", "why_passed", "WHY"),
    how: build("how", "how_passed", "HOW"),
    mechanic: build("mechanic", "mechanic_passed", "MECHANIC"),
  };
}

function normalizeBackendLayers(coverage) {
  const raw = coverage?.layers;
  if (!raw || typeof raw !== "object") return null;
  const pick = (k) => {
    const row = raw[k] || {};
    return {
      status: String(row.status || "locked").toLowerCase(),
      score: Math.min(1, Math.max(0, Number(row.score) || 0)),
    };
  };
  return { why: pick("why"), how: pick("how"), mechanic: pick("mechanic") };
}

function layerOverallScore(layers) {
  if (!layers) return null;
  return Math.round(
    (100 * (layers.why.score + layers.how.score + layers.mechanic.score)) / 3,
  );
}

/**
 * Единственный источник процента mastery для панели ноды и карты.
 * При coverage items прогресс считается из WHY/HOW/MECHANIC, а не из
 * устаревшего topic_mastery_score.
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
  const layers = items.length
    ? layersFromItems(items, normalizeBackendLayers(coverage), active)
    : null;
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

function subtopicHint(item) {
  const fromBackend = (
    item.status_hint ||
    item.statusHint ||
    ""
  ).trim();
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

function LayerBadge({ passed, label }) {
  const short = label === "MECHANIC" ? "MECH" : label;
  return React.createElement(
    "span",
    {
      className: `subtopic-layer-chip layer-${label.toLowerCase()}${
        passed ? " is-passed" : " is-pending"
      }`,
      title: passed ? `${label}: зачтено` : `${label}: не зачтено`,
    },
    React.createElement(
      "span",
      { className: "subtopic-layer-chip-mark", "aria-hidden": true },
      passed ? "✓" : "·",
    ),
    React.createElement("span", { className: "subtopic-layer-chip-text" }, short),
  );
}

function SubtopicsList({ items }) {
  if (!items?.length) return null;
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
        `${items.length}`,
      ),
    ),
    React.createElement(
      "ul",
      { className: "coverage-subtopic-list" },
      items.map((item) => {
        const why = itemFlag(item, "why_passed");
        const how = itemFlag(item, "how_passed");
        const mech = itemFlag(item, "mechanic_passed");
        const hint = subtopicHint(item);
        const st = item.state || "unchecked";
        return React.createElement(
          "li",
          {
            key: item.id || item.label,
            className: `coverage-subtopic coverage-subtopic-${st}`,
          },
          React.createElement(
            "div",
            { className: "coverage-subtopic-row" },
            React.createElement(
              "span",
              { className: "coverage-subtopic-label" },
              item.label || item.id,
            ),
            React.createElement(
              "span",
              { className: "coverage-subtopic-badges" },
              React.createElement(LayerBadge, { passed: why, label: "WHY" }),
              React.createElement(LayerBadge, { passed: how, label: "HOW" }),
              React.createElement(LayerBadge, {
                passed: mech,
                label: "MECHANIC",
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

function DepthLayersStrip({ layers, active }) {
  if (!layers) return null;
  return React.createElement(
    "div",
    { className: "depth-layers" },
    LAYER_META.map((meta) => {
      const row = layers[meta.id];
      let status = row.status;
      if (active === meta.key && status === "locked") status = "in_progress";
      const cls = LAYER_STATUS_CLS[status] || LAYER_STATUS_CLS.locked;
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

/** Future: facts_breakdown / item.facts when AtomicFactMatcher lands. */
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

function CoverageWidget({ coverage }) {
  const items = coverage?.items || [];
  if (!items.length) return null;

  const active = String(
    coverage?.active_layer || coverage?.activeLayer || "",
  )
    .trim()
    .toUpperCase();
  const backendLayers = normalizeBackendLayers(coverage);
  const layers = layersFromItems(items, backendLayers, active);
  const glossHint = (
    coverage?.gloss_hint ||
    coverage?.glossHint ||
    ""
  ).trim();
  const showGloss =
    Boolean(glossHint) ||
    (layers &&
      layers.why.status === "passed" &&
      layers.how.status === "passed" &&
      layers.mechanic.status !== "passed");

  return React.createElement(
    "div",
    { className: "coverage-widget coverage-widget-depth" },
    React.createElement(
      "div",
      { className: "coverage-widget-head" },
      React.createElement("span", { className: "coverage-label" }, "Depth"),
      React.createElement(
        "span",
        { className: "coverage-ratio muted" },
        active ? `фокус: ${active}` : "WHY → HOW → MECHANIC",
      ),
    ),
    React.createElement(DepthLayersStrip, { layers, active }),
    React.createElement(SubtopicsList, { items }),
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
  onModeSelect,
  disabled,
}) {
  const dash = masteryDashboard || {};
  const { coverage, items, active, layers, score } = resolveMasteryScore(
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
          title: layers ? "WHY ⅓ · HOW ⅓ · MECHANIC ⅓" : undefined,
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
    React.createElement(CoverageWidget, { coverage }),
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
    (dash.polish_zones || []).length > 0 &&
      React.createElement(
        "div",
        { className: "mastery-zone mastery-zone-warn" },
        React.createElement("h4", null, "Шлифовка"),
        React.createElement(
          "ul",
          null,
          dash.polish_zones.map((s, i) => React.createElement("li", { key: i }, s)),
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
