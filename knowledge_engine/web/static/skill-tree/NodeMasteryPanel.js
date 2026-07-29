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

export function NodeMasteryPanel({
  status,
  masteryDashboard,
  topicMasteryScore,
  onModeSelect,
  disabled,
}) {
  const dash = masteryDashboard || {};
  const score = dash.topic_mastery_score ?? topicMasteryScore ?? 0;
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
        onClick: () => onModeSelect?.(prefix),
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
        { className: "mastery-bar" },
        React.createElement("div", {
          className: "mastery-bar-fill",
          style: { width: `${Math.min(100, score)}%` },
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
