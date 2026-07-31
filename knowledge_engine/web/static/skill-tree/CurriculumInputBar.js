import React from "react";

/**
 * Верхняя панель: создание нового графа vs достройка текущего.
 */
export function CurriculumInputBar({
  goal,
  onGoalChange,
  sourcePolicy,
  onSourcePolicyChange,
  activeCurriculumId,
  workspaceBusy,
  genStatus,
  busyAction,
  onCreatePath,
  onExpandBranch,
  onCreateNew,
}) {
  const hasGraph = Boolean(activeCurriculumId);
  const expandBusy = workspaceBusy && busyAction === "expand";
  const createBusy = workspaceBusy && busyAction === "create";

  function onFormSubmit(e) {
    e.preventDefault();
    const text = (goal || "").trim();
    if (text.length < 8) return;
    if (hasGraph) {
      onExpandBranch(text);
    } else {
      onCreatePath(text);
    }
  }

  return React.createElement(
    "div",
    { className: "skill-header-actions" },
    React.createElement(
      "form",
      { className: "skill-goal-form", onSubmit: onFormSubmit },
      React.createElement("input", {
        value: goal,
        onChange: (e) => onGoalChange(e.target.value),
        placeholder: hasGraph
          ? "Впишите вектор для достройки (или введите новую тему и нажмите «Создать новый»)…"
          : "Чему вы хотите научиться?",
        required: true,
        minLength: 8,
        disabled: workspaceBusy,
      }),
      React.createElement(
        "select",
        {
          className: "skill-mode-select",
          value: sourcePolicy,
          onChange: (e) => onSourcePolicyChange(e.target.value),
          "aria-label": "Режим сбора источников",
          disabled: workspaceBusy,
        },
        React.createElement(
          "option",
          { value: "practical_only" },
          "⚡ Практика — блоги и кейсы",
        ),
        React.createElement(
          "option",
          { value: "academic_only" },
          "🔬 Академия — статьи и Consensus",
        ),
        React.createElement(
          "option",
          { value: "hybrid" },
          "🧠 Полный — наука + практика",
        ),
      ),
      genStatus &&
        React.createElement(
          "p",
          { className: "muted skill-gen-status", role: "status" },
          genStatus,
        ),
      hasGraph
        ? React.createElement(
            "div",
            { className: "skill-btn-group" },
            React.createElement(
              "button",
              {
                type: "button",
                className: "skill-btn-primary",
                disabled: workspaceBusy,
                onClick: () => {
                  const text = (goal || "").trim();
                  if (text.length >= 8) onExpandBranch(text);
                },
              },
              expandBusy ? "Достройка ветки…" : "+ Достроить ветку",
            ),
            React.createElement(
              "button",
              {
                type: "button",
                className: "skill-btn-secondary",
                disabled: workspaceBusy,
                onClick: () => {
                  const text = (goal || "").trim();
                  if (text.length >= 8) onCreateNew(text);
                },
              },
              createBusy ? "Сборка нового пути…" : "Создать новый",
            ),
          )
        : React.createElement(
            "button",
            {
              type: "submit",
              className: "skill-btn-primary",
              disabled: workspaceBusy,
            },
            createBusy || (workspaceBusy && !busyAction)
              ? sourcePolicy === "hybrid"
                ? "Полный сбор…"
                : sourcePolicy === "academic_only"
                  ? "Академический сбор…"
                  : "Сбор практики…"
              : "Создать путь",
          ),
    ),
  );
}
