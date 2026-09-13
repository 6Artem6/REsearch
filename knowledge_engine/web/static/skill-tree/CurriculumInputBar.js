import React from "react";

/**
 * Верхняя панель: создание нового графа vs достройка текущего.
 */
export function CurriculumInputBar({
  goal,
  onGoalChange,
  sourcePolicy,
  onSourcePolicyChange,
  controlAxis,
  onControlAxisChange,
  steeringMode,
  onSteeringModeChange,
  activeCurriculumId,
  workspaceBusy,
  genStatus,
  longWaitNotice,
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
    // БАГФИКС: "Достроить ветку" (onExpandBranch) не читает controlAxis — у
    // Штурвала нет expand-аналога (нет /curriculum/steering/expand). Раньше
    // при hasGraph===true (localStorage почти всегда хранит последний
    // активный курс, см. активность вокруг hasGraph выше) обычный Enter/сабмит
    // формы ВСЕГДА уходил в onExpandBranch, полностью игнорируя выбор
    // Штурвала — отсюда "при Штурвале идёт прежний полный пайплайн".
    if (hasGraph && controlAxis === "steering") {
      onCreateNew(text);
    } else if (hasGraph) {
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
      React.createElement(
        "select",
        {
          className: "skill-mode-select",
          value: controlAxis || "autopilot",
          onChange: (e) => onControlAxisChange(e.target.value),
          "aria-label": "Ось управления",
          // БАГФИКС: раньше здесь стояло `|| hasGraph`, из-за чего селектор
          // был заблокирован почти всегда — RoadmapDashboard на маунте сам
          // подгружает последний активный курс из localStorage
          // (readActiveCurriculumId), поэтому hasGraph чаще всего true уже
          // на старте, а не только "после создания". control_axis нужен не
          // только для формы при hasGraph===false, но и для кнопки
          // "Создать новый" (onCreateNew -> runCreateNewWhileLoaded ->
          // runCreatePath), которая доступна ИМЕННО при hasGraph===true —
          // блокировка отсюда убивала Steering именно в этом, самом частом
          // сценарии. sourcePolicy рядом (см. выше) блокируется только по
          // workspaceBusy — делаем control_axis симметрично.
          disabled: workspaceBusy,
        },
        React.createElement("option", { value: "autopilot" }, "🤖 Автопилот"),
        React.createElement(
          "option",
          { value: "steering" },
          "🎯 Штурвал (HITL)",
        ),
      ),
      // Разделение Штурвала на Mode 1 (per_node) / Mode 2 (standalone_digest)
      // — см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Разделение на Mode
      // 1/Mode 2". Селектор виден только при controlAxis==="steering" —
      // Автопилоту он не нужен, у него нет под-режимов.
      controlAxis === "steering" &&
        React.createElement(
          "select",
          {
            className: "skill-mode-select",
            value: steeringMode || "per_node",
            onChange: (e) => onSteeringModeChange(e.target.value),
            "aria-label": "Режим Штурвала",
            disabled: workspaceBusy,
          },
          React.createElement(
            "option",
            { value: "per_node" },
            "🗺️ По нодам — граф + точечное заземление",
          ),
          React.createElement(
            "option",
            { value: "standalone_digest" },
            "📄 Обзор темы — без графа, один документ",
          ),
        ),
      genStatus &&
        React.createElement(
          "p",
          { className: "muted skill-gen-status", role: "status" },
          genStatus,
        ),
      // Мягкое доп. уведомление после долгого ожидания (см. prompt.log,
      // "UX-уведомление о длительной обработке") — НЕ заменяет genStatus/
      // индикатор прогресса, просто дополняет его отдельной строкой.
      longWaitNotice &&
        React.createElement(
          "p",
          { className: "muted skill-gen-status skill-gen-long-wait", role: "status" },
          longWaitNotice,
        ),
      hasGraph
        ? React.createElement(
            "div",
            { className: "skill-btn-group" },
            // БАГФИКС: скрываем "Достроить ветку" при Штурвале — у него нет
            // expand-эквивалента (см. onFormSubmit выше), кнопка иначе молча
            // игнорировала бы выбор оси и запускала старый Autopilot-пайплайн.
            controlAxis !== "steering" &&
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
