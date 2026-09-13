import React, { useState } from "react";
import {
  steeringGenerate,
  steeringApproveGate1,
  steeringApproveGate2,
  waitWorkJob,
} from "./api.js";
import { SteeringGatePanel } from "./SteeringGatePanel.js";

const SOURCE_POLICIES = [
  { value: "practical_only", label: "⚡ Практика — блоги и кейсы" },
  { value: "academic_only", label: "🔬 Академия — статьи и Consensus" },
  { value: "hybrid", label: "🧠 Полный — наука + практика" },
];

const STEPS = [
  { key: "input", label: "1. Цель" },
  { key: "gate1", label: "2. Gate 1" },
  { key: "gate2", label: "3. Gate 2" },
  { key: "generating", label: "4. Генерация" },
];

function StepIndicator({ current }) {
  return React.createElement(
    "div",
    { className: "steering-studio-steps" },
    STEPS.map((s) =>
      React.createElement(
        "span",
        {
          key: s.key,
          className:
            "steering-studio-step-chip" + (s.key === current ? " active" : ""),
        },
        s.label,
      ),
    ),
  );
}

/**
 * Standalone-режим `?mode=studio` — полноэкранный пошаговый мастер Штурвала,
 * без канваса/дерева ноды (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md,
 * аудит фронтенда). Использует ровно те же api.js-функции и тот же
 * SteeringGatePanel, что и виртуальная нода-гейт внутри RoadmapDashboard —
 * это НЕ дублирование логики Gate 1/2, а второй, независимый вход в тот же
 * backend-поток (SteeringGraphService).
 */
export function SteeringStudio() {
  const [step, setStep] = useState("input");
  const [goal, setGoal] = useState("");
  const [sourcePolicy, setSourcePolicy] = useState("practical_only");
  const [workJobId, setWorkJobId] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [digests, setDigests] = useState([]);
  const [genStatus, setGenStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function runAnalysis(e) {
    e?.preventDefault();
    const text = goal.trim();
    if (text.length < 8) return;
    setError("");
    setBusy(true);
    setGenStatus("TaxonomyService + Light Discovery…");
    try {
      const depth =
        sourcePolicy === "hybrid" || sourcePolicy === "academic_only"
          ? "Deep Mechanics"
          : "Standard";
      const res = await steeringGenerate({
        target_goal: text,
        user_level: "Intermediate/Advanced",
        depth_level: depth,
        source_policy: sourcePolicy,
        generation_mode: sourcePolicy === "academic_only" ? "consensus" : "fast",
        control_axis: "steering",
      });
      setWorkJobId(res.work_job_id);
      setCandidates(res.taxonomy_discovery?.candidate_articles || []);
      setStep("gate1");
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
      setGenStatus("");
    }
  }

  async function approveGate1(urls) {
    setBusy(true);
    setError("");
    try {
      const res = await steeringApproveGate1(workJobId, urls);
      setDigests(res.surface_digests?.digests || []);
      setStep("gate2");
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function approveGate2(urls) {
    setBusy(true);
    setError("");
    setStep("generating");
    setGenStatus("Штурвал: тяжёлый Map-Reduce по утверждённым источникам…");
    try {
      const res = await steeringApproveGate2(workJobId, urls);
      const job = await waitWorkJob(res.generation_job_id);
      const curriculumId = job?.result?.curriculum_id;
      if (!curriculumId) {
        throw new Error("Генерация завершилась без curriculum_id");
      }
      const nodes = job.result.nodes || [];
      const target = new URL(window.location.href);
      target.search = "";
      if (nodes.length === 1) {
        // Один узел из немногих утверждённых источников — открываем сразу
        // в фокусном режиме, без визуального шума карты на одну ноду.
        target.searchParams.set("mode", "topic");
        target.searchParams.set("id", nodes[0].node_id);
        target.searchParams.set("curriculum", curriculumId);
      } else {
        // Обычный многоузловой курс — открываем как classic RoadmapDashboard
        // (curriculum= уже поддержан её собственным useEffect).
        target.searchParams.set("curriculum", curriculumId);
      }
      window.location.href = target.toString();
    } catch (err) {
      setError(String(err.message || err));
      setStep("gate2");
      setBusy(false);
    }
  }

  return React.createElement(
    "div",
    { className: "steering-studio-container" },
    React.createElement(
      "header",
      { className: "steering-studio-header" },
      React.createElement("h1", null, "🎯 Штурвал Studio"),
      React.createElement(
        "p",
        { className: "muted" },
        "Пошаговая сборка курса из проверенных источников — без визуального графа.",
      ),
    ),
    React.createElement(StepIndicator, { current: step }),
    error && React.createElement("div", { className: "skill-error" }, error),
    step === "input" &&
      React.createElement(
        "form",
        {
          className: "steering-studio-step steering-studio-input",
          onSubmit: runAnalysis,
        },
        React.createElement("label", null, "Цель обучения"),
        React.createElement("input", {
          value: goal,
          onChange: (e) => setGoal(e.target.value),
          placeholder: "Чему вы хотите научиться?",
          minLength: 8,
          required: true,
          disabled: busy,
        }),
        React.createElement("label", null, "Источники"),
        React.createElement(
          "select",
          {
            value: sourcePolicy,
            onChange: (e) => setSourcePolicy(e.target.value),
            disabled: busy,
          },
          SOURCE_POLICIES.map((p) =>
            React.createElement("option", { key: p.value, value: p.value }, p.label),
          ),
        ),
        React.createElement(
          "button",
          {
            type: "submit",
            className: "skill-btn-primary",
            disabled: busy || goal.trim().length < 8,
          },
          busy ? genStatus || "…" : "Запустить анализ",
        ),
      ),
    (step === "gate1" || step === "gate2") &&
      React.createElement(
        "div",
        { className: "steering-studio-step" },
        React.createElement(SteeringGatePanel, {
          status: step === "gate1" ? "awaiting_gate_1" : "awaiting_gate_2",
          candidates,
          digests,
          busy,
          onApprove: step === "gate1" ? approveGate1 : approveGate2,
        }),
      ),
    step === "generating" &&
      React.createElement(
        "div",
        { className: "steering-studio-step steering-studio-generating" },
        React.createElement("p", null, genStatus || "Генерация курса…"),
      ),
  );
}
