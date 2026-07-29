import React, { useCallback, useEffect, useState } from "react";
import { RoadmapCanvas } from "./RoadmapCanvas.js";
import { NodeDrawer } from "./NodeDrawer.js";
import { NodeTutorChat } from "./NodeTutorChat.js";
import { ColumnResizer } from "./ColumnResizer.js";
import {
  fetchRagStatus,
  generateCurriculum,
  fetchCurriculaList,
  fetchWorkspace,
  setActiveCurriculum,
  rememberActiveCurriculumId,
  readActiveCurriculumId,
  hydrateSessionsFromServer,
  historyToMessages,
  mergeHistoryWithPendingUser,
  tutorMessageFromApi,
  mergeNodeStatuses,
  nodeInit,
  nodeChat,
  nodeVerify,
  toNodeDataInput,
} from "./api.js";

export function RoadmapDashboard() {
  const [goal, setGoal] = useState("");
  const [mode, setMode] = useState("fast");
  const [ragStatus, setRagStatus] = useState(null);
  const [curriculum, setCurriculum] = useState(null);
  const [curriculaList, setCurriculaList] = useState([]);
  const [statuses, setStatuses] = useState({});
  const [selectedNode, setSelectedNode] = useState(null);
  const [sessions, setSessions] = useState({});
  const [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [genStatus, setGenStatus] = useState("");
  /** Нода, для которой сейчас ждём init/chat/verify; null — нет активной генерации. */
  const [tutorBusyNodeId, setTutorBusyNodeId] = useState(null);
  const [error, setError] = useState("");
  const [leftColWidth, setLeftColWidth] = useState(() => {
    const n = Number(localStorage.getItem("skillTreeColLeft"));
    return n >= 240 && n <= 720 ? n : 360;
  });
  const [rightColWidth, setRightColWidth] = useState(() => {
    const n = Number(localStorage.getItem("skillTreeColRight"));
    return n >= 280 && n <= 960 ? n : 420;
  });
  const leftColRef = React.useRef(leftColWidth);
  const rightColRef = React.useRef(rightColWidth);
  leftColRef.current = leftColWidth;
  rightColRef.current = rightColWidth;

  function persistColWidths() {
    localStorage.setItem("skillTreeColLeft", String(leftColRef.current));
    localStorage.setItem("skillTreeColRight", String(rightColRef.current));
  }

  const loadWorkspace = useCallback(async (curriculumId) => {
    if (!curriculumId) return;
    setError("");
    setWorkspaceBusy(true);
    try {
      const ws = await fetchWorkspace(curriculumId);
      setCurriculum(ws.curriculum);
      setGoal(ws.meta?.target_goal || "");
      setStatuses(
        mergeNodeStatuses(ws.curriculum, ws.statuses || {}),
      );
      setSessions(hydrateSessionsFromServer(ws.sessions));
      setSelectedNode(null);
      await setActiveCurriculum(curriculumId);
      rememberActiveCurriculumId(curriculumId);
      const url = new URL(window.location.href);
      url.searchParams.set("curriculum", curriculumId);
      window.history.replaceState(null, "", url.pathname + url.search);
      const list = await fetchCurriculaList();
      setCurriculaList(list.curricula || []);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setWorkspaceBusy(false);
    }
  }, []);

  useEffect(() => {
    fetchRagStatus()
      .then(setRagStatus)
      .catch(() =>
        setRagStatus({
          connected: false,
          label: "RAG: статус недоступен",
        }),
      );
    if (window.mermaid) {
      window.mermaid.initialize({
        startOnLoad: false,
        theme: "dark",
        securityLevel: "loose",
        themeVariables: {
          fontSize: "11px",
        },
        flowchart: { useMaxWidth: false, htmlLabels: true },
        sequence: {
          useMaxWidth: false,
          wrap: true,
          width: 240,
          messageFontSize: 10,
          noteFontSize: 10,
          actorFontSize: 11,
          messageMargin: 48,
          boxMargin: 10,
          mirrorActors: false,
        },
      });
    }

    (async () => {
      try {
        const list = await fetchCurriculaList();
        setCurriculaList(list.curricula || []);
        const params = new URLSearchParams(window.location.search);
        const fromUrl = params.get("curriculum");
        const active =
          fromUrl ||
          list.active_curriculum_id ||
          readActiveCurriculumId();
        if (active) await loadWorkspace(active);
      } catch (err) {
        setError(String(err.message || err));
      }
    })();
  }, [loadWorkspace]);

  async function onGenerate(e) {
    e.preventDefault();
    const text = goal.trim();
    if (text.length < 8) return;
    setError("");
    setWorkspaceBusy(true);
    let phaseTimer = null;
    const consensusPhases = [
      "Сбор научных статей в Consensus (Playwright)…",
      "Lite-валидация и Summarizer → LanceDB…",
      "Flash строит граф вокруг выдержек…",
    ];
    if (mode === "consensus") {
      let phaseIdx = 0;
      setGenStatus(consensusPhases[0]);
      phaseTimer = setInterval(() => {
        phaseIdx = (phaseIdx + 1) % consensusPhases.length;
        setGenStatus(consensusPhases[phaseIdx]);
      }, 8000);
    } else {
      setGenStatus("SearXNG / whitelist и Flash…");
    }
    try {
      const graph = await generateCurriculum(text, mode);
      await loadWorkspace(graph.curriculum_id);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      if (phaseTimer) clearInterval(phaseTimer);
      setGenStatus("");
      setWorkspaceBusy(false);
    }
  }

  const applyNodeResponse = useCallback((nodeId, res, userMsg) => {
    if (res.error) {
      setError(res.error);
      return;
    }
    setStatuses((prev) => ({ ...prev, [nodeId]: res.node_status }));
    setSessions((prev) => {
      const old = prev[nodeId] || { messages: [] };
      const hasHistory =
        Array.isArray(res.history) && res.history.length > 0;
      const messages = hasHistory
        ? mergeHistoryWithPendingUser(
            historyToMessages(res.history),
            userMsg,
          )
        : (() => {
            const next = [...old.messages];
            if (userMsg) next.push({ role: "user", content: userMsg });
            if (res.tutor_message) next.push(tutorMessageFromApi(res));
            return next;
          })();
      return {
        ...prev,
        [nodeId]: {
          initialized: true,
          content: res.content,
          messages,
          ragLabels: res.rag_fact_labels || old.ragLabels || [],
          masteryDashboard: res.mastery_dashboard || old.masteryDashboard,
          topicMasteryScore:
            res.topic_mastery_score ?? old.topicMasteryScore ?? 0,
          learningPhase: res.learning_phase || old.learningPhase,
          learningMode: res.learning_mode || old.learningMode,
          sourceRegistry: res.source_registry || old.sourceRegistry || [],
        },
      };
    });
  }, []);

  async function openNode(node) {
    if (!curriculum) return;
    const sid = node.node_id;
    const initialized = Boolean(sessions[sid]?.initialized);

    if (tutorBusyNodeId !== null && !initialized) return;

    setSelectedNode(node);
    setError("");
    if (initialized) return;

    setTutorBusyNodeId(sid);
    try {
      const res = await nodeInit(
        curriculum.curriculum_id,
        toNodeDataInput(node),
      );
      applyNodeResponse(sid, res);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setTutorBusyNodeId(null);
    }
  }

  async function sendTutorMessage(text) {
    if (!curriculum || !selectedNode || tutorBusyNodeId !== null) return;
    const msg = (text || "").trim();
    if (!msg) return;
    const nid = selectedNode.node_id;
    setTutorBusyNodeId(nid);
    onTutorPendingUser(msg);
    try {
      const res = await nodeChat(
        curriculum.curriculum_id,
        toNodeDataInput(selectedNode),
        msg,
      );
      applyNodeResponse(nid, res, msg);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setTutorBusyNodeId(null);
    }
  }

  async function runVerify() {
    if (!curriculum || !selectedNode || tutorBusyNodeId !== null) return;
    const nid = selectedNode.node_id;
    setTutorBusyNodeId(nid);
    try {
      const res = await nodeVerify(
        curriculum.curriculum_id,
        toNodeDataInput(selectedNode),
        "Готов ответить на практические вопросы по этой теме.",
      );
      onVerifyResponse(res);
    } catch (err) {
      onVerifyResponse({ error: String(err.message || err) });
    } finally {
      setTutorBusyNodeId(null);
    }
  }

  function onTutorPendingUser(userMsg) {
    if (!selectedNode) return;
    const nid = selectedNode.node_id;
    setSessions((prev) => {
      const old = prev[nid] || { messages: [], initialized: true };
      const u = (userMsg || "").trim();
      if (!u) return prev;
      if (
        (old.messages || []).some(
          (m) => m.role === "user" && (m.content || "").trim() === u,
        )
      ) {
        return prev;
      }
      return {
        ...prev,
        [nid]: {
          ...old,
          messages: (() => {
            const msgs = [...(old.messages || [])];
            if (msgs.length && msgs[msgs.length - 1].role === "tutor") {
              msgs.splice(msgs.length - 1, 0, { role: "user", content: u });
            } else {
              msgs.push({ role: "user", content: u });
            }
            return msgs;
          })(),
        },
      };
    });
  }

  function onVerifyResponse(res) {
    if (!selectedNode) return;
    applyNodeResponse(selectedNode.node_id, res, null);
  }

  async function onModeSelect(text) {
    await sendTutorMessage(text);
  }

  const session = selectedNode ? sessions[selectedNode.node_id] : null;
  const activeId = curriculum?.curriculum_id || "";
  const tutorBusy = tutorBusyNodeId !== null;
  const composeLocked = tutorBusy;
  const nodeGenerating =
    selectedNode && tutorBusyNodeId === selectedNode.node_id;

  function formatRouteLabel(c) {
    const title = (c.title || c.target_goal || c.curriculum_id || "").trim();
    const nodes = c.total_nodes ? ` · ${c.total_nodes} нод` : "";
    const suffix = c.has_graph === false ? " (без графа)" : "";
    return `${title}${nodes}${suffix}`;
  }

  return React.createElement(
    "div",
    { className: "skill-dashboard" },
    React.createElement(
      "header",
      { className: "skill-header" },
      React.createElement(
        "div",
        { className: "skill-header-top" },
        React.createElement(
          "div",
          null,
          React.createElement("h1", null, "AI Skill Tree & Tutor"),
          React.createElement(
            "p",
            { className: "muted" },
            "Маршруты: knowledge_engine/.runs/skill_tree_curricula.json",
          ),
          React.createElement(
            "a",
            { href: "/app", className: "nav-link-skill" },
            "← Исследовательский анализ",
          ),
        ),
        ragStatus &&
          React.createElement(
            "span",
            {
              className: `rag-pill${ragStatus.connected ? "" : " off"}`,
            },
            ragStatus.label,
          ),
      ),
      React.createElement(
        "div",
        { className: "skill-saved-section" },
        React.createElement(
          "p",
          { className: "skill-saved-title" },
          `Сохранённые маршруты (${curriculaList.length})`,
        ),
        curriculaList.length === 0
          ? React.createElement(
              "p",
              { className: "skill-saved-empty" },
              "Пока нет маршрутов. Создайте путь ниже — он сохранится автоматически.",
            )
          : React.createElement(
              "div",
              { className: "skill-route-list" },
              curriculaList.map((c) =>
                React.createElement(
                  "button",
                  {
                    key: c.curriculum_id,
                    type: "button",
                    className: [
                      "skill-route-btn",
                      c.curriculum_id === activeId ? "active" : "",
                      c.has_graph === false ? "missing-graph" : "",
                    ]
                      .filter(Boolean)
                      .join(" "),
                    onClick: () => {
                      if (c.has_graph === false) {
                        setError(
                          "Граф этого маршрута не сохранён. Создайте путь заново с той же темой.",
                        );
                        return;
                      }
                      loadWorkspace(c.curriculum_id);
                    },
                  },
                  formatRouteLabel(c),
                ),
              ),
            ),
      ),
      React.createElement(
        "div",
        { className: "skill-header-actions" },
        React.createElement(
          "form",
          { className: "skill-goal-form", onSubmit: onGenerate },
          React.createElement("input", {
            value: goal,
            onChange: (e) => setGoal(e.target.value),
            placeholder: "Чему вы хотите научиться?",
            required: true,
          }),
          React.createElement(
            "select",
            {
              className: "skill-mode-select",
              value: mode,
              onChange: (e) => setMode(e.target.value),
              "aria-label": "Режим генерации",
            },
            React.createElement(
              "option",
              { value: "fast" },
              "Fast — быстрый граф",
            ),
            React.createElement(
              "option",
              { value: "consensus" },
              "Consensus — глубокий анализ (v0.8)",
            ),
          ),
          workspaceBusy &&
            genStatus &&
            React.createElement(
              "p",
              { className: "muted skill-gen-status", role: "status" },
              genStatus,
            ),
          React.createElement(
            "button",
            { type: "submit", disabled: workspaceBusy },
            workspaceBusy
              ? mode === "consensus"
                ? "Глубокая генерация…"
                : "Генерация…"
              : "Создать путь",
          ),
        ),
      ),
    ),
    error && React.createElement("div", { className: "skill-error" }, error),
    React.createElement(
      "div",
      {
        className: "skill-split",
        style: {
          gridTemplateColumns: `${leftColWidth}px 6px minmax(180px, 1fr) 6px ${rightColWidth}px`,
        },
      },
      React.createElement(
        "aside",
        { className: "skill-chat-column" },
        curriculum && selectedNode
          ? React.createElement(NodeTutorChat, {
              session,
              ragLabels: session?.ragLabels,
              onSend: sendTutorMessage,
              disabled: composeLocked,
              generating: nodeGenerating,
              curriculumId: curriculum.curriculum_id,
              nodeData: toNodeDataInput(selectedNode),
            })
          : React.createElement(
              "div",
              { className: "tutor-panel skill-chat-placeholder" },
              React.createElement("h3", null, "Чат с тьютором"),
              React.createElement(
                "p",
                { className: "muted" },
                curriculum
                  ? "Выберите ноду на карте — диалог откроется здесь (как в Cursor)."
                  : "Создайте или выберите маршрут, затем откройте ноду на графе.",
              ),
            ),
      ),
      React.createElement(ColumnResizer, {
        onDragDelta: (dx) => {
          setLeftColWidth((w) => {
            const next = Math.min(720, Math.max(240, w + dx));
            leftColRef.current = next;
            return next;
          });
        },
        onDragEnd: persistColWidths,
      }),
      curriculum
        ? React.createElement(RoadmapCanvas, {
            curriculum,
            statuses,
            selectedNodeId: selectedNode?.node_id,
            onNodeClick: openNode,
            tutorBusyNodeId,
            sessions,
          })
        : React.createElement(
            "div",
            { className: "skill-canvas-wrap muted", style: { padding: "2rem" } },
            "Введите цель или выберите сохранённый маршрут.",
          ),
      React.createElement(ColumnResizer, {
        onDragDelta: (dx) => {
          setRightColWidth((w) => {
            const next = Math.min(960, Math.max(280, w - dx));
            rightColRef.current = next;
            return next;
          });
        },
        onDragEnd: persistColWidths,
      }),
      curriculum
        ? React.createElement(NodeDrawer, {
            curriculum,
            selectedNode,
            session,
            statuses,
            onSelectPrereq: openNode,
            onModeSelect,
            onVerify: runVerify,
            composeLocked,
            nodeGenerating,
            sessions,
          })
        : React.createElement("aside", { className: "node-drawer empty" }),
    ),
  );
}
