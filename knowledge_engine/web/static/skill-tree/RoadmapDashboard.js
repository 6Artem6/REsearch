import React, { useCallback, useEffect, useState } from "react";
import { RoadmapCanvas } from "./RoadmapCanvas.js";
import { CurriculumInputBar } from "./CurriculumInputBar.js";
import { NodeDrawer } from "./NodeDrawer.js";
import { NodeTutorChat } from "./NodeTutorChat.js";
import { ColumnResizer } from "./ColumnResizer.js";
import {
  fetchRagStatus,
  createCurriculum,
  expandCurriculum,
  fetchCurriculaList,
  fetchWorkspace,
  setActiveCurriculum,
  rememberActiveCurriculumId,
  readActiveCurriculumId,
  hydrateSessionsFromServer,
  historyToMessages,
  mergeHistoryWithPendingUser,
  sortDialogMessages,
  tutorMessageFromApi,
  mergeNodeStatuses,
  nodeInit,
  nodeChat,
  nodeVerify,
  toNodeDataInput,
} from "./api.js";

export function RoadmapDashboard() {
  const [goal, setGoal] = useState("");
  const [sourcePolicy, setSourcePolicy] = useState("practical_only");
  const [ragStatus, setRagStatus] = useState(null);
  const [curriculum, setCurriculum] = useState(null);
  const [curriculaList, setCurriculaList] = useState([]);
  const [statuses, setStatuses] = useState({});
  const [selectedNode, setSelectedNode] = useState(null);
  const [sessions, setSessions] = useState({});
  const [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [genStatus, setGenStatus] = useState("");
  /** expand | create — какая кнопка запустила busy */
  const [genBusyAction, setGenBusyAction] = useState(null);
  /** Нода, для которой сейчас ждём init/chat/verify; null — нет активной генерации. */
  const [tutorBusyNodeId, setTutorBusyNodeId] = useState(null);
  const [error, setError] = useState("");
  /** Смена после expand — принудительный Dagre + fitView. */
  const [layoutEpoch, setLayoutEpoch] = useState(0);
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
      setSourcePolicy("practical_only");
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

  const refreshCurriculumGraph = useCallback(async (curriculumId) => {
    if (!curriculumId) return null;
    try {
      const ws = await fetchWorkspace(curriculumId);
      setCurriculum(ws.curriculum);
      setStatuses(
        mergeNodeStatuses(ws.curriculum, ws.statuses || {}),
      );
      return ws.curriculum;
    } catch (err) {
      setError(String(err.message || err));
      return null;
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
          fontSize: "14px",
          fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
          primaryTextColor: "#eceff4",
          lineColor: "#7eb8b8",
          primaryBorderColor: "#4ec9b0",
        },
        flowchart: {
          useMaxWidth: false,
          htmlLabels: true,
          padding: 28,
          nodeSpacing: 56,
          rankSpacing: 64,
          curve: "basis",
        },
        sequence: {
          useMaxWidth: false,
          wrap: true,
          width: 240,
          messageFontSize: 11,
          noteFontSize: 11,
          actorFontSize: 12,
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

  function clearCanvasForNewRoute() {
    setCurriculum(null);
    setSelectedNode(null);
    setSessions({});
    setStatuses({});
    const url = new URL(window.location.href);
    url.searchParams.delete("curriculum");
    window.history.replaceState(null, "", url.pathname + url.search);
    setSourcePolicy("practical_only");
  }

  async function runCreatePath(text) {
    setError("");
    setWorkspaceBusy(true);
    setGenBusyAction("create");
    let phaseTimer = null;
    const policyPhases = {
      hybrid: [
        "Model-First: Flash строит структуру DAG…",
        "Lite: классификация нод (BASE / DEEP)…",
        "DEEP: Exa + блоги (практика)…",
        "Summarizer → LanceDB → привязка источников…",
      ],
      academic_only: [
        "Semantic Scholar / arXiv / Consensus…",
        "Summarizer → LanceDB…",
        "Grounding DEEP-нод…",
      ],
      practical_only: [
        "Model-First → Risk → DEEP: Exa / SearXNG…",
        "Summarizer → LanceDB…",
        "Привязка источников к нодам…",
      ],
    };
    const phases = policyPhases[sourcePolicy] || policyPhases.hybrid;
    setGenStatus(phases[0]);
    let phaseIdx = 0;
    phaseTimer = setInterval(() => {
      if (phaseIdx >= phases.length - 1) return;
      phaseIdx += 1;
      setGenStatus(phases[phaseIdx]);
    }, 12000);
    try {
      const graph = await createCurriculum(text, sourcePolicy);
      setGoal(text);
      await loadWorkspace(graph.curriculum_id);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      if (phaseTimer) clearInterval(phaseTimer);
      setGenStatus("");
      setGenBusyAction(null);
      setWorkspaceBusy(false);
    }
  }

  async function runExpandBranch(text) {
    if (!curriculum?.curriculum_id) return;
    setError("");
    setWorkspaceBusy(true);
    setGenBusyAction("expand");
    setGenStatus("Достройка: Lite → сбор источников (SearXNG / SS)…");
    const expandPhases = [
      "Lite → вектор расширения…",
      "Сбор по вектору (SearXNG / SS / arXiv)…",
      "Summarizer → LanceDB…",
      "Flash достраивает ветку…",
    ];
    let phaseIdx = 0;
    const phaseTimer = setInterval(() => {
      if (phaseIdx >= expandPhases.length - 1) return;
      phaseIdx += 1;
      setGenStatus(expandPhases[phaseIdx]);
    }, 10000);
    try {
      const graph = await expandCurriculum(
        curriculum.curriculum_id,
        text,
        sourcePolicy,
      );
      setGoal("");
      setLayoutEpoch((n) => n + 1);
      await loadWorkspace(graph.curriculum_id);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      clearInterval(phaseTimer);
      setGenStatus("");
      setGenBusyAction(null);
      setWorkspaceBusy(false);
    }
  }

  async function runCreateNewWhileLoaded(text) {
    clearCanvasForNewRoute();
    await runCreatePath(text);
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
        ? sortDialogMessages(
            mergeHistoryWithPendingUser(
              historyToMessages(res.history),
              userMsg,
            ),
          )
        : (() => {
            const next = [...old.messages];
            if (userMsg) {
              next.push({
                role: "user",
                content: userMsg,
                msg_id: `pending-${Date.now()}`,
              });
            }
            if (res.tutor_message) next.push(tutorMessageFromApi(res));
            return sortDialogMessages(next);
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
          sourceRegistry: Array.isArray(res.source_registry)
            ? res.source_registry
            : old.sourceRegistry || [],
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
      const freshGraph = await refreshCurriculumGraph(curriculum.curriculum_id);
      if (freshGraph) {
        const freshNode = freshGraph.nodes.find((n) => n.node_id === sid);
        if (freshNode) setSelectedNode(freshNode);
      }
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
            msgs.push({
              role: "user",
              content: u,
              msg_id: `pending-${Date.now()}`,
            });
            return sortDialogMessages(msgs);
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
      React.createElement(CurriculumInputBar, {
        goal,
        onGoalChange: setGoal,
        sourcePolicy,
        onSourcePolicyChange: setSourcePolicy,
        activeCurriculumId: activeId,
        workspaceBusy,
        genStatus,
        busyAction: genBusyAction,
        onCreatePath: runCreatePath,
        onExpandBranch: runExpandBranch,
        onCreateNew: runCreateNewWhileLoaded,
      }),
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
            layoutEpoch,
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
