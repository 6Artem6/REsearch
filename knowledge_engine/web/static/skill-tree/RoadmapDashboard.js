import { ensureMermaidInitialized } from "./mermaidRuntime.js";
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
  rememberCurriculumControlAxis,
  readCurriculumControlAxis,
  hydrateSessionsFromServer,
  buildMessagesAfterChatComplete,
  sortDialogMessages,
  userMessageMatches,
  isPendingMsgId,
  mergeNodeStatuses,
  nodeInit,
  nodeInitStream,
  nodeRestart,
  nodeChat,
  nodeChatStream,
  nodeVerify,
  fetchNodeSourceRegistry,
  toNodeDataInput,
  steeringGenerate,
  steeringApproveGate1,
  steeringApproveGate2,
  waitWorkJob,
  fetchWorkJob,
  nodeGroundingDiscover,
  nodeGroundingDigest,
  nodeGroundingFinalize,
  steeringGetStatus,
  ensureSteeringSourcesIngested,
} from "./api.js";
import { MATERIAL_VIEW_LS } from "./materialAssets.js";

function replaceSkillTreeSearchParams(patch) {
  const url = new URL(window.location.href);
  for (const [key, value] of Object.entries(patch)) {
    const v = String(value || "").trim();
    if (v) url.searchParams.set(key, v);
    else url.searchParams.delete(key);
  }
  window.history.replaceState(null, "", url.pathname + url.search);
}

const NODE_JOB_POLL_INTERVAL_MS = 30000;

/** "Статус worker: running · прошло 2 мин 30 с" (+ текущая стадия, если есть). */
function formatJobPollNotice({ status, elapsedSec }, stage = "") {
  const mins = Math.floor(elapsedSec / 60);
  const secs = elapsedSec % 60;
  const elapsed = mins > 0 ? `${mins} мин ${secs} с` : `${secs} с`;
  const head = stage ? `${stage} · ` : "";
  return `${head}Статус worker: ${status} · прошло ${elapsed}`;
}

export function RoadmapDashboard() {
  const [goal, setGoal] = useState("");
  const [sourcePolicy, setSourcePolicy] = useState("practical_only");
  const [controlAxis, setControlAxis] = useState("autopilot");
  const [steeringMode, setSteeringMode] = useState("per_node");
  const [ragStatus, setRagStatus] = useState(null);
  const [curriculum, setCurriculum] = useState(null);
  const [curriculaList, setCurriculaList] = useState([]);
  const [statuses, setStatuses] = useState({});
  const [selectedNode, setSelectedNode] = useState(null);
  const [sessions, setSessions] = useState({});
  const [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [genStatus, setGenStatus] = useState("");
  // RU ("Убрать жёсткий таймаут ожидания..."): waitWorkJob
  // больше не падает по истечении 10 минут — ждёт сколько нужно. Этот
  // текст — мягкое дополнительное уведомление (не блокирует genStatus/
  // индикатор), показывается один раз, когда суммарное ожидание одной
  // операции превысило 5 минут (см. onLongWait в api.js::waitWorkJob).
  const [longWaitNotice, setLongWaitNotice] = useState("");
  /** Живой статус фонового job (опрос бэка раз в 30 с) при генерации графа. */
  const [genPollNotice, setGenPollNotice] = useState("");
  const LONG_WAIT_MESSAGE =
    "Запрос идёт чуть дольше обычного. Пожалуйста, подождите, идёт глубокий аналитический сбор…";
  /** expand | create — какая кнопка запустила busy */
  const [genBusyAction, setGenBusyAction] = useState(null);
  /** Нода, для которой сейчас ждём init/chat/verify; null — нет активной генерации. */
  const [tutorBusyNodeId, setTutorBusyNodeId] = useState(null);
  /** Текст последнего FSM stage-события (см. schemas/fsm.py) — живой статус
   * в баннере NodeTutorChat вместо статичного "Генерация ответа…". */
  const [tutorStageMessage, setTutorStageMessage] = useState("");
  const [selectedMaterialId, setSelectedMaterialId] = useState(null);
  const [materialViewMode, setMaterialViewMode] = useState(() => {
    const v = localStorage.getItem(MATERIAL_VIEW_LS);
    return v === "carousel" ? "carousel" : "list";
  });
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

  function maxResizableColumnWidth(oppositeWidth) {
    const minCanvasWidth = 180;
    const dividerWidth = 12;
    return Math.max(
      0,
      window.innerWidth - oppositeWidth - minCanvasWidth - dividerWidth,
    );
  }

  useEffect(() => {
    if (genBusyAction === null) setGenPollNotice("");
  }, [genBusyAction]);

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
      setSelectedMaterialId(null);
      await setActiveCurriculum(curriculumId);
      rememberActiveCurriculumId(curriculumId);
      // Режим, с которым курс был создан, становится дефолтом переключателя;
      // пользователь может сменить его для следующей открываемой ноды.
      const savedAxis = readCurriculumControlAxis(curriculumId);
      if (savedAxis) setControlAxis(savedAxis);
      setSourcePolicy("practical_only");
      replaceSkillTreeSearchParams({ curriculum: curriculumId });
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

  /** Восстановление Штурвал-сессии после обновления страницы —

   * curriculum_id="steering-<jobId>" никогда не сохраняется в обычный
   * skill_tree_store (см. api/routes/steering.py::get_steering_status),
   * поэтому обычный loadWorkspace() для него всегда отвечал 404 "Маршрут
   * не найден". Строит ту же виртуальную ноду-гейт/session, что
   * handleGenerate/handleGateApprove — с точки зрения остального UI это
   * неотличимо от "только что сгенерированной" Штурвал-сессии. */
  const resumeSteeringSession = useCallback(async (jobId) => {
    setError("");
    setWorkspaceBusy(true);
    try {
      const res = await steeringGetStatus(jobId);
      if (res.status === "awaiting_gate_1" || res.status === "awaiting_gate_2") {
        const gateNode = {
          node_id: "steering-gate-node",
          title:
            res.status === "awaiting_gate_1"
              ? "🎯 Согласование источников (Gate 1)"
              : "📦 Обзор выжимок (Gate 2)",
          layer: "foundation",
          category: "Штурвал",
          prerequisites: [],
        };
        const virtualCurriculumId = `steering-${jobId}`;
        setGoal(res.target_goal || "");
        setCurriculum({ curriculum_id: virtualCurriculumId, nodes: [gateNode] });
        setStatuses({ [gateNode.node_id]: "in_progress" });
        setSelectedNode(gateNode);
        setSelectedMaterialId(null);
        setSessions({
          [gateNode.node_id]: {
            initialized: true,
            messages: [],
            steeringWorkJobId: jobId,
            steeringStatus: res.status,
            steeringCandidates: res.taxonomy_discovery?.candidate_articles || [],
            steeringDigests: res.surface_digests?.digests || [],
            steeringApprovedCount: (res.approved_gate1_urls || []).length,
          },
        });
        replaceSkillTreeSearchParams({
          curriculum: virtualCurriculumId,
          node: gateNode.node_id,
          material: "",
        });
        return;
      }
      if (res.status === "completed") {
        const genJobId = res.result?.generation_job_id;
        if (!genJobId) {
          setError(
            "Сессия Штурвала завершена, но задача генерации курса не найдена.",
          );
          return;
        }
        setGenBusyAction("create");
        setGenStatus("Штурвал: ожидаем завершения Map-Reduce…");
        const job = await waitWorkJob(genJobId, {
          onLongWait: () => setLongWaitNotice(LONG_WAIT_MESSAGE),
          onPoll: (info) => setGenPollNotice(formatJobPollNotice(info)),
        });
        if (job?.result?.curriculum_id) {
          await loadWorkspace(job.result.curriculum_id);
        } else if (job?.status === "failed") {
          setError(`Генерация курса Штурвала завершилась ошибкой: ${job.error || ""}`);
        } else {
          setError(
            "Генерация курса Штурвала ещё не завершена — попробуйте обновить страницу позже.",
          );
        }
        return;
      }
      setError(
        `Сессия Штурвала не может быть восстановлена (status=${res.status}).`,
      );
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setWorkspaceBusy(false);
      setGenBusyAction(null);
      setGenStatus("");
      setLongWaitNotice("");
    }
  }, [loadWorkspace]);

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
      ensureMermaidInitialized();
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
        if (active && active.startsWith("steering-")) {
          await resumeSteeringSession(active.slice("steering-".length));
        } else if (active) {
          await loadWorkspace(active);
        }
      } catch (err) {
        setError(String(err.message || err));
      }
    })();
  }, [loadWorkspace, resumeSteeringSession]);

  function clearCanvasForNewRoute() {
    setCurriculum(null);
    setSelectedNode(null);
    setSessions({});
    setStatuses({});
    replaceSkillTreeSearchParams({
      curriculum: "",
      node: "",
      material: "",
    });
    setSourcePolicy("practical_only");
  }

  async function runCreatePath(text) {
    if (controlAxis === "steering") {
      await runSteeringCreatePath(text);
      return;
    }
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
      const graph = await createCurriculum(text, sourcePolicy, {
        onLongWait: () => setLongWaitNotice(LONG_WAIT_MESSAGE),
          onPoll: (info) => setGenPollNotice(formatJobPollNotice(info)),
      });
      setGoal(text);
      rememberCurriculumControlAxis(graph.curriculum_id, "autopilot");
      await loadWorkspace(graph.curriculum_id);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      if (phaseTimer) clearInterval(phaseTimer);
      setGenStatus("");
      setLongWaitNotice("");
      setGenBusyAction(null);
      setWorkspaceBusy(false);
    }
  }

  /** Штурвал (control_axis="steering") — разводится по steeringMode (см.

   * docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Разделение на Mode 1/Mode 2"):
   * per_node — граф строится сразу (Model-First, без предв. поиска), никакого
   * Gate 1/2 на уровне курса нет, точечное заземление нод — при их открытии
   * (Node Grounding Gate); standalone_digest — прежний путь: TaxonomyService
   * + Light Discovery → Gate 1, показанный как виртуальная нода-интерцептор
   * на канвасе, итог — не граф, а единый документ (Gate 2 → см.
   * handleGateApprove). */
  async function runSteeringCreatePath(text) {
    if (steeringMode === "per_node") {
      await runSteeringCreatePerNode(text);
      return;
    }
    await runSteeringCreateStandaloneDigest(text);
  }

  /** Mode 1: тонкий алиас поверх того же WorkJobKind.CURRICULUM_GENERATE,

   * что обычный Autopilot (см. api/routes/steering.py::
   * _post_steering_generate_per_node) — граф без гейтов, ждём тот же job,
   * что и runCreatePath. */
  async function runSteeringCreatePerNode(text) {
    setError("");
    setWorkspaceBusy(true);
    setGenBusyAction("create");
    setGenStatus("Штурвал (по нодам): Model-First строит граф…");
    const policy = sourcePolicy || "practical_only";
    const depth =
      policy === "hybrid" || policy === "academic_only"
        ? "Deep Mechanics"
        : "Standard";
    try {
      const res = await steeringGenerate({
        target_goal: text,
        user_level: "Intermediate/Advanced",
        depth_level: depth,
        source_policy: policy,
        generation_mode: policy === "academic_only" ? "consensus" : "fast",
        control_axis: "steering",
        steering_mode: "per_node",
      });
      setGoal(text);
      let graph = res.graph;
      if (!graph) {
        setGenStatus("Штурвал (по нодам): ожидаем worker…");
        const job = await waitWorkJob(res.work_job_id, {
          onLongWait: () => setLongWaitNotice(LONG_WAIT_MESSAGE),
          onPoll: (info) => setGenPollNotice(formatJobPollNotice(info)),
        });
        graph = job.result;
      }
      if (graph?.curriculum_id) {
        rememberCurriculumControlAxis(graph.curriculum_id, "steering");
        await loadWorkspace(graph.curriculum_id);
      } else {
        setError("Штурвал: граф сгенерирован, но curriculum_id не найден.");
      }
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setGenStatus("");
      setLongWaitNotice("");
      setGenBusyAction(null);
      setWorkspaceBusy(false);
    }
  }

  /** Mode 2: прежний Gate 1/2 поток; после Gate 2 approve (handleGateApprove)

   * идёт РЕАЛЬНЫЙ тяжёлый Map-Reduce по ВСЕМ approved-статьям сразу →
   * CurriculumGraph из ОДНОЙ сфокусированной ноды (node_kind=
   * "steering_standalone", см. steering_topic_node_service.py) — дальше как
   * обычный курс, через loadWorkspace, тем же путём, что Mode 1. */
  async function runSteeringCreateStandaloneDigest(text) {
    setError("");
    setWorkspaceBusy(true);
    setGenBusyAction("create");
    setGenStatus("Штурвал (обзор темы): TaxonomyService + Light Discovery…");
    const policy = sourcePolicy || "practical_only";
    const depth =
      policy === "hybrid" || policy === "academic_only"
        ? "Deep Mechanics"
        : "Standard";
    try {
      const res = await steeringGenerate({
        target_goal: text,
        user_level: "Intermediate/Advanced",
        depth_level: depth,
        source_policy: policy,
        generation_mode: policy === "academic_only" ? "consensus" : "fast",
        control_axis: "steering",
        steering_mode: "standalone_digest",
      });
      setGoal(text);
      const gateNode = {
        node_id: "steering-gate-node",
        title: "🎯 Согласование источников (Gate 1)",
        layer: "foundation",
        category: "Штурвал",
        prerequisites: [],
      };
      const virtualCurriculumId = `steering-${res.work_job_id}`;
      setCurriculum({ curriculum_id: virtualCurriculumId, nodes: [gateNode] });
      setStatuses({ [gateNode.node_id]: "in_progress" });
      setSelectedNode(gateNode);
      setSelectedMaterialId(null);
      replaceSkillTreeSearchParams({
        curriculum: virtualCurriculumId,
        node: gateNode.node_id,
        material: "",
      });
      setSessions((prev) => ({
        ...prev,
        [gateNode.node_id]: {
          initialized: true,
          messages: [],
          steeringWorkJobId: res.work_job_id,
          steeringStatus: res.status,
          steeringCandidates: res.taxonomy_discovery?.candidate_articles || [],
          steeringDigests: [],
        },
      }));
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setGenStatus("");
      setLongWaitNotice("");
      setGenBusyAction(null);
      setWorkspaceBusy(false);
    }
  }

  /** Gate 1/2 approve из SteeringGatePanel (виртуальная нода-гейт). Gate 1 →

   * Batch Digest Generator (следующая пауза Gate 2, тот же гейт-узел). Gate
   * 2 → node_finalize_steering + постановка тяжёлого Map-Reduce в очередь
   * (generation_job_id); дожидаемся worker'а и заменяем виртуальную ноду
   * настоящим сгенерированным графом через loadWorkspace (тот же путь, что
   * обычный createCurriculum, см. work_handlers._run_steering_topic_digest —
   * он тоже вызывает save_curriculum_record). */
  async function handleGateApprove(selectedUrls) {
    const node = selectedNode;
    if (!node || node.node_id !== "steering-gate-node") return;
    const sess = sessions[node.node_id];
    if (!sess) return;
    const jobId = sess.steeringWorkJobId;
    setTutorBusyNodeId(node.node_id);
    setError("");
    try {
      if (sess.steeringStatus === "awaiting_gate_1") {
        const res = await steeringApproveGate1(jobId, selectedUrls);
        setSessions((prev) => ({
          ...prev,
          [node.node_id]: {
            ...prev[node.node_id],
            steeringStatus: res.status,
            steeringDigests: res.surface_digests?.digests || [],
            steeringApprovedCount: selectedUrls.length,
          },
        }));
        setSelectedNode((prev) =>
          prev && prev.node_id === node.node_id
            ? { ...prev, title: "📦 Обзор выжимок (Gate 2)" }
            : prev,
        );
      } else if (sess.steeringStatus === "awaiting_gate_2") {
        const res = await steeringApproveGate2(jobId, selectedUrls);
        setSessions((prev) => ({
          ...prev,
          [node.node_id]: {
            ...prev[node.node_id],
            steeringStatus: res.status,
            steeringGenerationJobId: res.generation_job_id,
          },
        }));
        setGenBusyAction("create");
        setGenStatus("Штурвал: тяжёлый Map-Reduce по утверждённым источникам…");
        const job = await waitWorkJob(res.generation_job_id, {
          onLongWait: () => setLongWaitNotice(LONG_WAIT_MESSAGE),
          onPoll: (info) => setGenPollNotice(formatJobPollNotice(info)),
        });
        if (job?.result?.curriculum_id) {
          await loadWorkspace(job.result.curriculum_id);
        }
      }
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setTutorBusyNodeId(null);
      setGenStatus("");
      setLongWaitNotice("");
      setGenBusyAction(null);
    }
  }

  /** Interaction Axis (per-node): пока только хранит выбор в сессии ноды —

   * фактическая отправка в NodeDeepDiveRequest/TopicQnaNodeDeepDiveRequest
   * — отдельный, ещё не начатый шаг (см. roadmap, Этап 4 backend
   * scaffolding). */
  function setNodeInteractionAxis(nodeId, axis) {
    setSessions((prev) => ({
      ...prev,
      [nodeId]: { ...(prev[nodeId] || { messages: [] }), interactionAxis: axis },
    }));
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
        {
          onLongWait: () => setLongWaitNotice(LONG_WAIT_MESSAGE),
          onPoll: (info) => setGenPollNotice(formatJobPollNotice(info)),
        },
      );
      setGoal("");
      setLayoutEpoch((n) => n + 1);
      await loadWorkspace(graph.curriculum_id);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      clearInterval(phaseTimer);
      setGenStatus("");
      setLongWaitNotice("");
      setGenBusyAction(null);
      setWorkspaceBusy(false);
    }
  }

  async function runCreateNewWhileLoaded(text) {
    clearCanvasForNewRoute();
    await runCreatePath(text);
  }

  useEffect(() => {
    function onPickMaterial(e) {
      const id = String(e.detail?.id || "").trim();
      if (!id) return;
      setSelectedMaterialId(id);
      replaceSkillTreeSearchParams({ material: id });
    }
    window.addEventListener("ke:select-material", onPickMaterial);
    return () => window.removeEventListener("ke:select-material", onPickMaterial);
  }, []);

  useEffect(() => {
    localStorage.setItem(MATERIAL_VIEW_LS, materialViewMode);
  }, [materialViewMode]);

  const applyNodeResponse = useCallback((nodeId, res, userMsg) => {
    if (res.error) {
      setError(res.error);
      return;
    }
    setStatuses((prev) => ({ ...prev, [nodeId]: res.node_status }));
    if (Array.isArray(res.mapped_source_ids)) {
      // Без этого патча selectedNode/curriculum.nodes хранят
      // mapped_source_ids на момент открытия ноды — панель "Адресация
      // ноды" в NodeDrawer остаётся пустой сразу после лекции, пока не
      // перезагрузить страницу (backend уже отдаёт свежие id в ответе хода).
      const freshMapped = res.mapped_source_ids;
      setSelectedNode((prev) =>
        prev && prev.node_id === nodeId
          ? { ...prev, mapped_source_ids: freshMapped }
          : prev,
      );
      setCurriculum((prev) => {
        if (!prev || !Array.isArray(prev.nodes)) return prev;
        const idx = prev.nodes.findIndex((n) => n.node_id === nodeId);
        if (idx === -1) return prev;
        const nodes = prev.nodes.slice();
        nodes[idx] = { ...nodes[idx], mapped_source_ids: freshMapped };
        return { ...prev, nodes };
      });
    }
    setSessions((prev) => {
      const old = prev[nodeId] || { messages: [] };
      const streamId = `stream-${nodeId}`;
      const messages = buildMessagesAfterChatComplete(
        res,
        userMsg,
        old.messages,
        streamId,
      );
      return {
        ...prev,
        [nodeId]: {
          // ...old сохраняет поля, которыми эта функция не управляет
          // (interactionAxis и т.п.) — раньше объект сессии пересобирался
          // ПОЛНОСТЬЮ по фиксированному списку ниже, и любое поле не из
          // этого списка (например Interaction Axis, выставленный в
          // NodeMasteryPanel) стиралось на первом же ответе тьютора.
          ...old,
          initialized: true,
          prepared: messages.length === 0 && Boolean(res.rag_facts_count || old.prepared),
          content: res.content,
          messages,
          ragLabels: res.rag_fact_labels || old.ragLabels || [],
          masteryDashboard: res.mastery_dashboard || old.masteryDashboard,
          coverageSummary:
            res.coverage_summary ||
            res.mastery_dashboard?.coverage_summary ||
            old.coverageSummary ||
            null,
          topicMasteryScore:
            res.topic_mastery_score ?? old.topicMasteryScore ?? 0,
          learningPhase: res.learning_phase || old.learningPhase,
          learningMode: res.learning_mode || old.learningMode,
          sourceRegistry: Array.isArray(res.source_registry)
            ? res.source_registry
            : old.sourceRegistry || [],
          lectureRagInspector: Array.isArray(res.lecture_rag_inspector)
            ? res.lecture_rag_inspector
            : old.lectureRagInspector || [],
          readyForTransition: Boolean(res.ready_for_transition),
          lastEvalDirective: String(
            res.last_eval_directive || old.lastEvalDirective || "",
          ).trim(),
          quickReplies: Array.isArray(res.quick_replies)
            ? res.quick_replies
            : [],
        },
      };
    });
  }, []);

  const openNode = useCallback(
    async (node) => {
      if (!curriculum || !node?.node_id) return;
      const sid = node.node_id;
      const initialized = Boolean(sessions[sid]?.initialized);

      if (tutorBusyNodeId !== null && !initialized) return;

      setSelectedNode(node);
      setSelectedMaterialId(null);
      replaceSkillTreeSearchParams({ node: sid, material: "" });
      setError("");
      if (initialized) {
        try {
          const regRes = await fetchNodeSourceRegistry(
            curriculum.curriculum_id,
            sid,
          );
          const freshReg = regRes.source_registry || [];
          setSessions((prev) => ({
            ...prev,
            [sid]: {
              ...(prev[sid] || { messages: [] }),
              sourceRegistry: freshReg,
            },
          }));
        } catch {
          /* keep cached registry */
        }
        return;
      }

      // Повторный клик по ноде, для которой Gate 1/2 уже идёт (например,
      // пользователь ещё не одобрил и снова кликнул по той же ноде на
      // канвасе) — просто показываем текущее состояние гейта, НЕ запускаем
      // discover заново (иначе слетел бы прогресс с Gate 2 обратно на Gate 1).
      const existingGateStatus = sessions[sid]?.nodeGateStatus;
      if (
        existingGateStatus === "awaiting_gate_1" ||
        existingGateStatus === "awaiting_gate_2"
      ) {
        return;
      }

      // Node Grounding Gate: ручной выбор источников — только в режиме
      // Штурвал (текущий переключатель). В Автопилоте нода ищет источники
      // сама (lazy grounding в nodeInitStream). Перехватываем ТОЛЬКО ещё не
      // прогруженные DEEP-ноды — safety-гейт против неконтролируемого SOTA-override
      // харвеста (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, Node
      // Grounding Gate). BASE-ноды и уже grounded DEEP-ноды идут прежним
      // путём без единого лишнего запроса.
      const needsNodeGate =
        controlAxis === "steering" &&
        node.node_risk_kind === "DEEP" &&
        node.grounding_status !== "grounded";
      if (needsNodeGate) {
        setTutorBusyNodeId(sid);
        setError("");
        try {
          const disc = await nodeGroundingDiscover(
            curriculum.curriculum_id,
            toNodeDataInput(node),
          );
          if ((disc.candidates || []).length > 0) {
            setSessions((prev) => ({
              ...prev,
              [sid]: {
                ...(prev[sid] || { messages: [] }),
                initialized: false,
                nodeGateStatus: "awaiting_gate_1",
                nodeGateCandidates: disc.candidates,
                nodeGateDigests: [],
              },
            }));
            return; // ждём Gate 1 — nodeInitStream ниже пока не вызывается
          }
        } catch (err) {
          // discover сам по себе fail-open на бэкенде; если и сетевой вызов
          // сломался — не блокируем ноду навсегда, идём обычным путём ниже.
          setError(String(err.message || err));
        } finally {
          setTutorBusyNodeId(null);
        }
      }

      setTutorBusyNodeId(sid);
      setTutorStageMessage("Проверяем материалы источника…");
      // Штурвал-сгенерированные ноды изначально несут только лёгкий Gate-2
      // дайджест (2-3 предложения Flash Lite), а не полноценный ingest —
      // см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md. Для любой другой ноды
      // (Autopilot, уже доингещенная Штурвал-нода) это мгновенный no-op на
      // бэкенде — ни одного лишнего сетевого вызова внутри самого запроса.
      // Fail-open: сбой здесь не должен блокировать открытие ноды.
      try {
        await ensureSteeringSourcesIngested(
          curriculum.curriculum_id,
          toNodeDataInput(node),
        );
      } catch {
        /* fail-open — идём в nodeInitStream с тем контентом, что есть */
      }
      setTutorStageMessage("");
      let pollTimer = null;
      try {
        // nodeInitStream (не nodeInit) — та же SSE-инфраструктура, что чат
        // (job_stream.py relay), даёт FSM stage-события (см. schemas/fsm.py)
        // на "подготовку ноды", а не только на ответ тьютора.
        let finalRes = null;
        let baseStage = "";
        const initStartedAt = Date.now();
        // Опрос бэка раз в 30 с (как waitWorkJob при генерации графа): job_id
        // приходит первым SSE-событием, статус тянем из /work-jobs/{id}.
        await nodeInitStream(
          curriculum.curriculum_id,
          toNodeDataInput(node),
          (evt) => {
            if (evt.type === "job" && evt.job_id && !pollTimer) {
              pollTimer = setInterval(async () => {
                try {
                  const j = await fetchWorkJob(evt.job_id);
                  setTutorStageMessage(
                    formatJobPollNotice(
                      {
                        status: j.status,
                        elapsedSec: Math.round((Date.now() - initStartedAt) / 1000),
                      },
                      baseStage,
                    ),
                  );
                } catch {
                  /* сбой опроса не должен ронять init-stream */
                }
              }, NODE_JOB_POLL_INTERVAL_MS);
            }
            if (evt.type === "stage" && evt.message) {
              baseStage = evt.message;
              setTutorStageMessage(evt.message);
            }
            if (evt.type === "complete" && evt.result) {
              finalRes = evt.result;
            }
            if (evt.type === "error") {
              throw new Error(evt.detail || "init-stream error");
            }
          },
          sessions[sid]?.interactionAxis || "lecture_self_check",
        );
        applyNodeResponse(sid, finalRes || {});
        const freshGraph = await refreshCurriculumGraph(curriculum.curriculum_id);
        if (freshGraph) {
          const freshNode = freshGraph.nodes.find((n) => n.node_id === sid);
          if (freshNode) setSelectedNode(freshNode);
        }
      } catch (err) {
        setError(String(err.message || err));
      } finally {
        if (pollTimer) clearInterval(pollTimer);
        setTutorBusyNodeId(null);
      }
    },
    [
      curriculum,
      sessions,
      tutorBusyNodeId,
      applyNodeResponse,
      refreshCurriculumGraph,
      controlAxis,
    ],
  );

  /** Gate 1/2 approve из SteeringGatePanel (kind: "node") — Node Grounding

   * Gate для ОДНОЙ реальной ноды (в отличие от handleGateApprove/Штурвала,
   * тут нет виртуальной ноды-курса: session хранится под настоящим
   * node_id). Gate 1 -> nodeGroundingDigest (Этап 2, BGE+Reranker+Gemma) ->
   * Gate 2 -> nodeGroundingFinalize (Этап 3, реальный Map-Reduce — тот же,
   * что уже используется lazy-grounding'ом) -> узел становится grounded,
   * дальше как обычно через nodeInitStream (existing
   * _apply_lazy_grounding_for_init в engine.py при grounded+source_ref
   * сам ничего не ищет, только refresh диаграмм). */
  async function handleNodeGateApprove(selectedUrls) {
    const node = selectedNode;
    if (!node) return;
    const sess = sessions[node.node_id];
    if (!sess || !sess.nodeGateStatus) return;
    setTutorBusyNodeId(node.node_id);
    setError("");
    try {
      if (sess.nodeGateStatus === "awaiting_gate_1") {
        const res = await nodeGroundingDigest(
          curriculum.curriculum_id,
          node.node_id,
          selectedUrls,
        );
        setSessions((prev) => ({
          ...prev,
          [node.node_id]: {
            ...prev[node.node_id],
            nodeGateStatus: "awaiting_gate_2",
            nodeGateDigests: res.digests || [],
            nodeGateApprovedCount: selectedUrls.length,
          },
        }));
      } else if (sess.nodeGateStatus === "awaiting_gate_2") {
        setTutorStageMessage("Загружаем материалы в ноду (Map-Reduce)…");
        await nodeGroundingFinalize(
          curriculum.curriculum_id,
          node.node_id,
          selectedUrls,
        );
        setSessions((prev) => {
          const next = { ...prev };
          delete next[node.node_id];
          return next;
        });
        const freshGraph = await refreshCurriculumGraph(curriculum.curriculum_id);
        const freshNode =
          (freshGraph?.nodes || []).find((n) => n.node_id === node.node_id) ||
          node;
        setSelectedNode(freshNode);
        setTutorBusyNodeId(null);
        setTutorStageMessage("");
        await openNode(freshNode);
        return;
      }
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setTutorBusyNodeId(null);
      setTutorStageMessage("");
    }
  }

  useEffect(() => {
    if (!curriculum?.nodes?.length || workspaceBusy) return;
    const params = new URLSearchParams(window.location.search);
    const nodeId = (params.get("node") || "").trim();
    if (!nodeId) return;
    if (selectedNode?.node_id === nodeId) return;
    const node = curriculum.nodes.find((n) => n.node_id === nodeId);
    if (node) {
      openNode(node);
      return;
    }
    replaceSkillTreeSearchParams({ node: "", material: "" });
  }, [
    curriculum?.curriculum_id,
    curriculum?.nodes,
    workspaceBusy,
    selectedNode?.node_id,
    openNode,
  ]);

  useEffect(() => {
    if (!selectedNode || workspaceBusy) return;
    const mid = (new URLSearchParams(window.location.search).get("material") ||
      "").trim();
    if (mid) setSelectedMaterialId(mid);
  }, [selectedNode?.node_id, workspaceBusy]);

  async function sendTutorMessage(text) {
    if (!curriculum || !selectedNode || tutorBusyNodeId !== null) return;
    const msg = (text || "").trim();
    if (!msg) return;
    const nid = selectedNode.node_id;
    setTutorBusyNodeId(nid);
    setTutorStageMessage("");
    onTutorPendingUser(msg);
    try {
      let finalRes = null;
      let streamed = "";
      await nodeChatStream(
        curriculum.curriculum_id,
        toNodeDataInput(selectedNode),
        msg,
        (evt) => {
          if (evt.type === "stage" && evt.message) {
            setTutorStageMessage(evt.message);
          }
          if (evt.type === "token" && evt.text) {
            streamed += evt.text;
            setSessions((prev) => {
              const old = prev[nid] || { messages: [], initialized: true };
              const msgs = [...(old.messages || [])];
              const streamId = `stream-${nid}`;
              const idx = msgs.findIndex((m) => m.msg_id === streamId);
              const row = {
                role: "tutor",
                content: streamed,
                msg_id: streamId,
              };
              if (idx >= 0) msgs[idx] = row;
              else msgs.push(row);
              return {
                ...prev,
                [nid]: { ...old, messages: msgs },
              };
            });
          }
          if (evt.type === "complete" && evt.result) {
            finalRes = evt.result;
          }
          if (evt.type === "error") {
            throw new Error(evt.detail || "chat-stream error");
          }
        },
        sessions[nid]?.interactionAxis || "lecture_self_check",
      );
      if (finalRes) {
        applyNodeResponse(nid, finalRes, msg);
      }
    } catch (err) {
      setError(String(err.message || err));
      const streamId = `stream-${nid}`;
      setSessions((prev) => {
        const old = prev[nid] || { messages: [] };
        return {
          ...prev,
          [nid]: {
            ...old,
            messages: (old.messages || []).filter((m) => m.msg_id !== streamId),
          },
        };
      });
    } finally {
      setTutorBusyNodeId(null);
      setTutorStageMessage("");
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
      setTutorStageMessage("");
    }
  }

  function onTutorPendingUser(userMsg) {
    if (!selectedNode) return;
    const nid = selectedNode.node_id;
    setSessions((prev) => {
      const old = prev[nid] || { messages: [], initialized: true };
      const u = (userMsg || "").trim();
      if (!u) return prev;
      const msgs = old.messages || [];
      const last = msgs[msgs.length - 1];
      // Allow repeating the same lecture stub; only skip double-pending at tail.
      if (
        last?.role === "user" &&
        userMessageMatches(last.content, u) &&
        isPendingMsgId(last.msg_id)
      ) {
        return prev;
      }
      return {
        ...prev,
        [nid]: {
          ...old,
          messages: (() => {
            const next = [...msgs];
            next.push({
              role: "user",
              content: u,
              msg_id: `pending-${Date.now()}`,
            });
            return sortDialogMessages(next);
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

  async function restartSelectedNode() {
    if (!curriculum || !selectedNode || tutorBusyNodeId !== null) return;
    const nid = selectedNode.node_id;
    const title = (selectedNode.title || nid).trim();
    const ok = window.confirm(
      `Сбросить прогресс ноды «${title}»?\n\n` +
        "Будут удалены материалы, диалог, память тьютора и статус прохождения. " +
        "Затем заново соберутся персональные факты (RAG) и подготовится сессия.",
    );
    if (!ok) return;

    setTutorBusyNodeId(nid);
    setError("");
    setSessions((prev) => {
      const next = { ...prev };
      delete next[nid];
      return next;
    });
    setStatuses((prev) => ({ ...prev, [nid]: "unexplored" }));

    try {
      const res = await nodeRestart(
        curriculum.curriculum_id,
        toNodeDataInput(selectedNode),
        {
          onLongWait: () => setLongWaitNotice(LONG_WAIT_MESSAGE),
          onPoll: (info) => setTutorStageMessage(formatJobPollNotice(info)),
        },
      );
      applyNodeResponse(nid, res);
      const freshGraph = await refreshCurriculumGraph(curriculum.curriculum_id);
      if (freshGraph) {
        const freshNode = freshGraph.nodes.find((n) => n.node_id === nid);
        if (freshNode) setSelectedNode(freshNode);
      }
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setTutorBusyNodeId(null);
      setTutorStageMessage("");
      setLongWaitNotice("");
    }
  }

  const session = selectedNode ? sessions[selectedNode.node_id] : null;
  const sessionReady = Boolean(session?.initialized);
  const activeId = curriculum?.curriculum_id || "";
  const tutorBusy = tutorBusyNodeId !== null;
  const composeLocked =
    tutorBusy || (selectedNode && !sessionReady);
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
        controlAxis,
        onControlAxisChange: setControlAxis,
        steeringMode,
        onSteeringModeChange: setSteeringMode,
        activeCurriculumId: activeId,
        workspaceBusy,
        genStatus,
        longWaitNotice,
        genPollNotice,
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
          gridTemplateColumns: `minmax(180px, 1fr) 6px ${leftColWidth}px 6px ${rightColWidth}px`,
        },
      },
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
          setLeftColWidth((w) => {
            const maxWidth = Math.min(
              720,
              maxResizableColumnWidth(rightColRef.current),
            );
            const next = Math.min(maxWidth, Math.max(240, w - dx));
            leftColRef.current = next;
            return next;
          });
        },
        onDragEnd: persistColWidths,
      }),
      React.createElement(
        "aside",
        { className: "skill-chat-column" },
        curriculum && selectedNode
          ? React.createElement(NodeTutorChat, {
              session,
              onSend: sendTutorMessage,
              disabled: composeLocked,
              generating: nodeGenerating,
              stageMessage: tutorStageMessage,
              curriculumId: curriculum.curriculum_id,
              nodeData: toNodeDataInput(selectedNode),
              curriculum,
              onOpenNode: openNode,
            })
          : React.createElement(
              "div",
              { className: "tutor-panel skill-chat-placeholder" },
              React.createElement("h3", null, "Чат с тьютором"),
              React.createElement(
                "p",
                { className: "muted" },
                curriculum
                  ? "Выберите ноду на карте — диалог откроется здесь."
                  : "Создайте или выберите маршрут, затем откройте ноду на графе.",
              ),
            ),
      ),
      React.createElement(ColumnResizer, {
        onDragDelta: (dx) => {
          setRightColWidth((w) => {
            const maxWidth = Math.min(
              960,
              maxResizableColumnWidth(leftColRef.current),
            );
            const next = Math.min(maxWidth, Math.max(280, w - dx));
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
            onRestart: restartSelectedNode,
            composeLocked,
            nodeGenerating,
            sessions,
            selectedMaterialId,
            materialViewMode,
            onMaterialViewModeChange: setMaterialViewMode,
            onInteractionAxisChange: setNodeInteractionAxis,
            onGateApprove: handleGateApprove,
            onNodeGateApprove: handleNodeGateApprove,
          })
        : React.createElement("aside", { className: "node-drawer empty" }),
    ),
  );
}
