import React, { useEffect, useState } from "react";
import {
  fetchWorkspace,
  nodeInitStream,
  nodeChat,
  toNodeDataInput,
  buildMessagesAfterChatComplete,
} from "./api.js";
import { LlmHtmlBlock } from "./LlmHtmlBlock.js";
import { SourceRegistryList } from "./SourceRegistryList.js";
import { NodeTutorChat } from "./NodeTutorChat.js";
import { structuredAnalysisToHtml } from "./llmTextRepair.js";

/**
 * Standalone-режим `?mode=topic&id=...&curriculum=...` — чистое фокусное
 * чтение одной ноды без канваса Skill Tree. `curriculumId` не входит в
 * буквальную сигнатуру из ТЗ (`<TopicFocusView nodeId={...} />`), но без
 * него нельзя вызвать ни один существующий node/*-эндпоинт (nodeInit/
 * nodeChat принимают только (curriculumId, nodeData) — "нода сама по себе"
 * бэкендом не адресуется) — main.js читает его тем же URLSearchParams,
 * что и nodeId, из параметра `curriculum`.
 *
 * interaction_axis="topic_qna" передаётся во все node/*-запросы этой
 * страницы (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Interaction Axis
 * подключён") — TOPIC_QNA_SYSTEM_PROMPT добавляется поверх обычного
 * dense-system в generate_dense_material.
 */
export function TopicFocusView({ nodeId, curriculumId }) {
  const [curriculum, setCurriculum] = useState(null);
  const [node, setNode] = useState(null);
  const [session, setSession] = useState({ messages: [] });
  const [stageMessage, setStageMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!curriculumId || !nodeId) {
      setError(
        "Не указан curriculum и/или id в URL (?mode=topic&id=...&curriculum=...).",
      );
      return;
    }
    let cancelled = false;
    (async () => {
      setBusy(true);
      setError("");
      try {
        const ws = await fetchWorkspace(curriculumId);
        const found = (ws.curriculum?.nodes || []).find(
          (n) => n.node_id === nodeId,
        );
        if (!found) {
          throw new Error(`Нода ${nodeId} не найдена в курсе ${curriculumId}`);
        }
        if (cancelled) return;
        setCurriculum(ws.curriculum);
        setNode(found);

        let finalRes = null;
        await nodeInitStream(
          curriculumId,
          toNodeDataInput(found),
          (evt) => {
            if (evt.type === "stage" && evt.message) setStageMessage(evt.message);
            if (evt.type === "complete" && evt.result) finalRes = evt.result;
            if (evt.type === "error") {
              throw new Error(evt.detail || "init-stream error");
            }
          },
          "topic_qna",
        );
        if (!cancelled) applyResponse(finalRes || {}, null);
      } catch (err) {
        if (!cancelled) setError(String(err.message || err));
      } finally {
        if (!cancelled) {
          setBusy(false);
          setStageMessage("");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curriculumId, nodeId]);

  function applyResponse(res, userMsg) {
    setSession((prev) => {
      const messages = buildMessagesAfterChatComplete(
        res,
        userMsg,
        prev.messages || [],
        "topic-focus-stream",
      );
      return {
        ...prev,
        initialized: true,
        content: res.content || prev.content,
        messages,
        readyForTransition: Boolean(res.ready_for_transition),
        quickReplies: Array.isArray(res.quick_replies) ? res.quick_replies : [],
        sourceRegistry: Array.isArray(res.source_registry)
          ? res.source_registry
          : prev.sourceRegistry || [],
      };
    });
  }

  async function send(text) {
    if (!curriculum || !node || busy) return;
    setBusy(true);
    setError("");
    try {
      const res = await nodeChat(
        curriculum.curriculum_id,
        toNodeDataInput(node),
        text,
        {},
        "topic_qna",
      );
      applyResponse(res, text);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  function openNextNode(nextNode) {
    if (!nextNode) return;
    const url = new URL(window.location.href);
    url.searchParams.set("id", nextNode.node_id);
    window.location.href = url.toString();
  }

  if (error) {
    return React.createElement(
      "div",
      { className: "topic-focus-container topic-focus-error" },
      error,
    );
  }
  if (!node) {
    return React.createElement(
      "div",
      { className: "topic-focus-container topic-focus-loading" },
      stageMessage || "Загрузка темы…",
    );
  }

  const content = session.content || {};
  const summaryHtml = (content.summary_html || "").trim()
    ? content.summary_html
    : structuredAnalysisToHtml(content.summary || "");

  return React.createElement(
    "div",
    { className: "topic-focus-container" },
    React.createElement(
      "div",
      { className: "topic-focus-reading" },
      React.createElement("h1", null, node.title),
      busy &&
        React.createElement(
          "p",
          { className: "muted topic-focus-busy-hint" },
          stageMessage || "Готовим материал…",
        ),
      summaryHtml
        ? React.createElement(LlmHtmlBlock, {
            html: summaryHtml,
            className: "drawer-summary md-body",
          })
        : content.summary
          ? React.createElement(
              "div",
              { className: "drawer-summary" },
              content.summary,
            )
          : null,
      React.createElement(SourceRegistryList, {
        registry: session.sourceRegistry || [],
      }),
    ),
    React.createElement(
      "div",
      { className: "topic-focus-chat" },
      React.createElement(NodeTutorChat, {
        session,
        onSend: send,
        disabled: busy,
        generating: busy,
        stageMessage,
        curriculumId: curriculum.curriculum_id,
        nodeData: toNodeDataInput(node),
        curriculum,
        onOpenNode: openNextNode,
      }),
    ),
  );
}
