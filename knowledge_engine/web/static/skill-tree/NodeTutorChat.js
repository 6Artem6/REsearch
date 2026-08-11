import React, { useEffect, useMemo, useRef, useState } from "react";
import { sortDialogMessages, dialogMsgId, tutorHtmlMatchesContentForMessage } from "./api.js";
import { LlmHtmlBlock } from "./LlmHtmlBlock.js";
import {
  postprocessTutorHtml,
  structuredAnalysisToHtml,
  tutorMarkdownToHtml,
} from "./llmTextRepair.js";
import { NodeSelectionExplain } from "./NodeSelectionExplain.js";
import { RagInspectorPanel } from "./RagInspectorPanel.js";
import {
  NextNodeSelector,
  QUICK_REPLY_INTENTS,
  QuickReplyChips,
  listSuccessorNodes,
} from "./QuickReplyChips.js";

const QUICK = [
  {
    label: "Начать",
    text: "[begin] Начать",
  },
  {
    label: "Плотная лекция",
    text: "[mode:lecture] Дай плотный материал по теме.",
  },
  {
    label: "Самопроверка",
    text: "Один короткий вопрос самопроверки по материалу справа.",
  },
  {
    label: "Следующий модуль",
    text: "INTENT_FINALIZE: что я усовоил и куда логично перейти дальше?",
  },
];

function lastTutorMsgId(messages) {
  const list = sortDialogMessages(messages || []);
  for (let i = list.length - 1; i >= 0; i -= 1) {
    if (list[i]?.role === "tutor") {
      return (
        list[i].msg_id ||
        `tutor-${dialogMsgId(list[i]) ?? (list[i].content || "").slice(0, 40)}`
      );
    }
  }
  return "";
}

export function NodeTutorChat({
  session,
  onSend,
  disabled,
  generating,
  curriculumId,
  nodeData,
  curriculum,
  onOpenNode,
}) {
  const [input, setInput] = useState("");
  const [chipsDismissed, setChipsDismissed] = useState(false);
  const [nodePickerOpen, setNodePickerOpen] = useState(false);
  const materialRef = useRef(null);
  const inputRef = useRef(null);
  const messages = sortDialogMessages(session?.messages || []);
  const composeLocked = Boolean(disabled);
  const explainEnabled = Boolean(curriculumId && nodeData);
  const tutorTurnKey = lastTutorMsgId(messages);
  const showTransitionChips =
    Boolean(session?.readyForTransition) &&
    !chipsDismissed &&
    !generating &&
    Boolean(tutorTurnKey);

  useEffect(() => {
    // New tutor turn / new transition flag → show chips again.
    setChipsDismissed(false);
    setNodePickerOpen(false);
  }, [tutorTurnKey, session?.readyForTransition, session?.lastEvalDirective]);

  const successorNodes = useMemo(
    () => listSuccessorNodes(curriculum, nodeData?.node_id),
    [curriculum, nodeData?.node_id],
  );

  async function send(text) {
    const msg = (text || "").trim();
    if (!msg || composeLocked || !onSend) return;
    setChipsDismissed(true);
    setNodePickerOpen(false);
    try {
      await onSend(msg);
      setInput("");
    } catch {
      /* ошибка в родителе */
    }
  }

  function dismissChips() {
    setChipsDismissed(true);
    setNodePickerOpen(false);
  }

  function handleQuickReply(chip) {
    if (!chip || composeLocked) return;
    if (chip.intent === QUICK_REPLY_INTENTS.nextNode) {
      dismissChips();
      setNodePickerOpen(true);
      return;
    }
    if (chip.intent === QUICK_REPLY_INTENTS.clarify) {
      dismissChips();
      inputRef.current?.focus?.();
      return;
    }
    // Gloss / HOW / MECH → send [mode:…] intent to Prompt Factory.
    dismissChips();
    send(chip.intent);
  }

  function handleSelectNextNode(node) {
    setNodePickerOpen(false);
    if (node && onOpenNode) onOpenNode(node);
  }

  return React.createElement(
    "div",
    { className: "tutor-panel" },
    React.createElement(NodeSelectionExplain, {
      curriculumId,
      nodeData,
      containerRef: materialRef,
      enabled: explainEnabled,
    }),
    generating &&
      React.createElement(
        "div",
        { className: "tutor-busy-hint", "aria-live": "polite" },
        "Генерация ответа… можно читать историю выше; новые сообщения временно недоступны.",
      ),
    React.createElement(
      "div",
      { className: "tutor-panel-scroll tutor-selectable", ref: materialRef },
      messages.length > 0 &&
        React.createElement(
          "p",
          { className: "tutor-selectable-hint muted" },
          "Выделите фрагмент в ответе тьютора — «Объяснить».",
        ),
      React.createElement(
        "div",
        { className: "chat-log node-selectable-material" },
        messages.map((m) => {
          const msgKey =
            m.msg_id ||
            `${m.role}-${dialogMsgId(m) ?? (m.content || "").slice(0, 40)}`;
          const tutorHtml = postprocessTutorHtml(m.contentHtml || "");
          const useTutorHtml =
            m.role === "tutor" &&
            tutorHtml &&
            tutorHtmlMatchesContentForMessage(m.content || "", tutorHtml);
          const tutorMarkdownHtml =
            m.role === "tutor" ? tutorMarkdownToHtml(m.content || "") : "";
          const isLastTutor = m.role === "tutor" && msgKey === tutorTurnKey;
          return React.createElement(
            "div",
            {
              key: msgKey,
              className: `chat-msg ${m.role}`,
              "data-msg-id": m.msg_id || "",
              "data-role": m.role,
            },
            useTutorHtml
              ? React.createElement(LlmHtmlBlock, {
                  html: tutorHtml,
                  className: "md-body chat-md",
                })
              : m.role === "tutor" && structuredAnalysisToHtml(m.content || "")
                ? React.createElement(LlmHtmlBlock, {
                    html: structuredAnalysisToHtml(m.content || ""),
                    className: "md-body chat-md",
                  })
                : tutorMarkdownHtml
                  ? React.createElement(LlmHtmlBlock, {
                      html: tutorMarkdownHtml,
                      className: "md-body chat-md",
                    })
                  : React.createElement(
                      "div",
                      { className: "chat-plain lecture-plain" },
                      m.content || "",
                    ),
            isLastTutor
              ? React.createElement(QuickReplyChips, {
                  visible: showTransitionChips,
                  session,
                  nodeLayer: nodeData?.layer || session?.nodeLayer || "foundation",
                  disabled: composeLocked,
                  onChip: handleQuickReply,
                })
              : null,
          );
        }),
      ),
      React.createElement(RagInspectorPanel, {
        items: session?.lectureRagInspector || [],
      }),
    ),
    React.createElement(NextNodeSelector, {
      open: nodePickerOpen,
      nodes: successorNodes,
      onSelect: handleSelectNextNode,
      onClose: () => setNodePickerOpen(false),
    }),
    React.createElement(
      "div",
      {
        className: composeLocked
          ? "tutor-panel-compose tutor-panel-compose-locked"
          : "tutor-panel-compose",
      },
      React.createElement(
        "div",
        { className: "quick-chips" },
        QUICK.map((q) =>
          React.createElement(
            "button",
            {
              key: q.label,
              type: "button",
              disabled: composeLocked,
              onClick: () => send(q.text),
            },
            q.label,
          ),
        ),
      ),
      React.createElement(
        "form",
        {
          className: "chat-form",
          onSubmit: (e) => {
            e.preventDefault();
            if (!composeLocked) send(input);
          },
        },
        React.createElement("input", {
          ref: inputRef,
          value: input,
          onChange: (e) => {
            setInput(e.target.value);
            if ((e.target.value || "").trim()) setChipsDismissed(true);
          },
          placeholder: composeLocked
            ? "Ждём ответ тьютора…"
            : "Вопрос тьютору…",
          disabled: composeLocked,
        }),
        React.createElement(
          "button",
          { type: "submit", disabled: composeLocked },
          composeLocked ? "…" : "Отправить",
        ),
      ),
    ),
  );
}
