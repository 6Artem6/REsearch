import React, { useState, useRef } from "react";
import { sortDialogMessages, dialogMsgId } from "./api.js";
import { LlmHtmlBlock } from "./LlmHtmlBlock.js";
import { repairLlMText, structuredAnalysisToHtml } from "./llmTextRepair.js";
import { NodeSelectionExplain } from "./NodeSelectionExplain.js";

const QUICK = [
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

export function NodeTutorChat({
  session,
  onSend,
  disabled,
  generating,
  curriculumId,
  nodeData,
}) {
  const [input, setInput] = useState("");
  const materialRef = useRef(null);
  const messages = sortDialogMessages(session?.messages || []);
  const composeLocked = Boolean(disabled);
  const explainEnabled = Boolean(curriculumId && nodeData);

  async function send(text) {
    const msg = (text || "").trim();
    if (!msg || composeLocked || !onSend) return;
    try {
      await onSend(msg);
      setInput("");
    } catch {
      /* ошибка в родителе */
    }
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
          return React.createElement(
            "div",
            {
              key: msgKey,
              className: `chat-msg ${m.role}`,
              "data-msg-id": m.msg_id || "",
              "data-role": m.role,
            },
            m.role === "tutor" && (m.contentHtml || "").length > 0
              ? React.createElement(LlmHtmlBlock, {
                  html: m.contentHtml,
                  className: "md-body chat-md",
                })
              : m.role === "tutor" && structuredAnalysisToHtml(m.content || "")
                ? React.createElement(LlmHtmlBlock, {
                    html: structuredAnalysisToHtml(m.content || ""),
                    className: "md-body chat-md",
                  })
                : React.createElement(
                    "div",
                    { className: "chat-plain" },
                    repairLlMText(m.content),
                  ),
          );
        }),
      ),
    ),
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
          value: input,
          onChange: (e) => setInput(e.target.value),
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
