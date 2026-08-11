import React, { useCallback, useEffect, useRef, useState } from "react";
import { nodeExplainSelectionStream, nodeSuggestQuestions } from "./api.js";
import { LlmHtmlBlock } from "./LlmHtmlBlock.js";
import { structuredAnalysisToHtml } from "./llmTextRepair.js";
import { getSurroundingParagraph } from "./selectionUtils.js";

const DEFAULT_QUESTIONS = [
  "Что это значит на практике?",
  "Каковы причины этого?",
  "Какая есть альтернатива?",
];

export function NodeSelectionExplain({
  curriculumId,
  nodeData,
  containerRef,
  enabled,
}) {
  const [toolbarPos, setToolbarPos] = useState(null);
  const [open, setOpen] = useState(false);
  const [selectedText, setSelectedText] = useState("");
  const [paragraph, setParagraph] = useState("");
  const [suggest, setSuggest] = useState(DEFAULT_QUESTIONS);
  const [suggestSource, setSuggestSource] = useState("");
  const [loadingSuggest, setLoadingSuggest] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [explanationHtml, setExplanationHtml] = useState("");
  const [explanationMd, setExplanationMd] = useState("");
  const [customQ, setCustomQ] = useState("");
  const abortRef = useRef(null);
  const explainAbortRef = useRef(null);

  const hideToolbar = useCallback(() => {
    setToolbarPos(null);
  }, []);

  useEffect(() => {
    if (!enabled) return;

    let debounceTimer = null;

    function syncSelection() {
      const root = containerRef?.current;
      if (!root) return;
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.rangeCount) {
        hideToolbar();
        return;
      }
      const range = sel.getRangeAt(0);
      const node = range.commonAncestorContainer;
      const el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
      if (!el || !root.contains(el)) {
        hideToolbar();
        return;
      }
      const text = sel.toString().trim();
      if (text.length < 2) {
        hideToolbar();
        return;
      }
      let rect = range.getBoundingClientRect();
      if (!rect.width && !rect.height) {
        const rects = range.getClientRects();
        if (rects.length > 0) rect = rects[0];
      }
      if (!rect.width && !rect.height) {
        hideToolbar();
        return;
      }
      setSelectedText(text);
      setParagraph(getSurroundingParagraph(el, text));
      setToolbarPos({
        x: Math.min(window.innerWidth - 16, Math.max(16, rect.left + rect.width / 2)),
        y: Math.max(8, rect.top),
      });
    }

    function onSelectionChange() {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(syncSelection, 80);
    }

    function onMouseUp() {
      setTimeout(syncSelection, 10);
    }

    document.addEventListener("selectionchange", onSelectionChange);
    document.addEventListener("mouseup", onMouseUp);
    return () => {
      if (debounceTimer) clearTimeout(debounceTimer);
      document.removeEventListener("selectionchange", onSelectionChange);
      document.removeEventListener("mouseup", onMouseUp);
    };
  }, [enabled, containerRef, hideToolbar]);

  async function loadSuggest() {
    if (!selectedText || !curriculumId || !nodeData) return;
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    setLoadingSuggest(true);
    try {
      const data = await nodeSuggestQuestions(
        curriculumId,
        nodeData,
        selectedText,
        paragraph,
        { signal: abortRef.current.signal },
      );
      if (Array.isArray(data.questions) && data.questions.length >= 3) {
        setSuggest(data.questions);
        setSuggestSource(data.source || "ollama");
      } else {
        setSuggest(DEFAULT_QUESTIONS);
        setSuggestSource("default");
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        setSuggest(DEFAULT_QUESTIONS);
        setSuggestSource("default");
      }
    } finally {
      setLoadingSuggest(false);
    }
  }

  function openDialog() {
    setOpen(true);
    setExplanationHtml("");
    setExplanationMd("");
    setCustomQ("");
    loadSuggest();
  }

  async function runExplain(question) {
    const q = (question || "").trim();
    if (!q || loadingExplain) return;
    if (explainAbortRef.current) explainAbortRef.current.abort();
    explainAbortRef.current = new AbortController();
    setLoadingExplain(true);
    setExplanationHtml("");
    setExplanationMd("");
    try {
      let streamed = "";
      await nodeExplainSelectionStream(
        curriculumId,
        nodeData,
        selectedText,
        paragraph,
        q,
        (evt) => {
          if (evt.type === "token" && evt.text) {
            streamed += evt.text;
            setExplanationMd(streamed);
          }
          if (evt.type === "complete" && evt.result) {
            const html = String(evt.result.explanation_html || "").trim();
            if (html) setExplanationHtml(html);
            else setExplanationMd(String(evt.result.explanation || streamed));
          }
          if (evt.type === "error") {
            throw new Error(evt.detail || "explain-stream error");
          }
        },
        { signal: explainAbortRef.current.signal },
      );
    } catch (e) {
      if (e.name !== "AbortError") {
        setExplanationHtml(`<p class="skill-error">${String(e.message || e)}</p>`);
        setExplanationMd("");
      }
    } finally {
      setLoadingExplain(false);
    }
  }

  const explainDisplayHtml =
    explanationHtml ||
    (explanationMd ? structuredAnalysisToHtml(explanationMd) : "");

  return React.createElement(
    React.Fragment,
    null,
    toolbarPos &&
      React.createElement(
        "div",
        {
          className: "node-explain-toolbar",
          style: {
            left: `${toolbarPos.x}px`,
            top: `${toolbarPos.y}px`,
          },
        },
        React.createElement(
          "button",
          { type: "button", onClick: openDialog },
          "Объяснить",
        ),
      ),
    open &&
      React.createElement(
        "div",
        { className: "node-explain-backdrop", onClick: () => setOpen(false) },
      ),
    open &&
      React.createElement(
        "aside",
        { className: "node-explain-dialog", role: "dialog" },
        React.createElement(
          "header",
          { className: "node-explain-head" },
          React.createElement("h3", null, "Пояснение фрагмента"),
          React.createElement(
            "button",
            {
              type: "button",
              className: "node-explain-close",
              onClick: () => setOpen(false),
            },
            "×",
          ),
        ),
        React.createElement("blockquote", { className: "node-explain-quote" }, selectedText),
        React.createElement(
          "button",
          {
            type: "button",
            className: "node-explain-primary",
            disabled: loadingExplain,
            onClick: () => runExplain("Объясни, что это значит?"),
          },
          "Объясни, что это значит?",
        ),
        React.createElement("div", { className: "node-explain-label" }, "Сложные вопросы"),
        loadingSuggest &&
          React.createElement("p", { className: "muted" }, "Подсказки (Ollama)…"),
        React.createElement(
          "div",
          { className: "node-explain-chips" },
          suggest.map((q, i) =>
            React.createElement(
              "button",
              {
                key: i,
                type: "button",
                disabled: loadingExplain,
                onClick: () => runExplain(q),
              },
              q,
            ),
          ),
        ),
        suggestSource === "default" &&
          React.createElement(
            "p",
            { className: "muted small" },
            "Базовые подсказки (Ollama недоступна)",
          ),
        React.createElement("textarea", {
          className: "node-explain-input",
          placeholder: "Свой вопрос…",
          value: customQ,
          onChange: (e) => setCustomQ(e.target.value),
        }),
        React.createElement(
          "button",
          {
            type: "button",
            className: "node-explain-primary",
            disabled: loadingExplain || !customQ.trim(),
            onClick: () => runExplain(customQ),
          },
          loadingExplain ? "…" : "Отправить",
        ),
        explainDisplayHtml &&
          React.createElement(LlmHtmlBlock, {
            html: explainDisplayHtml,
            className: "md-body node-explain-result",
          }),
      ),
  );
}
