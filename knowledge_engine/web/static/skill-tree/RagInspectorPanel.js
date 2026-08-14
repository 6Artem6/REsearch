import React, { useState } from "react";

function RagChunkCard({ item }) {
  const [open, setOpen] = useState(false);
  const ragId = item.rag_id || "[R?]";
  const title = (item.title || "source").trim();
  const url = (item.url || "").trim();
  const chunkIdx = Number(item.chunk_index) || 0;
  const chunkTotal = Number(item.chunks_in_doc) || 0;
  const score =
    typeof item.cosine_score === "number"
      ? item.cosine_score.toFixed(3)
      : String(item.cosine_score || "—");
  const chunkText = (item.chunk_text || "").trim();

  return React.createElement(
    "article",
    { className: "rag-inspector-card" },
    React.createElement(
      "header",
      { className: "rag-inspector-card-head" },
      React.createElement("span", { className: "rag-inspector-badge" }, ragId),
      url
        ? React.createElement(
            "a",
            {
              className: "source-link",
              href: url,
              target: "_blank",
              rel: "noopener noreferrer",
            },
            title,
          )
        : React.createElement("span", { className: "rag-inspector-title" }, title),
      React.createElement(
        "span",
        { className: "muted small rag-inspector-meta" },
        chunkIdx > 0 && chunkTotal > 0
          ? `Chunk ${chunkIdx}/${chunkTotal} · cos=${score}`
          : `cos=${score}`,
      ),
    ),
    chunkText &&
      React.createElement(
        "button",
        {
          type: "button",
          className: "rag-inspector-toggle",
          onClick: () => setOpen((v) => !v),
        },
        open ? "Скрыть текст чанка" : "Показать текст чанка",
      ),
    open &&
      chunkText &&
      React.createElement(
        "pre",
        { className: "rag-inspector-chunk-text" },
        chunkText,
      ),
  );
}

/** Collapsible panel: RAG chunks shown to Reduce (сверка [R1]…[Rn]). */
export function RagInspectorPanel({ items }) {
  const list = Array.isArray(items) ? items : [];
  if (!list.length) return null;

  return React.createElement(
    "details",
    { className: "drawer-section rag-inspector-panel", open: false },
    React.createElement(
      "summary",
      { className: "rag-inspector-summary" },
      "🔍 Использованные RAG-контексты (Inspector)",
      React.createElement(
        "span",
        { className: "muted small" },
        ` · ${list.length} чанков`,
      ),
    ),
    React.createElement(
      "p",
      { className: "muted small drawer-hint" },
      "Точные фрагменты из блока НАЧАЛО МАТЕРИАЛА перед генерацией лекции. Сверяйте сноски [R1]… в тексте.",
    ),
    React.createElement(
      "div",
      { className: "rag-inspector-list" },
      list.map((item, i) =>
        React.createElement(RagChunkCard, {
          key: `${item.rag_id || "R"}-${item.doc_id || i}-${item.chunk_index || 0}`,
          item,
        }),
      ),
    ),
  );
}
