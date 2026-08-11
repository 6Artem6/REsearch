import React, { useEffect, useRef } from "react";
import { renderMathInLlmHtml } from "./katexRender.js";
import { highlightCodeInRoot } from "./codeHighlight.js";

function wrapMarkdownTables(root) {
  if (!root) return;
  root.querySelectorAll("table").forEach((table) => {
    const parent = table.parentElement;
    if (parent?.classList?.contains("md-table-scroll")) return;
    const wrap = document.createElement("div");
    wrap.className = "md-table-scroll";
    table.parentNode?.insertBefore(wrap, table);
    wrap.appendChild(table);
  });
}

/**
 * Блок HTML из сервиса llm_markdown_to_html (.md-body).
 */
export function LlmHtmlBlock({ html, className = "md-body" }) {
  const ref = useRef(null);
  const safe = (html || "").trim();

  useEffect(() => {
    if (ref.current && safe) {
      wrapMarkdownTables(ref.current);
      renderMathInLlmHtml(ref.current);
      highlightCodeInRoot(ref.current);
    }
  }, [safe]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    function onClick(e) {
      const a = e.target.closest("a.ke-material-anchor");
      if (!a) return;
      e.preventDefault();
      const id = a.getAttribute("data-material-id");
      if (id) {
        window.dispatchEvent(
          new CustomEvent("ke:select-material", { detail: { id } }),
        );
      }
    }
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, [safe]);

  if (!safe) return null;
  return React.createElement("div", {
    ref,
    className,
    dangerouslySetInnerHTML: { __html: safe },
  });
}
