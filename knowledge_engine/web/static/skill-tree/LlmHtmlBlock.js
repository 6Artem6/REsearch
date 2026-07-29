import React, { useEffect, useRef } from "react";
import { renderMathInLlmHtml } from "./katexRender.js";
import { highlightCodeInRoot } from "./codeHighlight.js";

/**
 * Блок HTML из сервиса llm_markdown_to_html (.md-body).
 */
export function LlmHtmlBlock({ html, className = "md-body" }) {
  const ref = useRef(null);
  const safe = (html || "").trim();

  useEffect(() => {
    if (ref.current && safe) {
      renderMathInLlmHtml(ref.current);
      highlightCodeInRoot(ref.current);
    }
  }, [safe]);

  if (!safe) return null;
  return React.createElement("div", {
    ref,
    className,
    dangerouslySetInnerHTML: { __html: safe },
  });
}
