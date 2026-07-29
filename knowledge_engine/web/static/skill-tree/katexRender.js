/** KaTeX auto-render для HTML от llm_markdown_service (как в app.js). */
export function renderMathInLlmHtml(root) {
  if (!root || typeof window.renderMathInElement !== "function") return;
  window.renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
}
