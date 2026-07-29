/** Подсветка кода — делегирует в /app/static/codeHighlightGlobal.js (CDN hljs). */
export function highlightCodeInRoot(root) {
  if (typeof window.keHighlightCodeInRoot === "function") {
    window.keHighlightCodeInRoot(root);
  }
}
