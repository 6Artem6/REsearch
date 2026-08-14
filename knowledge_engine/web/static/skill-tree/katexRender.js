/** В чате вся математика — только inline \(…\), не display. */
function normalizeChatMathDelimiters(root) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const parent = node.parentElement;
    if (!parent) continue;
    if (/^(pre|code|script|style|textarea)$/i.test(parent.tagName)) continue;
    if (parent.closest("pre, code")) continue;
    let t = node.textContent || "";
    if (!/[$\\]/.test(t)) continue;
    const toInline = (inner) => `\\(${String(inner).trim()}\\)`;
    const next = t
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, inner) => toInline(inner))
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, inner) => toInline(inner))
      .replace(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g, (_, inner) => toInline(inner));
    if (next !== t) node.textContent = next;
  }
}

/** auto-render иногда всё равно даёт .katex-display — оставляем только .katex. */
function useInlineKatexOnly(root) {
  if (!root) return;
  root.querySelectorAll("span.katex-display").forEach((wrap) => {
    const inner = wrap.querySelector(":scope > span.katex");
    if (inner) wrap.replaceWith(inner);
    else wrap.classList.remove("katex-display");
  });
}

/** KaTeX в чате тьютора: только inline, без katex-display. */
export function renderMathInLlmHtml(root) {
  if (!root || typeof window.renderMathInElement !== "function") return;
  normalizeChatMathDelimiters(root);
  window.renderMathInElement(root, {
    delimiters: [{ left: "\\(", right: "\\)", display: false }],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
  useInlineKatexOnly(root);
}
