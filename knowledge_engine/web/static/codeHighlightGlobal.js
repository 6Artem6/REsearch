/** Глобальная подсветка (highlight.js) для app.js и skill-tree. */
(function () {
  const LANG_ALIASES = {
    py: "python",
    python3: "python",
    js: "javascript",
    ts: "typescript",
    sh: "bash",
    shell: "bash",
    yml: "yaml",
  };

  function inferLanguage(text) {
    const t = (text || "").trim();
    if (!t) return "plaintext";
    if (/^\s*#!\/bin\/(bash|sh)/m.test(t) || /^\s*(sudo |export )/m.test(t)) {
      return "bash";
    }
    if (/^\s*(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\s/im.test(t)) {
      return "sql";
    }
    if (/^\s*\{[\s\n]*"/.test(t) || /^\s*\[[\s\n]*\{/.test(t)) {
      return "json";
    }
    if (
      /\b(async def|await |import asyncio|from typing|@dataclass)/.test(t) ||
      /^\s*def \w+\s*\(/m.test(t)
    ) {
      return "python";
    }
    if (
      /\b(console\.log|function\s|const\s|let\s|=>|import \{)/.test(t) ||
      /^\s*export\s+(default\s+)?function/m.test(t)
    ) {
      return "javascript";
    }
    if (/^\s*(package |func \w+|go\.)/m.test(t)) return "go";
    return "python";
  }

  function normalizeLang(className) {
    const m = (className || "").match(/language-([\w-]+)/i);
    if (!m) return null;
    const raw = m[1].toLowerCase();
    return LANG_ALIASES[raw] || raw;
  }

  function highlightBlock(block) {
    if (!block || block.dataset.hljsDone === "1") return;
    const hljs = window.hljs;
    if (!hljs) return;

    let codeEl = block;
    if (block.tagName === "PRE") {
      let child = block.querySelector("code");
      if (!child) {
        child = document.createElement("code");
        child.textContent = block.textContent;
        block.textContent = "";
        block.appendChild(child);
      }
      codeEl = child;
    }

    let lang = normalizeLang(codeEl.className);
    if (!lang) {
      lang = inferLanguage(codeEl.textContent);
      codeEl.classList.add("language-" + lang);
    }

    try {
      if (lang && lang !== "plaintext" && hljs.getLanguage(lang)) {
        codeEl.innerHTML = hljs.highlight(codeEl.textContent, { language: lang }).value;
      } else {
        codeEl.innerHTML = hljs.highlightAuto(codeEl.textContent).value;
      }
      codeEl.classList.add("hljs");
      block.dataset.hljsDone = "1";
    } catch (_) {
      /* noop */
    }
  }

  function highlightCodeInRoot(root) {
    if (!root) return;
    root.querySelectorAll("pre code").forEach((el) => highlightBlock(el));
    root.querySelectorAll("pre.code-snippet, pre.hljs-snippet").forEach((el) =>
      highlightBlock(el),
    );
  }

  window.keHighlightCodeInRoot = highlightCodeInRoot;
})();
