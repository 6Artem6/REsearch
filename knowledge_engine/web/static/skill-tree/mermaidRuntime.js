/** Единая инициализация Mermaid (до любого mermaid.run в дочерних компонентах). */
export const MERMAID_INIT = {
  startOnLoad: false,
  theme: "dark",
  securityLevel: "loose",
  themeVariables: {
    fontSize: "14px",
    fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
    primaryTextColor: "#eceff4",
    lineColor: "#7eb8b8",
    primaryBorderColor: "#4ec9b0",
  },
  flowchart: {
    useMaxWidth: false,
    htmlLabels: true,
    padding: 28,
    nodeSpacing: 56,
    rankSpacing: 64,
    curve: "basis",
  },
  sequence: {
    useMaxWidth: false,
    wrap: true,
    width: 240,
    messageFontSize: 11,
    noteFontSize: 11,
    actorFontSize: 12,
    messageMargin: 48,
    boxMargin: 10,
    mirrorActors: false,
  },
};

let initialized = false;
let runChain = Promise.resolve();

export function ensureMermaidInitialized() {
  if (!window.mermaid) return false;
  if (!initialized) {
    window.mermaid.initialize(MERMAID_INIT);
    initialized = true;
  }
  return true;
}

export function waitForMermaidLibrary(timeoutMs = 4000) {
  if (window.mermaid) {
    ensureMermaidInitialized();
    return Promise.resolve(true);
  }
  const start = Date.now();
  return new Promise((resolve) => {
    function tick() {
      if (window.mermaid) {
        ensureMermaidInitialized();
        resolve(true);
        return;
      }
      if (Date.now() - start >= timeoutMs) {
        resolve(false);
        return;
      }
      setTimeout(tick, 40);
    }
    tick();
  });
}

/** Сериализация mermaid.run — параллельные схемы в списке материалов не ломают парсер. */
export function queueMermaidRun(task) {
  const job = runChain.then(() => task());
  runChain = job.catch(() => {});
  return job;
}

if (typeof window !== "undefined" && window.mermaid) {
  ensureMermaidInitialized();
}

export function waitForElementSize(el, minW = 32, minH = 32, maxFrames = 48) {
  if (!el) return Promise.resolve(false);
  return new Promise((resolve) => {
    let frames = 0;
    function tick() {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w >= minW && h >= minH) {
        resolve(true);
        return;
      }
      frames += 1;
      if (frames >= maxFrames) {
        resolve(w > 0 && h > 0);
        return;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}
