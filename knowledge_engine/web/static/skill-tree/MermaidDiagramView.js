import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  extractMermaidSource,
  formatMermaidInner,
  repairDiagramMarkdown,
  softenMermaidSource,
} from "./mermaidExtract.js";

const MIN_SCALE = 0.35;
const MAX_SCALE = 5;
const LS_ZOOM = "skillTreeDiagramZoom";

function clamp(n, lo, hi) {
  return Math.min(hi, Math.max(lo, n));
}

export function MermaidDiagramView({ diagram, nodeId }) {
  const rawDiagram = (diagram || "").trim();
  const text = repairDiagramMarkdown(rawDiagram);
  const hostRef = useRef(null);
  const viewportRef = useRef(null);
  const [scale, setScale] = useState(() => {
    const saved = Number(localStorage.getItem(LS_ZOOM));
    return saved > 0 ? clamp(saved, MIN_SCALE, MAX_SCALE) : 1;
  });
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const panStart = useRef(null);

  const fitToViewport = useCallback(() => {
    const viewport = viewportRef.current;
    const host = hostRef.current;
    const svg = host?.querySelector("svg");
    if (!viewport || !svg) return;
    const pad = 16;
    const vw = viewport.clientWidth - pad;
    const vh = viewport.clientHeight - pad;
    setScale((cur) => {
      const bb = svg.getBoundingClientRect();
      const unscaledW = bb.width / cur;
      const unscaledH = bb.height / cur;
      if (unscaledW < 8 || unscaledH < 8) return cur;
      const fit = clamp(
        Math.min(vw / unscaledW, vh / unscaledH),
        MIN_SCALE,
        MAX_SCALE,
      );
      return fit;
    });
    setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !text) return;

    function runMermaid(source, attempt) {
      const src = (source || "").trim();
      if (!src || !window.mermaid) {
        host.textContent = text || rawDiagram;
        return;
      }
      const id = `mmd-${nodeId}-${Date.now()}-${attempt}`;
      host.innerHTML = "";
      const el = document.createElement("div");
      el.className = "mermaid";
      el.id = id;
      el.textContent = src;
      host.appendChild(el);
      window.mermaid
        .run({ nodes: [el] })
        .then(() => {
          const svg = el.querySelector("svg");
          if (svg) {
            svg.style.maxWidth = "none";
            svg.style.height = "auto";
          }
          requestAnimationFrame(() => fitToViewport());
        })
        .catch((err) => {
          if (attempt === 0) {
            const soft = softenMermaidSource(src);
            if (soft && soft !== src) {
              runMermaid(soft, 1);
              return;
            }
          }
          if (attempt < 2) {
            const retry = formatMermaidInner(src);
            if (retry && retry !== src) {
              runMermaid(retry, attempt + 1);
              return;
            }
          }
          console.warn("mermaid.run failed", err, src);
          const pre = document.createElement("pre");
          pre.className = "mermaid-fallback";
          pre.textContent = src;
          host.innerHTML = "";
          host.appendChild(pre);
        });
    }

    const source = extractMermaidSource(text || rawDiagram);
    runMermaid(source, 0);
  }, [text, rawDiagram, nodeId, fitToViewport]);

  useEffect(() => {
    localStorage.setItem(LS_ZOOM, String(scale));
  }, [scale]);

  function zoomBy(factor) {
    setScale((s) => clamp(s * factor, MIN_SCALE, MAX_SCALE));
  }

  function onWheel(e) {
    if (!viewportRef.current?.contains(e.target)) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
    setScale((s) => clamp(s * factor, MIN_SCALE, MAX_SCALE));
  }

  function onPanStart(e) {
    if (e.button !== 0) return;
    panStart.current = {
      x: e.clientX,
      y: e.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  }

  function onPanMove(e) {
    if (!panStart.current) return;
    const dx = e.clientX - panStart.current.x;
    const dy = e.clientY - panStart.current.y;
    setPan({
      x: panStart.current.panX + dx,
      y: panStart.current.panY + dy,
    });
  }

  function onPanEnd() {
    panStart.current = null;
  }

  useEffect(() => {
    window.addEventListener("mousemove", onPanMove);
    window.addEventListener("mouseup", onPanEnd);
    return () => {
      window.removeEventListener("mousemove", onPanMove);
      window.removeEventListener("mouseup", onPanEnd);
    };
  }, []);

  if (!text) return null;

  return React.createElement(
    "div",
    { className: "drawer-section diagram-section" },
    React.createElement(
      "div",
      { className: "diagram-section-head" },
      React.createElement("h3", null, "Схема"),
      React.createElement(
        "div",
        { className: "diagram-zoom-toolbar" },
        React.createElement(
          "button",
          {
            type: "button",
            className: "diagram-zoom-btn",
            onClick: () => zoomBy(1 / 1.2),
            title: "Уменьшить",
          },
          "−",
        ),
        React.createElement(
          "span",
          { className: "diagram-zoom-label" },
          `${Math.round(scale * 100)}%`,
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: "diagram-zoom-btn",
            onClick: () => zoomBy(1.2),
            title: "Увеличить",
          },
          "+",
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: "diagram-zoom-btn",
            onClick: () => {
              setPan({ x: 0, y: 0 });
              setScale(1);
            },
            title: "Сброс",
          },
          "1:1",
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: "diagram-zoom-btn",
            onClick: () => fitToViewport(),
            title: "Подогнать по области",
          },
          "По области",
        ),
      ),
    ),
    React.createElement(
      "div",
      {
        className: "diagram-viewport",
        ref: viewportRef,
        onWheel,
        onMouseDown: onPanStart,
        title: "Колёсико — зум, перетаскивание — пан",
      },
      React.createElement("div", {
        className: "diagram-zoom-inner",
        style: {
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
        },
        ref: hostRef,
      }),
    ),
  );
}
