import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  extractMermaidSource,
  formatMermaidInner,
  repairDiagramMarkdown,
  softenMermaidSource,
} from "./mermaidExtract.js";
import { polishMermaidSvg, applyDiagramSvgScale } from "./mermaidSvgPolish.js";

const MIN_SCALE = 0.5;
const MAX_SCALE = 5;
const FIT_MIN_SCALE = 0.88;
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
    return saved > 0 ? clamp(saved, MIN_SCALE, MAX_SCALE) : 1.12;
  });
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const panStart = useRef(null);

  const fitToViewport = useCallback(() => {
    const viewport = viewportRef.current;
    const host = hostRef.current;
    const svg = host?.querySelector("svg");
    if (!viewport || !svg) return;
    const pad = 20;
    const vw = viewport.clientWidth - pad;
    const vh = viewport.clientHeight - pad;
    setScale(() => {
      applyDiagramSvgScale(svg, 1);
      const bb = svg.getBoundingClientRect();
      const unscaledW = bb.width;
      const unscaledH = bb.height;
      if (unscaledW < 8 || unscaledH < 8) return 1.12;
      const fit = Math.min(vw / unscaledW, vh / unscaledH);
      return clamp(fit, FIT_MIN_SCALE, MAX_SCALE);
    });
    setPan({ x: 0, y: 0 });
  }, []);

  const applyComfortableScale = useCallback(() => {
    const viewport = viewportRef.current;
    const host = hostRef.current;
    const svg = host?.querySelector("svg");
    if (!viewport || !svg) {
      setScale(1.12);
      return;
    }
    applyDiagramSvgScale(svg, 1);
    const pad = 20;
    const vw = viewport.clientWidth - pad;
    const vh = viewport.clientHeight - pad;
    const bb = svg.getBoundingClientRect();
    if (bb.width < 8 || bb.height < 8) {
      setScale(1.12);
      return;
    }
    const fit = Math.min(vw / bb.width, vh / bb.height);
    if (fit < 1) {
      setScale(clamp(fit, FIT_MIN_SCALE, 1));
    } else {
      setScale(clamp(Math.min(fit, 1.25), 1, 1.35));
    }
    setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const svg = hostRef.current?.querySelector("svg");
    applyDiagramSvgScale(svg, scale);
  }, [scale]);

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
            polishMermaidSvg(svg);
          }
          requestAnimationFrame(() => applyComfortableScale());
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
  }, [text, rawDiagram, nodeId, applyComfortableScale]);

  useEffect(() => {
    localStorage.setItem(LS_ZOOM, String(scale));
  }, [scale]);

  function zoomBy(factor) {
    setScale((s) => clamp(s * factor, MIN_SCALE, MAX_SCALE));
  }

  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const onWheelNative = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
      setScale((s) => clamp(s * factor, MIN_SCALE, MAX_SCALE));
    };
    el.addEventListener("wheel", onWheelNative, { passive: false });
    return () => el.removeEventListener("wheel", onWheelNative);
  }, []);

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
        onMouseDown: onPanStart,
        title: "Колёсико — зум, перетаскивание — пан",
      },
      React.createElement(
        "div",
        {
          className: "diagram-pan-layer",
          style: { transform: `translate(${pan.x}px, ${pan.y}px)` },
        },
        React.createElement("div", {
          className: "diagram-zoom-inner",
          ref: hostRef,
        }),
      ),
    ),
  );
}
