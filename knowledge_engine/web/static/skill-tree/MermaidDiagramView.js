import {
  queueMermaidRun,
  waitForElementSize,
  waitForMermaidLibrary,
} from "./mermaidRuntime.js";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  extractMermaidSource,
  extractRenderableMermaid,
  reconstructXychartFromGarbage,
  formatMermaidInner,
  repairDiagramMarkdown,
  softenMermaidSource,
} from "./mermaidExtract.js";
import { polishMermaidSvg, applyDiagramSvgScale } from "./mermaidSvgPolish.js";

const MIN_SCALE = 0.5;
const MAX_SCALE = 5;
/** Доля viewport для схемы (остаток — поля, чтобы рамки не обрезались). */
const FIT_MARGIN_DEFAULT = 0.85;
const FIT_MARGIN_COMPACT = 0.82;
const FIT_MARGIN_XYCHART = 0.78;
const FIT_MARGIN_XYCHART_COMPACT = 0.72;
const MAX_AUTO_SCALE_DEFAULT = 1.08;
const MAX_AUTO_SCALE_COMPACT = 1;
const LS_ZOOM = "skillTreeDiagramZoom";

function clamp(n, lo, hi) {
  return Math.min(hi, Math.max(lo, n));
}

function computeFitScale(bb, vw, vh, fitMargin, maxAuto) {
  if (!bb || bb.width < 8 || bb.height < 8) return 1;
  const fit = Math.min(vw / bb.width, vh / bb.height) * fitMargin;
  return clamp(fit, MIN_SCALE, maxAuto);
}

function computeCenterPan(viewport, bb, scale, opts = {}) {
  const scaledW = bb.width * scale;
  const scaledH = bb.height * scale;
  const minInset = 10;
  const bottomInset = opts.xychart ? 6 : 0;
  return {
    x: Math.max(minInset, (viewport.clientWidth - scaledW) / 2),
    y: Math.max(
      minInset,
      (viewport.clientHeight - scaledH) / 2 - bottomInset,
    ),
  };
}

function isXychartMermaidSource(src) {
  return /^\s*xychart(?:-beta)?\b/im.test((src || "").trim());
}

function hostShowsXychart(host) {
  if (!host) return false;
  if (host.querySelector(".mermaid.mermaid-xychart")) return true;
  const pre = host.querySelector(".mermaid");
  return isXychartMermaidSource(pre?.textContent || "");
}

export function MermaidDiagramView({
  diagram,
  nodeId,
  compact = false,
  hideTitle = false,
}) {
  const rawDiagram = (diagram || "").trim();
  const text = repairDiagramMarkdown(rawDiagram);
  const sourceForDisplay =
    extractMermaidSource(text || rawDiagram) || text || rawDiagram;
  const hostRef = useRef(null);
  const viewportRef = useRef(null);
  const fitMargin = compact ? FIT_MARGIN_COMPACT : FIT_MARGIN_DEFAULT;
  const maxAutoScale = compact ? MAX_AUTO_SCALE_COMPACT : MAX_AUTO_SCALE_DEFAULT;
  const [viewMode, setViewMode] = useState("render");
  const [renderFailed, setRenderFailed] = useState(false);
  const [scale, setScale] = useState(() => {
    if (compact) return 1;
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
    const xychart = hostShowsXychart(host);
    const pad = xychart ? { x: 24, y: 32, bottom: 40 } : { x: 24, y: 24, bottom: 24 };
    const vw = viewport.clientWidth - pad.x * 2;
    const vh = viewport.clientHeight - pad.y - pad.bottom;
    const margin = xychart
      ? compact
        ? FIT_MARGIN_XYCHART_COMPACT
        : FIT_MARGIN_XYCHART
      : fitMargin;
    applyDiagramSvgScale(svg, 1);
    const bb = svg.getBoundingClientRect();
    const newScale = computeFitScale(bb, vw, vh, margin, MAX_SCALE);
    setScale(newScale);
    setPan(computeCenterPan(viewport, bb, newScale, { xychart }));
  }, [fitMargin, compact]);

  const applyComfortableScale = useCallback(() => {
    const viewport = viewportRef.current;
    const host = hostRef.current;
    const svg = host?.querySelector("svg");
    if (!viewport || !svg) {
      setScale(1);
      setPan({ x: 0, y: 0 });
      return;
    }
    const xychart = hostShowsXychart(host);
    applyDiagramSvgScale(svg, 1);
    const pad = xychart ? { x: 24, y: 32, bottom: 40 } : { x: 24, y: 24, bottom: 24 };
    const vw = viewport.clientWidth - pad.x * 2;
    const vh = viewport.clientHeight - pad.y - pad.bottom;
    const margin = xychart
      ? compact
        ? FIT_MARGIN_XYCHART_COMPACT
        : FIT_MARGIN_XYCHART
      : fitMargin;
    const bb = svg.getBoundingClientRect();
    const newScale = computeFitScale(bb, vw, vh, margin, maxAutoScale);
    setScale(newScale);
    setPan(computeCenterPan(viewport, bb, newScale, { xychart }));
  }, [fitMargin, maxAutoScale, compact]);

  useEffect(() => {
    const svg = hostRef.current?.querySelector("svg");
    applyDiagramSvgScale(svg, scale);
  }, [scale]);

  useEffect(() => {
    if (viewMode === "source") return undefined;

    let cancelled = false;
    let observer = null;
    let started = false;

    function beginRender() {
      if (started || cancelled) return;
      started = true;
      if (observer) observer.disconnect();

      const viewport = viewportRef.current;
      const host = hostRef.current;
      if (!host || !text) return;

      function runMermaid(source, attempt) {
        if (cancelled) return Promise.resolve();
        const src = (source || "").trim();
        if (!src) {
          setRenderFailed(true);
          return Promise.resolve();
        }

        setRenderFailed(false);
        const id = `mmd-${nodeId}-${Date.now()}-${attempt}`;
        host.innerHTML = "";
        const el = document.createElement("div");
        el.className = "mermaid";
        el.id = id;
        el.textContent = src;
        host.appendChild(el);

        return queueMermaidRun(() => {
          if (cancelled) return Promise.resolve();
          return window.mermaid.run({ nodes: [el] }).then(() => {
            if (cancelled) return;
            const svg = el.querySelector("svg");
            if (svg) {
              svg.style.maxWidth = "none";
              svg.style.height = "auto";
              const xychart = isXychartMermaidSource(src);
              polishMermaidSvg(svg, { xychart });
              if (xychart) el.classList.add("mermaid-xychart");
            }
            requestAnimationFrame(() => {
              if (!cancelled) applyComfortableScale();
            });
          });
        }).catch((err) => {
          if (cancelled) return;
          if (attempt === 0) {
            const rebuilt = reconstructXychartFromGarbage(text || rawDiagram);
            if (rebuilt && rebuilt !== src) {
              return runMermaid(rebuilt, 3);
            }
          }
          if (attempt === 0) {
            const soft = softenMermaidSource(src);
            if (soft && soft !== src) {
              return runMermaid(soft, 1);
            }
          }
          if (attempt < 2) {
            const retry = formatMermaidInner(src);
            if (retry && retry !== src) {
              return runMermaid(retry, attempt + 1);
            }
          }
          console.warn("mermaid.run failed", err, src);
          setRenderFailed(true);
          const pre = document.createElement("pre");
          pre.className = "mermaid-fallback";
          pre.textContent = src;
          host.innerHTML = "";
          host.appendChild(pre);
        });
      }

      (async () => {
        const ready = await waitForMermaidLibrary();
        if (cancelled) return;
        if (!ready) {
          host.textContent = text || rawDiagram;
          setRenderFailed(true);
          return;
        }
        await waitForElementSize(viewport);
        if (cancelled) return;
        const source = extractRenderableMermaid(text || rawDiagram);
        await runMermaid(source, 0);
      })();
    }

    const viewport = viewportRef.current;
    if (!viewport) {
      requestAnimationFrame(() => beginRender());
    } else {
      observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) beginRender();
        },
        { root: null, threshold: 0.02, rootMargin: "80px" },
      );
      observer.observe(viewport);
      requestAnimationFrame(() => {
        const r = viewport.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) beginRender();
      });
    }

    return () => {
      cancelled = true;
      if (observer) observer.disconnect();
    };
  }, [text, rawDiagram, nodeId, applyComfortableScale, viewMode]);

  useEffect(() => {
    if (!compact) localStorage.setItem(LS_ZOOM, String(scale));
  }, [scale, compact]);

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

  const toolbar = React.createElement(
    "div",
    { className: "diagram-zoom-toolbar" },
    React.createElement(
      "button",
      {
        type: "button",
        className: viewMode === "render" ? "diagram-zoom-btn active" : "diagram-zoom-btn",
        onClick: () => setViewMode("render"),
        title: "Визуальная схема",
      },
      "Схема",
    ),
    React.createElement(
      "button",
      {
        type: "button",
        className: viewMode === "source" ? "diagram-zoom-btn active" : "diagram-zoom-btn",
        onClick: () => setViewMode("source"),
        title: "Исходный Mermaid",
      },
      "Код",
    ),
    viewMode === "render" &&
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
    viewMode === "render" &&
      React.createElement(
        "span",
        { className: "diagram-zoom-label" },
        `${Math.round(scale * 100)}%`,
      ),
    viewMode === "render" &&
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
    viewMode === "render" &&
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
    viewMode === "render" &&
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
  );

  const renderFooter =
    renderFailed && viewMode === "render"
      ? React.createElement(
          "p",
          { className: "diagram-render-footer muted" },
          "Ошибка рендера Mermaid — откройте вкладку «Код» или проверьте xychart-beta / flowchart в сыром тексте.",
        )
      : null;

  return React.createElement(
    "div",
    { className: "drawer-section diagram-section" },
    !hideTitle &&
      React.createElement(
        "div",
        { className: "diagram-section-head" },
        React.createElement("h3", null, "Схема"),
        toolbar,
      ),
    hideTitle &&
      React.createElement(
        "div",
        { className: "diagram-section-head diagram-section-head-compact" },
        toolbar,
      ),
    viewMode === "source"
      ? React.createElement(
          "pre",
          { className: "diagram-source-code" },
          sourceForDisplay,
        )
      : React.createElement(
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
    renderFooter,
  );
}
