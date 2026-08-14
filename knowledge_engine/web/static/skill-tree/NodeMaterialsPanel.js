import React, { useEffect, useMemo, useRef } from "react";
import { DiagramBlock } from "./NodeDrawerDiagram.js";
import { ResourceCard } from "./ResourceCard.js";
import { CodeSnippet } from "./CodeSnippet.js";

function kindLabel(kind) {
  if (kind === "diagram") return "Схема";
  if (kind === "code") return "Код";
  return "Карточка";
}

function materialHeaderLabel(item) {
  const title = String(item.payload?.title || "").trim();
  const id = item.id || "";
  const kind = kindLabel(item.kind);
  if (item.kind === "card") {
    return `${kind} ${id}`;
  }
  if (title) return `${kind} ${id}: ${title}`;
  return `${kind} ${id}`;
}

function MaterialItemBody({ item, nodeId }) {
  const { kind, payload } = item;
  if (kind === "diagram") {
    return React.createElement(DiagramBlock, {
      diagram: payload.mermaid,
      nodeId,
      compact: true,
    });
  }
  if (kind === "code") {
    return React.createElement(CodeSnippet, {
      code: payload.code,
      language: payload.language || undefined,
    });
  }
  return React.createElement(ResourceCard, { item: payload });
}

export function NodeMaterialsPanel({
  items,
  viewMode,
  onViewModeChange,
  selectedId,
  nodeId,
}) {
  const listRef = useRef(null);
  const activeIndex = useMemo(() => {
    if (!selectedId) return 0;
    const idx = items.findIndex((it) => it.id === selectedId);
    return idx >= 0 ? idx : 0;
  }, [items, selectedId]);

  useEffect(() => {
    if (!selectedId || viewMode !== "list" || !listRef.current) return;
    const el = listRef.current.querySelector(
      `[data-material-id="${selectedId}"]`,
    );
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [selectedId, viewMode, items.length]);

  if (!items.length) return null;

  return React.createElement(
    "div",
    { className: "drawer-section material-panel" },
    React.createElement(
      "div",
      { className: "material-panel-head" },
      React.createElement("h3", null, "Материалы"),
      React.createElement(
        "div",
        { className: "material-view-toggle" },
        React.createElement(
          "button",
          {
            type: "button",
            className: viewMode === "list" ? "active" : "",
            onClick: () => onViewModeChange("list"),
          },
          "Список",
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: viewMode === "carousel" ? "active" : "",
            onClick: () => onViewModeChange("carousel"),
          },
          "Карусель",
        ),
      ),
    ),
    viewMode === "carousel"
      ? React.createElement(CarouselView, {
          items,
          activeIndex,
          nodeId,
        })
      : React.createElement(
          "div",
          { className: "material-list", ref: listRef },
          items.map((item) =>
            React.createElement(
              "article",
              {
                key: `${item.kind}-${item.id}`,
                id: `ke-material-${item.id}`,
                "data-material-id": item.id,
                className:
                  selectedId === item.id
                    ? "material-asset material-asset-active"
                    : "material-asset",
              },
              React.createElement(
                "header",
                { className: "material-asset-label" },
                React.createElement(
                  "span",
                  { className: "material-asset-heading" },
                  materialHeaderLabel(item),
                ),
              ),
              React.createElement(MaterialItemBody, { item, nodeId }),
            ),
          ),
        ),
  );
}

function CarouselView({ items, activeIndex, nodeId }) {
  const [index, setIndex] = React.useState(activeIndex);
  useEffect(() => {
    setIndex(activeIndex);
  }, [activeIndex]);

  const item = items[index] || items[0];
  if (!item) return null;

  function go(delta) {
    setIndex((i) => {
      const next = i + delta;
      if (next < 0) return items.length - 1;
      if (next >= items.length) return 0;
      return next;
    });
  }

  return React.createElement(
    "div",
    {
      className: "material-carousel",
      id: `ke-material-${item.id}`,
      "data-material-id": item.id,
    },
    React.createElement(
      "div",
      { className: "material-carousel-toolbar" },
      React.createElement(
        "button",
        {
          type: "button",
          className: "diagram-zoom-btn",
          onClick: () => go(-1),
        },
        "‹",
      ),
      React.createElement(
        "span",
        { className: "material-carousel-meta" },
        `${index + 1} / ${items.length} · ${materialHeaderLabel(item)}`,
      ),
      React.createElement(
        "button",
        {
          type: "button",
          className: "diagram-zoom-btn",
          onClick: () => go(1),
        },
        "›",
      ),
    ),
    React.createElement(
      "div",
      { className: "material-carousel-body material-asset-active" },
      React.createElement(MaterialItemBody, { item, nodeId }),
    ),
    React.createElement(
      "div",
      { className: "material-carousel-dots" },
      items.map((it, i) =>
        React.createElement("button", {
          key: `${it.kind}-${it.id}`,
          type: "button",
          className: i === index ? "active" : "",
          title: it.id,
          onClick: () => setIndex(i),
        }),
      ),
    ),
  );
}
