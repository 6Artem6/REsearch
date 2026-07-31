import React from "react";

/** Список источников [Sx] как в обзоре v07 (scholarly list). */
export function SourceRegistryList({ registry }) {
  const items = registry || [];
  if (!items.length) return null;
  return React.createElement(
    "div",
    { className: "drawer-section source-registry-section" },
    React.createElement("h3", null, "Источники в материале"),
    React.createElement(
      "p",
      { className: "muted small drawer-hint" },
      "Реестр сессии ноды (mapped / лекция), не глобальная библиотека курса.",
    ),
    React.createElement(
      "ul",
      { className: "source-registry-list" },
      items.map((entry, i) => {
        const sid = entry.id || entry.source_id || `S${i + 1}`;
        const title = (entry.title || "source").trim();
        const url = (entry.url || "").trim();
        const snippet = (entry.snippet || "").trim().slice(0, 280);
        return React.createElement(
          "li",
          { key: sid },
          React.createElement("span", { className: "source-anchor-tag" }, `[${sid}]`),
          url
            ? React.createElement(
                "a",
                {
                  className: "source-link",
                  href: url,
                  target: "_blank",
                  rel: "noopener noreferrer",
                },
                title,
              )
            : React.createElement("span", null, title),
          snippet &&
            React.createElement("p", { className: "muted snippet" }, snippet),
        );
      }),
    ),
  );
}
