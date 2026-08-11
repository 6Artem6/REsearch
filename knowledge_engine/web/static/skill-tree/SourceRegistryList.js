import React from "react";

/** Сноска в тексте [Sx] + при наличии id библиотеки курса: [S1] · src_6 */
function sessionAnchorLabel(entry, index) {
  const sid =
    entry.id ||
    (String(entry.source_id || "").match(/^S\d+$/i) ? entry.source_id : null) ||
    `S${index + 1}`;
  const courseId = (entry.course_source_id || "").trim();
  if (courseId && courseId !== sid) {
    return `[${sid}] · ${courseId}`;
  }
  return `[${sid}]`;
}

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
      "Реестр сессии ноды: [S1] — сноска в тексте; src_N — id в библиотеке курса.",
    ),
    React.createElement(
      "ul",
      { className: "source-registry-list" },
      items.map((entry, i) => {
        const sid =
          entry.id ||
          (String(entry.source_id || "").match(/^S\d+$/i) ? entry.source_id : null) ||
          `S${i + 1}`;
        const anchorLabel = sessionAnchorLabel(entry, i);
        const title = (entry.title || "source").trim();
        const url = (entry.url || "").trim();
        const snippet = (entry.snippet || "").trim().slice(0, 280);
        return React.createElement(
          "li",
          { key: `${sid}-${entry.course_source_id || url || i}` },
          React.createElement(
            "span",
            { className: "source-anchor-tag" },
            anchorLabel,
          ),
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
