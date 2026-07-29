import React from "react";

export function ResourceCard({ item }) {
  if (!item) return null;
  const title = (item.title || item.source_name || item.url || "").trim();
  const mins = item.read_time_minutes || item.read_time_minutes === 0
    ? item.read_time_minutes
  : null;
  return React.createElement(
    "article",
    { className: "resource-card" },
    React.createElement(
      "header",
      { className: "resource-card-head" },
      React.createElement(
        "a",
        {
          className: "resource-card-title",
          href: item.url,
          target: "_blank",
          rel: "noopener noreferrer",
        },
        title,
      ),
      mins != null &&
        mins > 0 &&
        React.createElement("span", { className: "resource-card-time" }, `~${mins} мин`),
    ),
    React.createElement("div", { className: "resource-card-source" }, item.source_name),
    (item.why_read || "").trim() &&
      React.createElement(
        "p",
        { className: "resource-card-why" },
        React.createElement("strong", null, "Зачем: "),
        item.why_read,
      ),
    (item.key_focus || "").trim() &&
      React.createElement(
        "p",
        { className: "resource-card-focus" },
        React.createElement("strong", null, "Фокус: "),
        item.key_focus,
      ),
  );
}
