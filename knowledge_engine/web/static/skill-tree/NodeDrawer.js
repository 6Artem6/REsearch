import React, { useRef } from "react";
import { toNodeDataInput } from "./api.js";
import { LlmHtmlBlock } from "./LlmHtmlBlock.js";
import { MermaidDiagramView } from "./MermaidDiagramView.js";
import { ResourceCard } from "./ResourceCard.js";
import { NodeMasteryPanel } from "./NodeMasteryPanel.js";
import { SourceRegistryList } from "./SourceRegistryList.js";
import { CodeSnippet } from "./CodeSnippet.js";
import { NodeSelectionExplain } from "./NodeSelectionExplain.js";

function DiagramBlock({ diagram, nodeId }) {
  return React.createElement(MermaidDiagramView, { diagram, nodeId });
}

function renderRouteCurriculumMaterials(selectedNode, curriculum) {
  const registry = curriculum?.curriculum_sources_registry || [];
  const routeSources = curriculum?.route_sources || [];
  const library =
    registry.length > 0
      ? registry
      : routeSources.map((r) => ({
          source_id: r.source_id,
          title: r.source_name,
          whitelist_domain: r.whitelist_category,
          source_type: "",
          url: r.url,
          why_read: r.why_read,
        }));
  const lm = selectedNode?.learning_materials?.primary_whitelist_source;
  const lres = selectedNode?.learning_resources || [];
  const pid = (selectedNode?.primary_source_id || "").trim();
  const mappedIds = (selectedNode?.mapped_source_ids || []).filter(Boolean);
  const byId = Object.fromEntries(
    library.map((e) => [e.source_id, e]),
  );
  const linked = library.find((r) => r.source_id === pid);
  const mappedEntries = mappedIds.map((id) => byId[id]).filter(Boolean);
  const learningGoal = (selectedNode?.learning_goal || "").trim();

  if (!library.length && !lm && !lres.length && !mappedIds.length) return null;

  return React.createElement(
    "div",
    { className: "drawer-section route-curriculum-sources" },
    React.createElement("h3", null, "Материалы маршрута (Whitelist)"),
    learningGoal &&
      React.createElement(
        "p",
        { className: "muted small node-learning-goal" },
        `Цель ноды: ${learningGoal}`,
      ),
    library.length > 0 &&
      React.createElement(
        "div",
        { className: "drawer-subsection" },
        React.createElement(
          "h4",
          { className: "drawer-subtitle" },
          "Библиотека курса (curriculum_sources_registry)",
        ),
        React.createElement(
          "ul",
          { className: "source-registry-list" },
          library.map((entry) =>
            React.createElement(
              "li",
              { key: entry.source_id || entry.url },
              React.createElement(
                "span",
                { className: "source-anchor-tag" },
                `[${entry.source_id}]`,
              ),
              entry.url
                ? React.createElement(
                    "a",
                    {
                      className: "source-link",
                      href: entry.url,
                      target: "_blank",
                      rel: "noopener noreferrer",
                    },
                    entry.title || entry.url,
                  )
                : React.createElement("span", null, entry.title || "source"),
              entry.whitelist_domain &&
                React.createElement(
                  "span",
                  { className: "muted small" },
                  ` · ${entry.whitelist_domain}`,
                ),
              entry.why_read &&
                React.createElement("p", { className: "muted snippet" }, entry.why_read),
            ),
          ),
        ),
      ),
    mappedEntries.length > 0 &&
      React.createElement(
        "div",
        { className: "drawer-subsection" },
        React.createElement(
          "h4",
          { className: "drawer-subtitle" },
          "Адресация ноды (mapped_source_ids)",
        ),
        React.createElement(
          "ul",
          { className: "source-registry-list mapped-sources" },
          mappedEntries.map((entry) =>
            React.createElement(
              "li",
              { key: entry.source_id },
              React.createElement(
                "span",
                { className: "source-anchor-tag" },
                `[${entry.source_id}]`,
              ),
              entry.title || entry.source_id,
            ),
          ),
        ),
      ),
    lm &&
      React.createElement(
        "div",
        { className: "drawer-subsection" },
        React.createElement("h4", { className: "drawer-subtitle" }, "Фундамент ноды"),
        linked &&
          React.createElement(
            "p",
            { className: "muted small" },
            `Ссылка на реестр: [${linked.source_id}] ${linked.source_name}`,
          ),
        React.createElement("p", null, lm.source_name),
        React.createElement("p", { className: "muted" }, lm.chapter_or_article),
        (lm.core_concepts || []).length > 0 &&
          React.createElement(
            "ul",
            { className: "drawer-concepts" },
            lm.core_concepts.map((c) => React.createElement("li", { key: c }, c)),
          ),
      ),
    lres.length > 0 &&
      React.createElement(
        "div",
        { className: "drawer-subsection" },
        React.createElement("h4", { className: "drawer-subtitle" }, "Ссылки на чтение"),
        React.createElement(
          "div",
          { className: "resource-card-list" },
          lres.map((r, i) => React.createElement(ResourceCard, { key: i, item: r })),
        ),
      ),
  );
}

export function NodeDrawer({
  curriculum,
  selectedNode,
  session,
  statuses,
  onSelectPrereq,
  onModeSelect,
  onVerify,
  composeLocked,
  nodeGenerating,
  sessions,
}) {
  const materialRef = useRef(null);

  if (!selectedNode) {
    return React.createElement(
      "aside",
      { className: "node-drawer empty" },
      "Выберите ноду на карте или сгенерируйте учебный путь.",
    );
  }

  const st = statuses[selectedNode.node_id] || "unexplored";
  const content = session?.content || {};
  const refs = content.references || [];
  const snippets = content.code_snippets || [];
  const registry = session?.sourceRegistry || [];

  const nodeData = toNodeDataInput(selectedNode);

  return React.createElement(
    "aside",
    { className: "node-drawer" },
    React.createElement(NodeSelectionExplain, {
      curriculumId: curriculum.curriculum_id,
      nodeData,
      containerRef: materialRef,
      enabled: Boolean(
        content.summary ||
          content.summary_html ||
          refs.length ||
          snippets.length,
      ),
    }),
    React.createElement(
      "div",
      { className: "drawer-scroll node-selectable-material", ref: materialRef },
      React.createElement(
        "div",
        { className: "drawer-header" },
        React.createElement("h2", null, selectedNode.title),
        React.createElement(NodeMasteryPanel, {
          status: st,
          masteryDashboard: session?.masteryDashboard,
          topicMasteryScore: session?.topicMasteryScore,
          onModeSelect: onModeSelect,
          disabled: composeLocked,
        }),
        React.createElement(
          "div",
          { className: "drawer-meta" },
          React.createElement("span", { className: "chip" }, selectedNode.layer),
          React.createElement("span", { className: "chip" }, st),
          selectedNode.category &&
            React.createElement("span", { className: "chip" }, selectedNode.category),
        ),
        renderRouteCurriculumMaterials(selectedNode, curriculum),
        (selectedNode.prerequisites || []).length > 0 &&
          React.createElement(
            "div",
            { className: "drawer-section" },
            React.createElement("h3", null, "Предшествующие темы"),
            React.createElement(
              "div",
              { className: "drawer-meta" },
              selectedNode.prerequisites.map((pid) => {
                const pre = curriculum.nodes.find((n) => n.node_id === pid);
                const preInit = Boolean(sessions?.[pid]?.initialized);
                return React.createElement(
                  "span",
                  {
                    key: pid,
                    className: "chip clickable",
                    onClick: () => {
                      if (composeLocked && !preInit) return;
                      pre && onSelectPrereq(pre);
                    },
                    title:
                      composeLocked && !preInit
                        ? "Дождитесь завершения генерации"
                        : undefined,
                    style:
                      composeLocked && !preInit
                        ? { opacity: 0.5, cursor: "not-allowed" }
                        : undefined,
                  },
                  pre?.title || pid,
                );
              }),
            ),
          ),
      ),
      nodeGenerating &&
        React.createElement(
          "p",
          { className: "muted drawer-gen-hint" },
          "Генерация для этой ноды… материал ниже можно читать и прокручивать.",
        ),
      content.summary &&
        React.createElement(
          "div",
          { className: "drawer-section node-selectable-material" },
          React.createElement("h3", null, "Суть механики"),
          React.createElement(
            "p",
            { className: "muted small drawer-hint" },
            "Выделите фрагмент — «Объяснить» и сложные вопросы, как в обзоре анализа.",
          ),
          (content.summary_html || "").trim()
            ? React.createElement(LlmHtmlBlock, {
                html: content.summary_html,
                className: "drawer-summary md-body",
              })
            : React.createElement(
                "div",
                { className: "drawer-summary" },
                content.summary,
              ),
        ),
      React.createElement(SourceRegistryList, { registry }),
      React.createElement(DiagramBlock, {
        diagram: content.diagram,
        nodeId: selectedNode.node_id,
      }),
      refs.length > 0 &&
        React.createElement(
          "div",
          { className: "drawer-section" },
          React.createElement("h3", null, "Карточки материалов"),
          React.createElement(
            "div",
            { className: "resource-card-list" },
            refs.map((r, i) =>
              React.createElement(ResourceCard, { key: i, item: r }),
            ),
          ),
        ),
      snippets.length > 0 &&
        React.createElement(
          "div",
          { className: "drawer-section" },
          React.createElement("h3", null, "Код и edge cases"),
          snippets.map((block, i) =>
            React.createElement(CodeSnippet, { key: i, code: block }),
          ),
        ),
      React.createElement(
        "button",
        {
          type: "button",
          className: "verify-btn",
          onClick: () => {
            if (!composeLocked) onVerify?.();
          },
          disabled: composeLocked,
        },
        "Финальная проверка",
      ),
    ),
  );
}
