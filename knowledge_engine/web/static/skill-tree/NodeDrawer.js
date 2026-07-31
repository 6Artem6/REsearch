import React, { useRef } from "react";
import { toNodeDataInput } from "./api.js";
import { LlmHtmlBlock } from "./LlmHtmlBlock.js";
import { MermaidDiagramView } from "./MermaidDiagramView.js";
import { ResourceCard } from "./ResourceCard.js";
import { NodeMasteryPanel } from "./NodeMasteryPanel.js";
import { SourceRegistryList } from "./SourceRegistryList.js";
import { CodeSnippet } from "./CodeSnippet.js";
import { NodeSelectionExplain } from "./NodeSelectionExplain.js";
import { structuredAnalysisToHtml } from "./llmTextRepair.js";

function DiagramBlock({ diagram, nodeId }) {
  return React.createElement(MermaidDiagramView, { diagram, nodeId });
}

function sourceTierBadge(tier) {
  const t = (tier || "").trim().toLowerCase();
  if (t === "consensus") {
    return { className: "source-tier-badge consensus", label: "Consensus" };
  }
  if (t === "exa") {
    return { className: "source-tier-badge exa", label: "Exa" };
  }
  if (t === "searxng") {
    return { className: "source-tier-badge searxng", label: "SearXNG" };
  }
  if (t === "arxiv" || t === "semantic_scholar" || t === "searxng_science") {
    return { className: "source-tier-badge academic", label: "Academic" };
  }
  if (t === "gemini_grounding" || t === "whitelist_blog" || t === "gemini_web") {
    return { className: "source-tier-badge grounding", label: "Grounding" };
  }
  return null;
}

function courseLibraryIndex(curriculum) {
  const registry = curriculum?.curriculum_sources_registry || [];
  const routeSources = curriculum?.route_sources || [];
  const library =
    registry.length > 0
      ? registry
      : routeSources.map((r) => ({
          source_id: r.source_id,
          title: r.source_name,
          source_name: r.source_name,
          whitelist_domain: r.whitelist_category,
          source_type: "",
          url: r.url,
          why_read: r.why_read,
          source_tier: r.source_tier,
        }));
  const byId = Object.fromEntries(
    library.map((e) => [String(e.source_id || "").trim(), e]),
  );
  return { library, byId };
}

/** mapped_source_ids → title + url (реестр курса, resource_urls, сессия, source_ref). */
function resolveMappedSourceRows(selectedNode, curriculum, session) {
  const mappedIds = (selectedNode?.mapped_source_ids || [])
    .map((id) => String(id || "").trim())
    .filter(Boolean);
  if (!mappedIds.length) return [];

  const { byId } = courseLibraryIndex(curriculum);
  const resourceUrls = (selectedNode?.resource_urls || [])
    .map((u) => String(u || "").trim())
    .filter((u) => u.startsWith("http"));
  const sessionById = Object.fromEntries(
    (session?.sourceRegistry || []).map((e) => {
      const sid = String(e.id || e.source_id || "").trim();
      return [sid, e];
    }),
  );
  const ref = selectedNode?.source_ref;
  const primaryId = String(selectedNode?.primary_source_id || "").trim();

  return mappedIds.map((id, index) => {
    let title = id;
    let url = "";

    const lib = byId[id];
    if (lib) {
      title = (lib.title || lib.source_name || id).trim();
      url = String(lib.url || "").trim();
    }

    if (!url && sessionById[id]) {
      const s = sessionById[id];
      title = (s.title || title).trim();
      url = String(s.url || "").trim();
    }

    if (!url && resourceUrls[index]) {
      url = resourceUrls[index];
    }

    if (!url && ref && (id === primaryId || id === String(ref.source_id || "").trim())) {
      url = String(ref.url || "").trim();
      if (!title || title === id) {
        title = url.slice(0, 80) || title;
      }
    }

    return { source_id: id, title, url };
  });
}

function renderMappedSourceRow(row) {
  const badge = sourceTierBadge(row.source_tier);
  const label = (row.title || row.source_id).trim();
  return React.createElement(
    "li",
    { key: row.source_id },
    badge &&
      React.createElement(
        "span",
        { className: badge.className },
        badge.label === "Consensus" ? "🟣 " : "🟢 ",
        `[${badge.label}]`,
      ),
    React.createElement(
      "span",
      { className: "source-anchor-tag" },
      `[${row.source_id}]`,
    ),
    row.url
      ? React.createElement(
          "a",
          {
            className: "source-link",
            href: row.url,
            target: "_blank",
            rel: "noopener noreferrer",
          },
          label,
        )
      : React.createElement("span", null, label),
    !row.url &&
      React.createElement(
        "span",
        { className: "muted small" },
        " · URL не в реестре — см. «Библиотека курса»",
      ),
  );
}

function renderCourseKnowledgePool(selectedNode, curriculum) {
  const { library } = courseLibraryIndex(curriculum);

  return React.createElement(
    "details",
    { className: "drawer-section knowledge-pool-panel" },
    React.createElement(
      "summary",
      { className: "knowledge-pool-summary" },
      "Библиотека курса (Knowledge Pool)",
      React.createElement(
        "span",
        { className: "muted small" },
        ` · ${library.length} в реестре, не сессия ноды`,
      ),
    ),
    React.createElement(
      "p",
      { className: "muted small drawer-hint" },
      "Общий пул маршрута. Источники текущей ноды — только в блоке «Источники в материале» ниже.",
    ),
    React.createElement(
      "ul",
      { className: "source-registry-list course-library-list" },
      library.map((entry) => {
        const badge = sourceTierBadge(entry.source_tier);
        return React.createElement(
          "li",
          { key: entry.source_id || entry.url },
          badge &&
            React.createElement(
              "span",
              { className: badge.className },
              badge.label === "Consensus" ? "🟣 " : "🟢 ",
              `[${badge.label}]`,
            ),
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
        );
      }),
    ),
  );
}

function renderRouteCurriculumMaterials(selectedNode, curriculum, session) {
  const lm = selectedNode?.learning_materials?.primary_whitelist_source;
  const lres = selectedNode?.learning_resources || [];
  const pid = (selectedNode?.primary_source_id || "").trim();
  const { library, byId } = courseLibraryIndex(curriculum);
  const linked = library.find((r) => r.source_id === pid);
  const mappedRows = resolveMappedSourceRows(selectedNode, curriculum, session);
  const learningGoal = (selectedNode?.learning_goal || "").trim();

  if (!lm && !lres.length && !mappedRows.length && !learningGoal) return null;

  return React.createElement(
    "div",
    { className: "drawer-section route-curriculum-sources" },
    learningGoal &&
      React.createElement(
        "p",
        { className: "muted small node-learning-goal" },
        `Цель ноды: ${learningGoal}`,
      ),
    mappedRows.length > 0 &&
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
          mappedRows.map((row) => {
            const tier = byId[row.source_id]?.source_tier;
            return renderMappedSourceRow({ ...row, source_tier: tier });
          }),
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
            "Ссылка на реестр: ",
            React.createElement(
              "span",
              { className: "source-anchor-tag" },
              `[${linked.source_id}]`,
            ),
            linked.url
              ? React.createElement(
                  "a",
                  {
                    className: "source-link",
                    href: linked.url,
                    target: "_blank",
                    rel: "noopener noreferrer",
                  },
                  linked.title || linked.source_name || linked.url,
                )
              : React.createElement("span", null, linked.title || linked.source_id),
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
        renderRouteCurriculumMaterials(selectedNode, curriculum, session),
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
            : structuredAnalysisToHtml(content.summary || "")
              ? React.createElement(LlmHtmlBlock, {
                  html: structuredAnalysisToHtml(content.summary || ""),
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
      renderCourseKnowledgePool(selectedNode, curriculum),
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
