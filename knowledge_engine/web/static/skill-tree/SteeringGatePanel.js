import React, { useEffect, useMemo, useState } from "react";

function sourceBadge(hubOrTier) {
  const h = (hubOrTier || "").toLowerCase();
  // Node Grounding Gate передаёт точный source_tier (habr/exa/consensus/
  // arxiv/semantic_scholar); Штурвал (курс) — source_hub, свободный текст
  // хаба — отсюда сначала точное совпадение, потом substring-эвристика.
  if (h === "habr" || h.includes("habr")) {
    return { className: "source-tier-badge searxng", label: "Habr" };
  }
  if (h === "consensus") {
    return { className: "source-tier-badge consensus", label: "Consensus" };
  }
  if (
    h === "arxiv" ||
    h === "semantic_scholar" ||
    h.includes("arxiv") ||
    h.includes("semantic") ||
    h.includes("consensus")
  ) {
    return { className: "source-tier-badge academic", label: "Academic" };
  }
  return { className: "source-tier-badge exa", label: "Exa" };
}

function Gate1Candidate({ item, checked, disabled, onToggle }) {
  const badge = sourceBadge(item.source_tier || item.source_hub);
  // lead_paragraph — Штурвал (курс, CandidateArticleMeta); gist — Node
  // Grounding Gate (одна нода, NodeCandidatePassport). topic/tech_stack
  // существуют только у паспортов ноды — Штурвал их просто не пришлёт.
  const bodyText = item.gist || item.lead_paragraph || "";
  return React.createElement(
    "label",
    {
      className:
        "steering-gate-card" + (disabled ? " steering-gate-card-disabled" : ""),
    },
    React.createElement("input", {
      type: "checkbox",
      checked,
      disabled: Boolean(disabled),
      onChange: () => onToggle(item.url),
    }),
    React.createElement(
      "div",
      { className: "steering-gate-card-body" },
      React.createElement(
        "div",
        { className: "steering-gate-card-head" },
        React.createElement("span", { className: badge.className }, badge.label),
        React.createElement(
          "span",
          { className: "steering-gate-card-title" },
          item.title,
        ),
        item.topic &&
          React.createElement("span", { className: "muted small" }, item.topic),
      ),
      React.createElement("p", { className: "muted small" }, bodyText),
      (item.tech_stack || []).length > 0 &&
        React.createElement(
          "div",
          { className: "drawer-meta" },
          item.tech_stack.map((t) =>
            React.createElement("span", { key: t, className: "chip" }, t),
          ),
        ),
      React.createElement(
        "a",
        {
          className: "source-link small",
          href: item.url,
          target: "_blank",
          rel: "noopener noreferrer",
        },
        item.url,
      ),
    ),
  );
}

function Gate2Digest({ item, checked, onToggle }) {
  // NodeDigestItem (Node Grounding Gate, одна нода) — architecture/
  // practical_case/limitations. SurfaceDigestItem (Штурвал, курс) —
  // two_sentence_summary/problem_solved/main_tech_stack/company_or_author.
  // Различаем по наличию architecture — оба контракта не пересекаются.
  const isNodeDigest = item.architecture !== undefined;
  return React.createElement(
    "label",
    { className: "steering-gate-card" },
    React.createElement("input", {
      type: "checkbox",
      checked,
      onChange: () => onToggle(item.url),
    }),
    React.createElement(
      "div",
      { className: "steering-gate-card-body" },
      React.createElement(
        "div",
        { className: "steering-gate-card-head" },
        React.createElement(
          "span",
          { className: "steering-gate-card-title" },
          item.title,
        ),
        item.company_or_author &&
          React.createElement(
            "span",
            { className: "muted small" },
            item.company_or_author,
          ),
      ),
      isNodeDigest
        ? React.createElement(
            React.Fragment,
            null,
            React.createElement(
              "p",
              null,
              React.createElement("strong", null, "Архитектура: "),
              item.architecture,
            ),
            React.createElement(
              "p",
              null,
              React.createElement("strong", null, "Практический кейс: "),
              item.practical_case,
            ),
            React.createElement(
              "p",
              { className: "muted small" },
              React.createElement("strong", null, "Ограничения: "),
              item.limitations,
            ),
          )
        : React.createElement(
            React.Fragment,
            null,
            React.createElement("p", null, item.two_sentence_summary),
            item.problem_solved &&
              React.createElement(
                "p",
                { className: "muted small" },
                `Проблема: ${item.problem_solved}`,
              ),
          ),
      (item.main_tech_stack || []).length > 0 &&
        React.createElement(
          "div",
          { className: "drawer-meta" },
          item.main_tech_stack.map((t) =>
            React.createElement("span", { key: t, className: "chip" }, t),
          ),
        ),
      // Раньше здесь не было ссылки на источник вообще (в отличие от
      // Gate1Candidate) — на Gate 2 пользователь не мог проверить, на
      // какую именно статью ссылается дайджест, ни открыть оригинал.
      React.createElement(
        "a",
        {
          className: "source-link small",
          href: item.url,
          target: "_blank",
          rel: "noopener noreferrer",
        },
        item.url,
      ),
    ),
  );
}

/**
 * HITL-виджет Штурвала: Gate 1 (отбор кандидатов по лидам) / Gate 2 (обзор
 * готовых дайджестов перед тяжёлым Map-Reduce). Монтируется в NodeDrawer.js
 * для виртуальной ноды-гейта (см. steering-контракты steering_contracts.py:
 * CandidateArticleMeta для Gate 1, SurfaceDigestItem для Gate 2).
 */
export function SteeringGatePanel({
  status,
  candidates,
  digests,
  busy,
  onApprove,
  maxApproved,
  kind, // "course" (default, Штурвал) | "node" (Node Grounding Gate)
  approvedCount, // сколько URL было утверждено на Gate 1 (для Gate 2 hint)
}) {
  const isGate1 = status === "awaiting_gate_1";
  const isGate2 = status === "awaiting_gate_2";
  const items = isGate1 ? candidates || [] : isGate2 ? digests || [] : [];
  const itemUrls = useMemo(() => items.map((it) => it.url), [items]);
  // maxApproved — опциональный проп: Node Grounding Gate (Gate 1 для одной
  // ноды) требует ≤4 утверждённых. Gate 1 (первичный отбор из большого
  // пула кандидатов) по умолчанию НИЧЕГО не выбирает — осознанный ручной
  // отбор, не "снять лишнее" из предвыбранных всех. Gate 2 — наоборот: тут
  // уже только те статьи, что пользователь САМ утвердил на Gate 1 (теперь
  // с готовым дайджестом) — по умолчанию все отмечены, Gate 2 это ревью
  // ("снять то, что по дайджесту оказалось нерелевантным"), а не отбор с
  // нуля заново.
  const hasCap = typeof maxApproved === "number" && maxApproved > 0;
  const [selected, setSelected] = useState(() =>
    isGate2 ? new Set(itemUrls) : new Set(),
  );

  useEffect(() => {
    // Новый набор элементов (переход Gate 1 -> Gate 2, или новый список
    // кандидатов) — пересчитываем выбор к тому же дефолту, что при
    // монтировании (все отмечены на Gate 2, пусто на Gate 1).
    setSelected(isGate2 ? new Set(itemUrls) : new Set());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [itemUrls.join("|"), isGate2]);

  if (!isGate1 && !isGate2) return null;

  function toggle(url) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(url)) {
        next.delete(url);
        return next;
      }
      if (hasCap && next.size >= maxApproved) return prev; // лимит достигнут
      next.add(url);
      return next;
    });
  }

  const selectedUrls = itemUrls.filter((u) => selected.has(u));
  const atCap = hasCap && selectedUrls.length >= maxApproved;

  return React.createElement(
    "div",
    { className: "drawer-section steering-gate-panel" },
    React.createElement(
      "h3",
      null,
      isGate1 ? "Gate 1 · Отбор источников" : "Gate 2 · Обзор выжимок",
    ),
    React.createElement(
      "p",
      { className: "muted small drawer-hint" },
      isGate1
        ? kind === "node"
          ? "Отметьте статьи, которые войдут в дайджест (шаг 2 — точечный разбор)."
          : "Отметьте статьи, которые войдут в дайджест (шаг 2 — Batch Digest)."
        : kind === "node"
          ? "Отметьте статьи, дайджест которых войдёт в базу знаний ноды."
          : "Отметьте статьи, которые войдут в финальную генерацию курса.",
    ),
    hasCap &&
      React.createElement(
        "p",
        { className: "muted small steering-gate-cap-hint" },
        `Выбрано ${selectedUrls.length} из ${maxApproved} максимум.`,
      ),
    !hasCap &&
      items.length > 0 &&
      selectedUrls.length === 0 &&
      React.createElement(
        "p",
        { className: "muted small steering-gate-cap-hint" },
        "Отметьте хотя бы одну статью, чтобы продолжить.",
      ),
    // На Gate 1 могли утвердить дубли одной и той же статьи (та же
    // статья найдена через разные хабы — дедуп добавлен, но старые сессии
    // и редкие остаточные случаи возможны) или URL не открылся (сеть/
    // блокировка сайта) — дайджест на него просто не появится. Без этой
    // подсказки расхождение выглядит как "статьи пропали без причины".
    isGate2 &&
      typeof approvedCount === "number" &&
      approvedCount > items.length &&
      React.createElement(
        "p",
        { className: "muted small steering-gate-cap-hint" },
        `Из ${approvedCount} утверждённых на Gate 1 дайджест удалось ` +
          `построить для ${items.length} — остальные не открылись (дубли ` +
          "той же статьи или сайт заблокировал доступ) и не появятся здесь.",
      ),
    !items.length &&
      React.createElement(
        "p",
        { className: "muted" },
        "Кандидатов не найдено — попробуйте другую формулировку цели.",
      ),
    React.createElement(
      "div",
      { className: "steering-gate-list" },
      items.map((item) =>
        isGate1
          ? React.createElement(Gate1Candidate, {
              key: item.url,
              item,
              checked: selected.has(item.url),
              disabled: atCap && !selected.has(item.url),
              onToggle: toggle,
            })
          : React.createElement(Gate2Digest, {
              key: item.url,
              item,
              checked: selected.has(item.url),
              onToggle: toggle,
            }),
      ),
    ),
    items.length > 0 &&
      React.createElement(
        "button",
        {
          type: "button",
          className: "skill-btn-primary steering-gate-approve-btn",
          disabled:
            Boolean(busy) ||
            selectedUrls.length === 0 ||
            (hasCap && selectedUrls.length > maxApproved),
          onClick: () => onApprove(selectedUrls),
        },
        busy
          ? "…"
          : isGate1
            ? `Утвердить источники (Gate 1) · ${selectedUrls.length}`
            : kind === "node"
              ? `Загрузить в ноду (Gate 2) · ${selectedUrls.length}`
              : `Запустить генерацию (Gate 2) · ${selectedUrls.length}`,
      ),
  );
}
