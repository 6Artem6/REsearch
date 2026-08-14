/** Нормализация content ноды: накопленные диаграммы, код, карточки. */

/** Эвристика: не показывать plain-text / Deep dive заголовки как код. */
export function isLikelyCodeSnippet(text) {
  const raw = String(text || "").trim();
  if (raw.length < 4) return false;
  if (/^deep\s*dive\s*:/i.test(raw)) return false;
  if (/^```/m.test(raw)) return true;
  let inner = raw.replace(/^```[\w-]*\s*\n?/i, "").replace(/\n?```\s*$/i, "").trim();
  if (
    /\b(def|class|import|from|return|elif|else|async|await|function|const|let|var|SELECT|CREATE)\b/i.test(
      inner,
    )
  ) {
    return true;
  }
  if (/^\s*(if|for|while|switch)\s*[\(:]/im.test(inner)) return true;
  const lines = inner.split("\n").filter((ln) => ln.trim());
  if (lines.length >= 2) {
    const indented = lines.filter(
      (ln) =>
        ln.startsWith("    ") ||
        ln.startsWith("\t") ||
        /^\s*(def |class |#|\/\/)/.test(ln),
    ).length;
    if (indented >= 1) return true;
    if (lines.some((ln) => /[;{}]/.test(ln))) return true;
  }
  if (inner.includes(";") && (inner.includes("=") || inner.includes("("))) return true;
  if ((inner.match(/\(/g) || []).length >= 2 && inner.includes("=")) return true;
  return false;
}

function inferCodeTitle(code) {
  const raw = String(code || "").trim();
  if (!raw) return "Фрагмент кода";
  let inner = raw.replace(/^```[\w-]*\s*\n?/i, "").replace(/\n?```\s*$/i, "").trim();
  for (const ln of inner.split("\n")) {
    const s = ln.trim();
    if (s.startsWith("#") && !s.startsWith("#!")) return s.replace(/^#+\s*/, "").slice(0, 200);
  }
  const cls = /\bclass\s+(\w+)/.exec(inner);
  if (cls) return `Класс ${cls[1]}`.slice(0, 200);
  const fn = /\bdef\s+(\w+)/.exec(inner);
  if (fn) return `Функция ${fn[1]}`.slice(0, 200);
  return "Фрагмент кода";
}

function inferDiagramTitle(mermaid) {
  const raw = String(mermaid || "").trim();
  if (!raw) return "Схема";
  for (const ln of raw.split("\n")) {
    const s = ln.trim();
    if (s.startsWith("%%")) {
      const t = s.replace(/^%+/, "").trim();
      if (t) return t.slice(0, 200);
    }
  }
  return "Схема";
}

function assignUniqueCardIds(cards) {
  const used = new Set();
  const out = [];
  for (let i = 0; i < cards.length; i += 1) {
    const r = cards[i];
    let id = String(r.asset_id || "").trim();
    if (!id || used.has(id)) {
      let n = 1;
      while (used.has(`card-${n}`)) n += 1;
      id = `card-${n}`;
    }
    used.add(id);
    out.push({ ...r, asset_id: id });
  }
  return out;
}

export function normalizeNodeMaterials(content) {
  const c = content || {};
  let diagrams = Array.isArray(c.diagrams) ? [...c.diagrams] : [];
  if (!diagrams.length && String(c.diagram || "").trim()) {
    diagrams = [{ id: "diagram-1", title: "", mermaid: c.diagram }];
  }
  diagrams = diagrams.map((d, i) => ({
    ...d,
    title: String(d.title || "").trim() || inferDiagramTitle(d.mermaid) || `Схема ${i + 1}`,
  }));
  let codes = Array.isArray(c.code_assets) ? [...c.code_assets] : [];
  codes = codes
    .filter((item) => isLikelyCodeSnippet(item?.code))
    .map((item, i) => ({
      ...item,
      id: (item.id || `code-${i + 1}`).trim(),
      title: String(item.title || "").trim() || inferCodeTitle(item.code),
    }));
  if (!codes.length && Array.isArray(c.code_snippets)) {
    codes = c.code_snippets
      .map((code) => String(code || "").trim())
      .filter((code) => code && isLikelyCodeSnippet(code))
      .map((code, i) => ({
        id: `code-${i + 1}`,
        title: inferCodeTitle(code),
        code,
      }));
  }
  const cards = assignUniqueCardIds(
    (Array.isArray(c.references) ? c.references : []).map((r) => ({ ...r })),
  );
  return { diagrams, codes, cards };
}

export function flattenMaterials({ diagrams, codes, cards }) {
  const items = [];
  diagrams.forEach((d) => {
    const mermaid = String(d.mermaid || "").trim();
    if (!mermaid) return;
    items.push({
      kind: "diagram",
      id: String(d.id || "").trim(),
      payload: d,
    });
  });
  cards.forEach((r) => {
    const id = String(r.asset_id || "").trim();
    if (!id) return;
    items.push({ kind: "card", id, payload: r });
  });
  codes.forEach((c) => {
    const code = String(c.code || "").trim();
    if (!code || !isLikelyCodeSnippet(code)) return;
    items.push({
      kind: "code",
      id: String(c.id || "").trim(),
      payload: c,
    });
  });
  return items;
}

export const MATERIAL_VIEW_LS = "skillTreeMaterialView";
