/** Литералы \\n из structured LLM → реальные переносы (с повтором при двойном escape). */
export function repairLlMText(text) {
  if (!text) return "";
  let t = String(text);
  for (let i = 0; i < 6; i += 1) {
    const next = t
      .replace(/\\r\\n/g, "\n")
      .replace(/\\r/g, "\n")
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t");
    if (next === t) break;
    t = next;
  }
  return t;
}
