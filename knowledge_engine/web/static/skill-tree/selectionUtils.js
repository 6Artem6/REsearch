export function getSurroundingParagraph(node, selectedText) {
  let el = node;
  while (el && el !== document.body) {
    if (el.matches && el.matches("p, li, blockquote, h3, h4, td")) {
      return (el.textContent || "").trim();
    }
    el = el.parentElement;
  }
  const block = node?.parentElement?.closest(
    ".node-selectable-material, .tutor-selectable, .md-body",
  );
  const text = (block?.textContent || "").trim();
  if (!text) return selectedText;
  const idx = text.indexOf(selectedText);
  if (idx < 0) return text.slice(0, 1200);
  const start = Math.max(0, idx - 400);
  const end = Math.min(text.length, idx + selectedText.length + 400);
  return text.slice(start, end);
}
