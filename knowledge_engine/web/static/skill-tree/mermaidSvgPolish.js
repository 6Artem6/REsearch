const FLOWCHART_BOX_PAD_W = 22;
const FLOWCHART_BOX_PAD_H = 6;

function expandFlowchartNodeBoxes(svg) {
  svg.querySelectorAll("g.node").forEach((node) => {
    const fo = node.querySelector("foreignObject");
    if (!fo) return;
    const extraW = FLOWCHART_BOX_PAD_W;
    const extraH = FLOWCHART_BOX_PAD_H;
    const w = Number(fo.getAttribute("width") || 0);
    const h = Number(fo.getAttribute("height") || 0);
    const x = Number(fo.getAttribute("x") || 0);
    const y = Number(fo.getAttribute("y") || 0);
    if (w > 0) {
      fo.setAttribute("width", String(w + extraW));
      fo.setAttribute("x", String(x - extraW / 2));
    }
    if (h > 0) {
      fo.setAttribute("height", String(h + extraH));
      fo.setAttribute("y", String(y - extraH / 2));
    }
    const rect = node.querySelector("rect");
    if (rect) {
      const rw = Number(rect.getAttribute("width") || 0);
      const rh = Number(rect.getAttribute("height") || 0);
      const rx = Number(rect.getAttribute("x") || 0);
      const ry = Number(rect.getAttribute("y") || 0);
      if (rw > 0) {
        rect.setAttribute("width", String(rw + extraW));
        rect.setAttribute("x", String(rx - extraW / 2));
      }
      if (rh > 0) {
        rect.setAttribute("height", String(rh + extraH));
        rect.setAttribute("y", String(ry - extraH / 2));
      }
    }
    const poly = node.querySelector("polygon");
    if (poly) {
      const pts = (poly.getAttribute("points") || "")
        .trim()
        .split(/\s+/)
        .map((p) => p.split(",").map(Number));
      if (pts.length >= 4) {
        const xs = pts.map((p) => p[0]);
        const ys = pts.map((p) => p[1]);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const scaleX = (maxX - minX + extraW) / (maxX - minX);
        const scaleY = (maxY - minY + extraH) / (maxY - minY);
        const scaled = pts.map(([px, py]) => {
          const nx = cx + (px - cx) * scaleX;
          const ny = cy + (py - cy) * scaleY;
          return `${nx},${ny}`;
        });
        poly.setAttribute("points", scaled.join(" "));
      }
    }
  });
}

/** Полировка SVG: чуть шире узлы flowchart (htmlLabels), без правок layout div. */
export function polishMermaidSvg(svg) {
  if (!svg) return;
  expandFlowchartNodeBoxes(svg);
  svg.querySelectorAll(".edgeLabel").forEach((el) => {
    el.setAttribute("font-size", "13");
  });
  svg.querySelectorAll(".node rect, .node polygon, .node path").forEach((el) => {
    el.style.strokeWidth = "1.5px";
  });
}

export function applyDiagramSvgScale(svg, scale) {
  if (!svg) return;
  svg.style.transformOrigin = "0 0";
  if (!scale || scale === 1) {
    svg.style.transform = "";
  } else {
    svg.style.transform = `scale(${scale})`;
  }
}
