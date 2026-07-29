import React from "react";

export function ColumnResizer({ onDragDelta, onDragEnd }) {
  function onMouseDown(e) {
    e.preventDefault();
    const startX = e.clientX;
    let lastX = startX;
    const onMove = (ev) => {
      const dx = ev.clientX - lastX;
      lastX = ev.clientX;
      if (dx) onDragDelta(dx);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.classList.remove("skill-col-resizing");
      onDragEnd?.();
    };
    document.body.classList.add("skill-col-resizing");
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  return React.createElement("div", {
    className: "skill-col-resizer",
    role: "separator",
    "aria-orientation": "vertical",
    onMouseDown,
  });
}
