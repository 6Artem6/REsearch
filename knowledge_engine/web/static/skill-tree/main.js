import React from "react";
import { createRoot } from "react-dom/client";
import { RoadmapDashboard } from "./RoadmapDashboard.js";
import { SteeringStudio } from "./SteeringStudio.js";
import { TopicFocusView } from "./TopicFocusView.js";

const root = createRoot(document.getElementById("skill-tree-root"));

// Standalone-режимы (Штурвал Studio / фокусное чтение темы) — без визуального
// шума канваса Skill Tree, см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md.
// Читается один раз при монтировании; дальнейшая навигация внутри каждого
// режима — обычный локальный useState, как и во всём остальном UI (см. аудит
// фронтенда: намеренно без сторонней роутинг-библиотеки).
const params = new URLSearchParams(window.location.search);
const mode = params.get("mode");

let app;
if (mode === "studio") {
  app = React.createElement(SteeringStudio);
} else if (mode === "topic") {
  app = React.createElement(TopicFocusView, {
    nodeId: params.get("id"),
    curriculumId: params.get("curriculum"),
  });
} else {
  app = React.createElement(RoadmapDashboard);
}

root.render(app);
