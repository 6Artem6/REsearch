import React from "react";
import { createRoot } from "react-dom/client";
import { RoadmapDashboard } from "./RoadmapDashboard.js";

const root = createRoot(document.getElementById("skill-tree-root"));
root.render(React.createElement(RoadmapDashboard));
