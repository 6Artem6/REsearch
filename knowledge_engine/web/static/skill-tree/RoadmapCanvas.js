import React, { useMemo, useEffect, useCallback } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { skillNodeTypes } from "./SkillNode.js";
import { curriculumToFlow, layoutFlowNodes } from "./layout.js";

const FIT_PADDING = 0.22;
const EXTENT_PAD = 520;

function translateExtentFromNodes(nodes, getNodesBounds) {
  if (!nodes?.length || typeof getNodesBounds !== "function") {
    return [
      [-2000, -2000],
      [2000, 2000],
    ];
  }
  const b = getNodesBounds(nodes);
  return [
    [b.x - EXTENT_PAD, b.y - EXTENT_PAD],
    [b.x + b.width + EXTENT_PAD, b.y + b.height + EXTENT_PAD],
  ];
}

function RoadmapFlowInner({
  curriculum,
  statuses,
  selectedNodeId,
  onNodeClick,
  tutorBusyNodeId,
  sessions,
}) {
  const { fitView, getNodesBounds } = useReactFlow();
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => curriculumToFlow(curriculum, statuses, selectedNodeId),
    [curriculum, statuses, selectedNodeId],
  );

  const laidOut = useMemo(
    () => layoutFlowNodes(curriculum?.nodes || [], initialNodes, initialEdges),
    [curriculum, initialNodes, initialEdges],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(laidOut);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const translateExtent = useMemo(
    () => translateExtentFromNodes(nodes, getNodesBounds),
    [nodes, getNodesBounds],
  );

  const fitMapToView = useCallback(() => {
    fitView({ padding: FIT_PADDING, duration: 280, maxZoom: 1.15 });
  }, [fitView]);

  useEffect(() => {
    const { nodes: n, edges: e } = curriculumToFlow(
      curriculum,
      statuses,
      selectedNodeId,
    );
    const positioned = layoutFlowNodes(curriculum?.nodes || [], n, e);
    setNodes(positioned);
    setEdges(e);
  }, [curriculum, statuses, selectedNodeId, setNodes, setEdges]);

  useEffect(() => {
    const t = window.setTimeout(() => fitMapToView(), 50);
    return () => window.clearTimeout(t);
  }, [curriculum?.curriculum_id, nodes.length, fitMapToView]);

  return React.createElement(
    "div",
    { className: "skill-flow-root" },
    React.createElement(
      "div",
      { className: "skill-flow-toolbar" },
      React.createElement(
        "button",
        {
          type: "button",
          className: "skill-map-fit-btn",
          onClick: fitMapToView,
          title: "Показать всю карту на экране",
        },
        "На экран",
      ),
    ),
    React.createElement(
      ReactFlow,
      {
        nodes,
        edges,
        onNodesChange,
        onEdgesChange,
        nodeTypes: skillNodeTypes,
        fitView: true,
        fitViewOptions: { padding: FIT_PADDING, maxZoom: 1.15 },
        minZoom: 0.35,
        maxZoom: 1.5,
        translateExtent,
        panOnScroll: true,
        zoomOnScroll: true,
        zoomOnPinch: true,
        preventScrolling: false,
        onNodeClick: (_, node) => {
          const raw = node.data.raw;
          const nid = raw?.node_id;
          const initialized = Boolean(sessions?.[nid]?.initialized);
          if (tutorBusyNodeId !== null && !initialized) return;
          onNodeClick(raw);
        },
      },
      React.createElement(Background, { gap: 16, size: 1 }),
      React.createElement(Controls, {
        showFitView: true,
        fitViewOptions: { padding: FIT_PADDING, maxZoom: 1.15 },
      }),
      React.createElement(MiniMap, {
        zoomable: true,
        pannable: true,
        className: "skill-flow-minimap",
      }),
    ),
  );
}

export function RoadmapCanvas(props) {
  return React.createElement(
    "div",
    { className: "skill-canvas-wrap" },
    React.createElement(
      ReactFlowProvider,
      null,
      React.createElement(RoadmapFlowInner, props),
    ),
  );
}
