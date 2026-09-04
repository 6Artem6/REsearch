"""Isolated system prompt: Deep Analysis — legacy alias of Deep Design (L5/L6).

``[mode:deep_analysis]`` remains a back-compat alias of ``[mode:deep_design]``.
Bloom L4 vulnerability analysis lives in ``advanced_analysis_prompt.py``.
"""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.deep_design_prompt import DEEP_DESIGN_PROMPT

DEEP_ANALYSIS_PROMPT = DEEP_DESIGN_PROMPT
"""
RU (пояснение): [mode:deep_analysis] — алиас [mode:deep_design] (L5/L6);
code-1 не пересказывать; follow_up обязателен; флаги оркестрации — хост.
"""
