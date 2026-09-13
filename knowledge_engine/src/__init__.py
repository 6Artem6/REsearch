"""Пакетный корень для всего кода knowledge_engine/src (DDA-дерево).

Раньше здесь были eager-реэкспорты v0.7 research-pipeline (analytics/dedup/
fetcher/graph/guardrails/locks/state) — но ни один вызывающий код в проекте
не импортирует их через `from knowledge_engine.src import X` (все идут по
полному пути вида `knowledge_engine.src.graph.compile_v07_graph`), поэтому
эти реэкспорты были мёртвым весом. При этом любой импорт ЛЮБОГО модуля под
`knowledge_engine.src.*` (включая новые src/config, src/core, src/domains и
т.д.) заставлял Python сначала выполнить этот файл — то есть транзитивно
тянул весь v0.7-пайплайн и его тяжёлые зависимости (LangGraph и т.д.) туда,
где они не нужны и не установлены (например, минимальный Docker-образ
migrator). Реэкспорты убраны как часть переноса config.py под src/config/ —
поведение существующих вызывающих мест не меняется, т.к. они уже используют
полные пути.
"""

from __future__ import annotations
