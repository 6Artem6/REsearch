"""Галочки по всем блокам контекста (по умолчанию 7B) + сборка payload."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import CONTEXT_EVAL_MODEL, CONTEXT_EVAL_NUM_PREDICT
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE, RUSSIAN_ROUTER_RULE
from knowledge_engine.schemas import (
    ContextBlock,
    ContextBlocksEvaluation,
    EngineGraphState,
    EngineState,
)
from knowledge_engine.services.context_blocks import (
    apply_hard_hint_exclusions,
    assemble_gemini_payload,
    blocks_from_state_dicts,
    build_context_blocks,
    catalog_for_evaluator,
    default_selections,
)
from knowledge_engine.ui.errors import format_error_with_cause
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start, trace


def _merge_selections(
    blocks: list[ContextBlock],
    evaluation: ContextBlocksEvaluation,
) -> dict[str, bool]:
    blocks_by_id = {b.block_id: b for b in blocks}
    base = default_selections(blocks)
    seen: set[str] = set()
    for d in evaluation.decisions:
        if d.block_id not in base:
            continue
        seen.add(d.block_id)
        block = blocks_by_id.get(d.block_id)
        if block and block.always_include:
            base[d.block_id] = True
        else:
            base[d.block_id] = d.include

    missing = [
        b.block_id for b in blocks if b.block_id not in seen and not b.always_include
    ]
    if missing:
        trace(
            f"CONTEXT evaluator: нет галочек для {len(missing)} блоков — default_include/hints"
        )
    return apply_hard_hint_exclusions(blocks, base)


def _evaluate_blocks(
    parsed: EngineState,
    blocks: list[ContextBlock],
) -> ContextBlocksEvaluation:
    rule = (
        RUSSIAN_ROUTER_RULE
        if "1.5b" in CONTEXT_EVAL_MODEL.lower()
        else RUSSIAN_OUTPUT_RULE
    )
    structured = structured_chat(
        CONTEXT_EVAL_MODEL,
        ContextBlocksEvaluation,
        temperature=0.05,
        num_predict=CONTEXT_EVAL_NUM_PREDICT,
    )
    catalog = catalog_for_evaluator(blocks)
    block_ids = [b.block_id for b in blocks]
    system = SystemMessage(
        content=(
            f"{rule} "
            "Context Quality Evaluator: для КАЖДОГО block_id — include (галочка).\n"
            "Один decision на id, без пропусков. hints с default_include=false → include=false.\n\n"
            "• profile: HW/RAM/LanceDB + критерии отбора; выключи курсы, e-commerce, проекты если не про задачу.\n"
            "• source/fact/abstraction: выключи мусор, дубли, «Классика (LRU…)» strawman, учебниковую воду.\n"
            "• system: короткая роль — обычно include=true.\n"
            "• user_task: always_include=true.\n"
            "Не переписывай текст блоков."
        )
    )
    human = HumanMessage(
        content=(
            f"Задача: {parsed.user_problem}\n"
            f"Ограничения: {parsed.context_constraints or '(не указаны)'}\n"
            f"Нужно {len(block_ids)} decisions.\n"
            f"block_ids: {', '.join(block_ids)}\n\n"
            f"КАТАЛОГ:\n{catalog}"
        )
    )
    label = f"context_evaluator / {CONTEXT_EVAL_MODEL} ContextBlocksEvaluation"
    result = invoke_logged(structured, [system, human], label)
    if result is None:
        return ContextBlocksEvaluation(
            decisions=[],
            is_context_optimal=False,
            rationale="Оценка не удалась — только эвристики hints",
        )
    return result


def _blocks_for_state(parsed: EngineState) -> list[ContextBlock]:
    if parsed.context_blocks:
        return blocks_from_state_dicts(parsed.context_blocks)
    return build_context_blocks(parsed)


def evaluate_and_refine_context_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("evaluate_and_refine_context_node (block ticks)")
    parsed = EngineState.model_validate(state)
    blocks = _blocks_for_state(parsed)

    if not blocks:
        node_end("evaluate_and_refine_context_node", "no blocks")
        return {"is_ready_for_gemini": True}

    if parsed.context_corrected_once and parsed.context_block_selections:
        set_status("[context_evaluator] сборка из сохранённых галочек…")
        selections = apply_hard_hint_exclusions(
            blocks, dict(parsed.context_block_selections)
        )
        payload = assemble_gemini_payload(blocks, selections)
        included = sum(
            1 for b in blocks if selections.get(b.block_id, b.default_include)
        )
        trace(f"CONTEXT assemble (cached) | blocks={included}/{len(blocks)}")
        node_end("evaluate_and_refine_context_node", f"cached → {len(payload)} sym")
        return {
            "gemini_payload": payload,
            "context_block_selections": selections,
            "is_ready_for_gemini": True,
        }

    kinds = sorted({b.kind for b in blocks})
    set_status(
        f"[context_evaluator] {CONTEXT_EVAL_MODEL}: галочки "
        f"{len(blocks)} блоков ({', '.join(kinds)})…"
    )
    try:
        evaluation = _evaluate_blocks(parsed, blocks)
    except Exception as exc:
        raise RuntimeError(
            f"Оценка блоков не удалась ({CONTEXT_EVAL_MODEL}): "
            f"{format_error_with_cause(exc)}"
        ) from exc

    selections = _merge_selections(blocks, evaluation)
    payload = assemble_gemini_payload(blocks, selections)
    included_ids = [
        b.block_id for b in blocks if selections.get(b.block_id, b.default_include)
    ]
    trace(
        f"CONTEXT ticks | optimal={evaluation.is_context_optimal} | "
        f"included={len(included_ids)}/{len(blocks)} | {evaluation.rationale[:120]}"
    )
    for d in evaluation.decisions:
        if not d.include:
            trace(f"  skip {d.block_id}: {d.reason[:80]}")

    node_end(
        "evaluate_and_refine_context_node",
        f"ticks {len(included_ids)}/{len(blocks)} → {len(payload)} sym",
    )
    return {
        "context_block_selections": selections,
        "gemini_payload": payload,
        "context_corrected_once": True,
        "is_context_optimal": evaluation.is_context_optimal,
        "is_ready_for_gemini": True,
    }
