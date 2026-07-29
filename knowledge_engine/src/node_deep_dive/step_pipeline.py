"""Шаг пайплайна: интент, обновление матрицы, сжатие rolling summary."""

from __future__ import annotations

from knowledge_engine.config import GEMINI_LITE_MODEL, GEMINI_RPM_PAUSE_SEC
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.chat_session_manager import ChatSessionManager
from knowledge_engine.services.gemini_stateless import run_gemini_structured_with_chain
from knowledge_engine.ui.run_log import trace
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    RollingCompressOutput,
    SessionMemory,
    StepAnalysisOutput,
    UserIntent,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    ACTIVE_WINDOW_MAX,
    apply_concept_updates,
    append_to_active_window,
    build_handoff_summary,
    format_evicted_for_llm,
    format_matrix_for_llm,
    format_window_for_llm,
    pop_evicted_message,
)

_STEP_ANALYSIS_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — аналитик шага Сократовского диалога по одной учебной ноде.\n"
    "На вход: матрица концептов, скользящее окно, rolling summary, сообщение пользователя.\n\n"
    "1) intent — тип сообщения:\n"
    "  ANSWER — ответ на вопрос тьютора / аргумент по теме (режим dialogue_feedback, НЕ лекция).\n"
    "  INTENT_EXPLAIN — только явный запрос теории, схемы, плотной лекции, [mode:lecture].\n"
    "  INTENT_SHIFT_FOCUS — смена ракурса внутри ноды.\n"
    "  INTENT_FINALIZE — подведение итогов, «закрой тему».\n\n"
    "Не классифицируй развёрнутый ответ пользователя на вопрос тьютора как INTENT_EXPLAIN.\n"
    "Если learning_phase=intro_assessment — intent почти всегда ANSWER, "
    "кроме [mode:lecture] / явного запроса плотной лекции.\n\n"
    "2) concept_updates — сопоставь аргументы user_message и окна с core_concepts.\n"
    "   Для доказанного понимания: status=verified, evidence=короткая цитата пользователя, "
    "mastery_score 70-100. Частичное: in_progress, 20-60. Не выдумывай концепты вне списка.\n\n"
    "3) critical_gap — только если базовый пробел в ключевом концепте (блокер), иначе null.\n"
)

_ROLLING_COMPRESS_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — саммаризатор контекста учебного диалога (одна нода skill tree).\n"
    "На вход: previous_summary, evicted_messages (вытесненные реплики), learning_phase/mode.\n\n"
    "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
    "1) Delivered vs Pending: в COVERED_POINTS — только то, что ассистент УЖЕ РЕАЛЬНО отправил "
    "(развёрнутая лекция, плотный материал, схема, длинный tutorial). "
    "Короткий вопрос тьютора или «[mode:lecture]» в сообщении пользователя — НЕ лекция.\n"
    "2) Если пользователь запросил материал ([mode:lecture], «дай лекцию», «плотный материал», "
    "«объясни подробнее»), но в evicted_messages НЕТ развёрнутого ответа ассистента с лекцией — "
    "запиши это в pending_deliverables с маркером [!]. НЕ пиши «разобрали тему», если лекции не было.\n"
    "3) Запрет на галлюцинацию завершения: не считай тему закрытой, если dense_material не выдан.\n"
    "4) next_action_for_tutor: если есть pending — закончи указанием "
    "«Сгенерируй запрошенный материал по теме X сейчас, не финальные вопросы».\n\n"
    "Заполни JSON-поля (русский):\n"
    "- current_state: режим (lecture/socratic/blitz), фаза, активный топик.\n"
    "- covered_points: буллеты — только доставленный контент и понимание пользователя.\n"
    "- pending_deliverables: невыполненные обязательства или «(нет)».\n"
    "- next_action_for_tutor: одна чёткая инструкция для следующего ответа модели.\n"
)


def _assemble_rolling_summary(out: RollingCompressOutput) -> str:
    pending = (out.pending_deliverables or "").strip() or "(нет)"
    next_act = (out.next_action_for_tutor or "").strip() or "Продолжить диалог по текущей фазе."
    return (
        "### КОНТЕКСТ ДИАЛОГА (SUMMARY)\n\n"
        "1. ТЕКУЩАЯ ТЕМА И РЕЖИМ [CURRENT_STATE]:\n"
        f"{(out.current_state or '').strip()}\n\n"
        "2. ЧТО УЖЕ ОБСУДИЛИ [COVERED_POINTS]:\n"
        f"{(out.covered_points or '').strip()}\n\n"
        "3. ДОЛГИ И НЕВЫПОЛНЕННЫЕ ЗАПРОСЫ (PENDING_ACTION):\n"
        f"{pending}\n\n"
        "[NEXT_ACTION_FOR_TUTOR]\n"
        f"{next_act}"
    )[:8000]


def _fallback_rolling_summary(
    memory: SessionMemory,
    evicted: dict[str, str],
    prev: str,
) -> str:
    chunk = format_evicted_for_llm([evicted])
    user_blob = (evicted.get("content") or "").lower()
    pending = ""
    if any(
        k in user_blob
        for k in ("[mode:lecture]", "mode:lecture", "дай лекцию", "плотн", "dense", "материал")
    ):
        pending = (
            "[!] Пользователь запросил плотный материал/лекцию; в вытесненных репликах "
            "нет развёрнутой лекции ассистента — материал ещё не выдан."
        )
    state = (
        f"- Режим: {memory.learning_mode}, фаза: {memory.learning_phase}.\n"
        f"- topic_mastery: {memory.topic_mastery_score}%."
    )
    covered = f"- Вытесненные реплики (сырой дайджест):\n{chunk[:2000]}"
    next_act = (
        "Сгенерируй запрошенный плотный материал по теме, если есть pending."
        if pending
        else "Продолжить по текущей фазе."
    )
    return _assemble_rolling_summary(
        RollingCompressOutput(
            current_state=state,
            covered_points=covered if not prev else f"{prev}\n{covered}"[:4000],
            pending_deliverables=pending or "(нет)",
            next_action_for_tutor=next_act,
        )
    )


def tutor_behavior_hint(
    intent: UserIntent,
    action: str,
    learning_mode: str = "lecture",
    learning_phase: str = "intro_assessment",
    user_message: str = "",
) -> str:
    msg = (user_message or "").strip()
    wants_lecture = intent == "INTENT_EXPLAIN" or (
        learning_mode == "lecture"
        and any(
            k in msg.lower()
            for k in (
                "дай лекцию",
                "плотный материал",
                "dense material",
                "[mode:lecture]",
                "mode:lecture",
                "объясни подроб",
                "дай плотн",
            )
        )
    )
    if learning_mode == "socratic_point" or learning_phase == "socratic_focus":
        return (
            "Точечный Сократ: ОДИН контрвопрос или edge-case. "
            "Без лекции и без списка ссылок."
        )
    if wants_lecture:
        return (
            "mode:lecture_dense — tutor_message = полная лекция (300–600 слов) "
            "с примерами, формулами и архитектурными нюансами. "
            "ЗАПРЕЩЕНО краткое резюме и заглушки «материал в панели/перед вами» без текста лекции. "
            "summary/diagram/references в JSON дополняют чат, не заменяют лекцию. "
            "Если в payload user_focus / targeted — лекция только про фокус, не обзор всей ноды."
        )
    if intent == "INTENT_SHIFT_FOCUS":
        return (
            "mode:dialogue_feedback — INTENT_SHIFT_FOCUS: смени ракурс внутри ноды. "
            "Формат: рецензия → deep dive → один провокационный вопрос. Без реферата."
        )
    if intent == "INTENT_FINALIZE":
        return (
            "INTENT_FINALIZE: подведи итог по матрице концептов и mastery_score. "
            "Если не 100% — честно назови что осталось."
        )
    if action == "verify":
        return (
            "verify: финальная проверка — оценка по матрице, без бесконечного допроса."
        )
    return (
        "mode:dialogue_feedback — на сообщение пользователя: "
        "1) рецензия его аргументов, 2) архитектурный deep dive без базовых определений, "
        "3) один сократовский вопрос в контексте. НЕ лекция, НЕ самопроверка в конце."
    )


def heuristic_step_analysis(
    user_message: str,
    learning_phase: str = "",
) -> StepAnalysisOutput:
    """Без LLM, если step_analysis недоступен (503/квота)."""
    text = (user_message or "").lower()
    phase = (learning_phase or "").strip()
    if phase == "intro_assessment":
        if not any(
            k in text
            for k in (
                "[mode:lecture]",
                "mode:lecture",
                "дай лекцию",
                "плотный материал",
                "dense material",
            )
        ):
            return StepAnalysisOutput(intent="ANSWER", concept_updates=[], critical_gap=None)
    intent: UserIntent = "ANSWER"
    if any(
        k in text
        for k in (
            "[mode:lecture]",
            "mode:lecture",
            "дай лекцию",
            "плотный материал",
            "dense material",
            "объясни подроб",
            "разжуй",
            "покажи пример",
            "пример кода",
            "дай схем",
            "нарисуй",
            "mermaid",
            "как работает",
        )
    ):
        intent = "INTENT_EXPLAIN"
    elif any(
        k in text
        for k in ("итог", "закрой тему", "заверш", "подведи", "резюме", "finalize")
    ):
        intent = "INTENT_FINALIZE"
    elif any(
        k in text
        for k in ("другой ракурс", "смени тему", "другую тему", "переключ")
    ):
        intent = "INTENT_SHIFT_FOCUS"
    return StepAnalysisOutput(intent=intent, concept_updates=[], critical_gap=None)


def run_step_analysis(
    user_message: str,
    memory: SessionMemory,
    node: NodeDataInput,
    anchor: str,
) -> StepAnalysisOutput:
    payload = (
        f"### learning_phase\n{memory.learning_phase}\n"
        f"### learning_mode\n{memory.learning_mode}\n"
        f"### node_title\n{node.title}\n"
        f"### core_concepts\n"
        + "\n".join(f"- {c}" for c in node.core_concepts)
        + f"\n\n### concepts_matrix\n{format_matrix_for_llm(memory.concepts_matrix)}\n"
        f"### rolling_summary\n{memory.rolling_dialogue_summary or '(пусто)'}\n"
    )
    chat_mgr = ChatSessionManager.from_memory_blob(anchor, memory.chat_sessions)
    handoff = build_handoff_summary(memory)
    out = run_gemini_structured_with_chain(
        GEMINI_LITE_MODEL,
        _STEP_ANALYSIS_SYSTEM,
        payload,
        anchor,
        StepAnalysisOutput,
        "node_deep_dive / step_analysis",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=chat_mgr,
        chat_label="node_deep_dive/step_analysis",
        delta_user_message=(user_message or "").strip(),
        handoff_summary=handoff,
    )
    memory.chat_sessions = chat_mgr.to_memory_blob()
    return out


def compress_rolling_summary(
    memory: SessionMemory,
    evicted: dict[str, str],
    anchor: str,
) -> str:
    payload = (
        f"### learning_phase\n{memory.learning_phase}\n"
        f"### learning_mode\n{memory.learning_mode}\n"
        f"### topic_mastery_score\n{memory.topic_mastery_score}%\n"
        f"### previous_summary\n{memory.rolling_dialogue_summary or '(пусто)'}\n"
        f"### evicted_messages\n{format_evicted_for_llm([evicted])}"
    )
    try:
        out = run_gemini_structured_with_chain(
            GEMINI_LITE_MODEL,
            _ROLLING_COMPRESS_SYSTEM,
            payload,
            anchor,
            RollingCompressOutput,
            "node_deep_dive / rolling_compress",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        )
        return _assemble_rolling_summary(out)
    except Exception as exc:
        trace(f"NODE_DIVE rolling_compress fallback | {exc}")
        prev = (memory.rolling_dialogue_summary or "").strip()
        return _fallback_rolling_summary(memory, evicted, prev)


def rotate_window_after_message(
    memory: SessionMemory,
    anchor: str,
) -> None:
    while len(memory.active_window) > ACTIVE_WINDOW_MAX:
        evicted = pop_evicted_message(memory)
        if not evicted:
            break
        memory.rolling_dialogue_summary = compress_rolling_summary(
            memory, evicted, anchor
        )


def process_user_message_pipeline(
    user_message: str,
    memory: SessionMemory,
    node: NodeDataInput,
    anchor: str,
    action: str,
) -> tuple[UserIntent, str | None]:
    """Интент, обновление матрицы; user ещё не в active_window."""
    try:
        analysis = run_step_analysis(user_message, memory, node, anchor)
    except Exception as exc:
        trace(
            f"NODE_DIVE step_analysis fallback (heuristic) | {type(exc).__name__}: {exc}"
        )
        analysis = heuristic_step_analysis(
            user_message, memory.learning_phase
        )
    apply_concept_updates(memory, analysis.concept_updates)
    intent = analysis.intent
    gap = (analysis.critical_gap or "").strip() or None
    append_to_active_window(memory, "user", user_message)
    rotate_window_after_message(memory, anchor)
    return intent, gap
