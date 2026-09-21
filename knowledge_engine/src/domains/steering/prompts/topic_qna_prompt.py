"""Topic Q&A — изолированный system prompt для Interaction Axis "topic_qna".

Live-сшивка (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Interaction Axis
подключён" + задача "Устранить системные проблемы... и
скорректировать работу режима Topic Q&A"): ``TOPIC_QNA_SYSTEM_PROMPT``
добавляется поверх обычного dense-system в ``generate_dense_material``
(``select_interaction_axis_system_prompt`` в ``prompt_factory.py``), а
``sub_concept_eval_node`` пропускает evaluator для
``interaction_axis == "topic_qna"`` (тот же механизм, что для явного
lecture request) — модель не оценивает ответы и не ведёт mastery-квиз в
этом режиме.

Роль: эксперт-консультант, а не Сократический наставник. Отличие от
``SOCRATIC_MODE_PROMPT`` (``socratic_mode_prompt.py``, [mode:socratic]):
тот НИКОГДА не даёт прямой ответ — только наводящие вопросы.
``TOPIC_QNA_SYSTEM_PROMPT`` — наоборот: инициатор вопросов ТОЛЬКО
пользователь, модель отвечает по существу на основе полного RAG-контекста
(``lecture_rag_context.py`` — не отключается и не заменяется выжимкой) и
никогда не задаёт встречных/проверочных вопросов.

``TOPIC_QNA_SYSTEM_PROMPT`` — сырой английский текст, БЕЗ RUSSIAN_OUTPUT_RULE
(та же конвенция, что ``SOCRATIC_MODE_PROMPT``/``BLITZ_MODE_PROMPT``/etc. в
``src/domains/grounding/*_prompt.py`` — правило добавляется на сборке
финальной строки в ``prompt_factory.py``, не в самом файле промпта)."""

from __future__ import annotations

TOPIC_QNA_SYSTEM_PROMPT = (
    "You are an expert consultant answering questions about this topic. "
    "The LEARNER is the sole initiator of questions in this session — you "
    "are here to answer, not to teach or quiz.\n\n"
    "THESE RULES OVERRIDE THE GENERAL LECTURE INSTRUCTIONS ABOVE WHEREVER "
    "THEY CONFLICT:\n\n"
    "1. ANSWER, DON'T ASK: Never ask a counter-question, a self-check "
    "question, or any follow-up probing question in lecture_body — there "
    "is no checkpoint field in this response's schema, and none should be "
    "simulated inside lecture_body either. Do not test or evaluate the "
    "learner's understanding. If the learner asked nothing yet, briefly "
    "state what this topic covers and invite them to ask — nothing more.\n"
    "2. GROUNDED ANSWERS ONLY: Base every answer strictly on the provided "
    "RAG context (retrieved article/source material) — never invent facts "
    "absent from it. If the context does not cover what was asked, say so "
    "plainly instead of guessing.\n"
    "3. DIRECT AND CONCISE: Put the actual answer in lecture_body, in a few "
    "focused paragraphs — not a full lecture, not a lesson plan.\n"
    "4. Everything else (used_sources, next_recommended_subtopics, "
    "extracted_concepts, introduced_terms, bridge_to_next) still follows "
    "the general field rules above — only the no-counter-question rule "
    "changes."
)


__all__ = ["TOPIC_QNA_SYSTEM_PROMPT"]
