"""Single source of truth for tutor control-intent rules.

Production ``VectorIntentRouter`` (LanceDB / BGE reference phrases), exact chip
labels, ``[mode:]`` / ``[action:]`` aliases, and the offline
``lexical_probe_embed`` all read from ``INTENT_RULES``. Edit this file to add
or change an intent — do not copy cue lists into tests or the router.
"""

from __future__ import annotations

from typing import Any, NamedTuple

# Canonical Quick Reply labels (exact UI strings). No asterisk glyph in Python.
CHIP_GLOSS = "Хочу Gloss"
CHIP_HOW = "Дожать HOW"
CHIP_MECH = "Дожать MECH"
CHIP_LECTURE = "Дай плотный материал по теме"
CHIP_LECTURE_PERIOD = "Дай плотный материал по теме."
CHIP_OVERLAY_NEXT = "Идем дальше"
CHIP_ADVANCED_ANALYSIS = "Анализ уязвимостей (задачка со звёздочкой)"
CHIP_DEEP_DESIGN = "Архитектурный дизайн (сложная звёздочка)"
CHIP_DEEP_ANALYSIS_LEGACY = "Задачка со звёздочкой"
CHIP_DEEP_ANALYSIS_LEGACY_ASCII = "Задачка со звездочкой"
# Fast-track / intro mode-selection chips (exact UI strings).
CHIP_PRACTICE = "практика"
CHIP_CHECK = "проверка"
CHIP_SKIP = "пропустить"

# FSM-слот ожидания ветки после fast-track intro.
MODE_SELECTION_SLOT = "mode_selection"
MODE_SELECTION_SLOT_INTENTS: frozenset[str] = frozenset(
    {"practice", "check", "skip"}
)
MODE_SELECTION_CHIP_LABELS: tuple[str, ...] = (
    CHIP_PRACTICE,
    CHIP_CHECK,
    CHIP_SKIP,
)
# Порог BGE только внутри слота (каталог из 3 интентов, не глобальный 0.82).
MODE_SELECTION_VECTOR_THRESHOLD = 0.70


class IntentRule(NamedTuple):
    """One control intent: BGE catalog phrases, exact UI labels, optional [mode:].

    ``cues`` feed the offline probe embedder only — production routing is
    tags + exact chips + vector catalog, never substring scans.
    """

    intent: str
    cues: tuple[str, ...]  # probe-embed attractors only; not production substring routing
    reference_phrases: tuple[str, ...] = ()
    exact_labels: tuple[str, ...] = ()
    system_mode: str | None = None
    action_aliases: tuple[str, ...] = ()
    factory_modes: tuple[str, ...] = ()


# Strict evaluation order: specific overlays (L4 / L5) MUST precede generic
# ``deep_analysis`` so lexical + argmax-tie routing prefer the narrower kind.
INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        "gloss",
        cues=(
            "хочу gloss",
            "дай gloss",
            "глосс",
            "glossary",
            "выжимк",
        ),
        reference_phrases=(
            CHIP_GLOSS,
            "дай gloss",
            "хочу глосс",
            "краткий glossary по оставшимся слоям",
            "сформируй сжатую выжимку Glossary",
            "дай краткий gloss summary",
        ),
        exact_labels=(CHIP_GLOSS,),
        system_mode="[mode:gloss]",
        action_aliases=("glossary",),
        factory_modes=("gloss",),
    ),
    IntentRule(
        "how",
        cues=(
            "дожать how",
            "дожать хоу",
            "хочу how",
            "слой how",
            "deep_dive_how",
        ),
        reference_phrases=(
            CHIP_HOW,
            "хочу how",
            "слой how",
            "дожать хоу",
            "разбери архитектуру темы deeper HOW",
            "[mode:deep_dive_how] разбери архитектуру",
        ),
        exact_labels=(CHIP_HOW,),
        system_mode="[mode:deep_dive_how]",
        action_aliases=("how_deep", "deep_dive_how"),
        factory_modes=("deep_dive_how",),
    ),
    IntentRule(
        "mech",
        cues=(
            "дожать mech",
            "дожать мех",
            "хочу mech",
            "deep_dive_mech",
            "механики и код",
        ),
        reference_phrases=(
            CHIP_MECH,
            "хочу mech",
            "дожать мех",
            "разбери механики и код темы",
            "[mode:deep_dive_mech] разбери механики",
            "хочу MECHANIC слой с кодом",
        ),
        exact_labels=(CHIP_MECH,),
        system_mode="[mode:deep_dive_mech]",
        action_aliases=("mechanic", "deep_dive_mech"),
        factory_modes=("deep_dive_mech",),
    ),
    IntentRule(
        "lecture",
        cues=(
            "плотный материал",
            "дай лекцию",
            "dense material",
            "mode:lecture",
            "лекцию по",
        ),
        reference_phrases=(
            CHIP_LECTURE,
            "Дай лекцию",
            "плотный материал",
            "dense material please",
            "дай лекцию по теме",
            "[mode:lecture] дай плотный материал",
        ),
        exact_labels=(
            CHIP_LECTURE_PERIOD,
            CHIP_LECTURE,
            "Дай плотный материал",
            "Дай лекцию",
        ),
        system_mode="[mode:lecture]",
        factory_modes=("lecture",),
    ),
    IntentRule(
        "next",
        cues=(
            "идем дальше",
            "идём дальше",
            "next node",
            "следующей нод",
            "перейти дальше",
        ),
        reference_phrases=(
            CHIP_OVERLAY_NEXT,
            "идём дальше",
            "к следующей ноде",
            "next node",
            "перейти дальше",
        ),
        exact_labels=(CHIP_OVERLAY_NEXT,),
        action_aliases=("next_node",),
    ),
    IntentRule(
        "practice",
        cues=(
            "практика",
            "хочу практику",
            "глубокий кейс",
            "сразу к практике",
        ),
        reference_phrases=(
            CHIP_PRACTICE,
            "хочу практику",
            "глубокий кейс",
            "сразу к практике",
            "переключиться на практику",
        ),
        exact_labels=(CHIP_PRACTICE,),
    ),
    IntentRule(
        "check",
        cues=(
            "проверка",
            "экспресс-проверк",
            "экспресс проверка",
            "сделаем проверку",
        ),
        reference_phrases=(
            CHIP_CHECK,
            "экспресс-проверка",
            "сделаем проверку",
            "хочу проверку",
            "экспресс проверка знаний",
        ),
        exact_labels=(CHIP_CHECK,),
    ),
    IntentRule(
        "skip",
        cues=(
            "уже знаю",
            "знаю тему",
            "пропустить",
            "пропусти ноду",
            "equivalence",
        ),
        reference_phrases=(
            "уже знаю",
            "знаю тему",
            "пропустить",
            "пропусти ноду",
            "не нужно проходить",
            "equivalence skip",
        ),
        exact_labels=(
            "уже знаю",
            "знаю тему",
            CHIP_SKIP,
            "пропусти ноду",
            "уже знаю тему — пропустить",
            "уже знаю тему - пропустить",
        ),
    ),
    IntentRule(
        "begin",
        cues=(
            "[begin]",
            "начать",
            "start lesson",
            "приступим",
            "давай начнём",
        ),
        reference_phrases=(
            "начать",
            "start",
            "приступим",
            "start lesson",
            "[begin]",
            "давай начнём урок",
        ),
        exact_labels=("начать", "start", "приступим"),
    ),
    IntentRule(
        "accept_deep",
        cues=(
            "углуб",
            "погрузимся",
            "deep dive",
            "да, давай",
            "разберём глубже",
        ),
        reference_phrases=(
            "да, давай углубимся",
            "хочу углубиться",
            "давай разберём глубже",
            "deep dive пожалуйста",
            "да, хочу углубить",
            "погрузимся в детали слоя",
        ),
    ),
    IntentRule(
        "advanced_analysis",
        cues=(
            "анализ уязвимостей",
            "mode:advanced_analysis",
            "race conditions",
            "p99 latency",
            "анализ уязвимостей (задачка со звёздочкой)",
        ),
        reference_phrases=(
            CHIP_ADVANCED_ANALYSIS,
            "анализ уязвимостей",
            "race conditions и edge-cases",
            "P99 latency и корректность в экстремальных условиях",
            "[mode:advanced_analysis] анализ уязвимостей",
            "разбери уязвимости и race conditions",
        ),
        exact_labels=(CHIP_ADVANCED_ANALYSIS,),
        system_mode="[mode:advanced_analysis]",
        factory_modes=("advanced_analysis",),
    ),
    IntentRule(
        "deep_design",
        cues=(
            "архитектурный дизайн",
            "mode:deep_design",
            "сложная звёздочка",
            "спроектируй систему",
            "архитектурный дизайн (сложная звёздочка)",
        ),
        reference_phrases=(
            CHIP_DEEP_DESIGN,
            "архитектурный дизайн",
            "спроектируй систему с нуля",
            "обоснование trade-offs и ключевых решений",
            "[mode:deep_design] архитектурный дизайн",
            "сложная звёздочка системный дизайн",
        ),
        exact_labels=(CHIP_DEEP_DESIGN,),
        system_mode="[mode:deep_design]",
        factory_modes=("deep_design",),
    ),
    IntentRule(
        "deep_analysis",
        cues=(
            "задачка со звёздочкой",
            "задачка со звездочкой",
            "mode:deep_analysis",
            "design challenge",
        ),
        reference_phrases=(
            CHIP_DEEP_ANALYSIS_LEGACY,
            CHIP_DEEP_ANALYSIS_LEGACY_ASCII,
            "deep analysis",
            "инженерная задача на проектирование",
            "разбери trade-offs и дай design challenge",
            "[mode:deep_analysis] дай задачу со звёздочкой",
        ),
        exact_labels=(
            CHIP_DEEP_ANALYSIS_LEGACY,
            CHIP_DEEP_ANALYSIS_LEGACY_ASCII,
            "Deep Analysis",
        ),
        system_mode="[mode:deep_analysis]",
        action_aliases=("deepanalysis", "star_challenge"),
        factory_modes=("deep_analysis",),
    ),
)

INTENT_NAMES: tuple[str, ...] = tuple(rule.intent for rule in INTENT_RULES)

# Whole-message lowercase stubs used by the offline probe when no cue matched.
PROBE_WHOLE_MESSAGE_FALLBACK: dict[str, str] = {
    "начать": "begin",
    "start": "begin",
    "приступим": "begin",
    "go": "begin",
    "давай": "begin",
    "да": "accept_deep",
    "углубиться": "accept_deep",
    CHIP_PRACTICE: "practice",
    CHIP_CHECK: "check",
    CHIP_SKIP: "skip",
}

# Quick-reply chips that must not be scored by the gap evaluator.
EVALUATOR_SKIP_INTENTS: frozenset[str] = frozenset(
    {
        "gloss",
        "how",
        "mech",
        "deep_analysis",
        "advanced_analysis",
        "deep_design",
        "next",
        "practice",
        "check",
        "skip",
        "begin",
        "lecture",
    }
)


def _dedupe_phrases(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for raw in group:
            phrase = (raw or "").strip()
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
    return tuple(out)


def probe_cues(rule: IntentRule) -> tuple[str, ...]:
    """Lowercased substring markers for ``lexical_probe_embed``."""
    extra: list[str] = []
    if rule.system_mode:
        extra.append(rule.system_mode)
        extra.append(rule.system_mode.strip("[]"))
    merged = _dedupe_phrases(rule.cues, tuple(extra))
    return tuple(p.lower() for p in merged)


def catalog_phrases(rule: IntentRule) -> tuple[str, ...]:
    """Phrases persisted as LanceDB vectors (exact chips + BGE paraphrases)."""
    return _dedupe_phrases(rule.exact_labels, rule.reference_phrases)


INTENT_REFERENCE_PHRASES: dict[str, tuple[str, ...]] = {
    rule.intent: catalog_phrases(rule) for rule in INTENT_RULES
}

REGISTERED_CONTROL_CHIPS: dict[str, str] = {}
for _rule in INTENT_RULES:
    for _label in _rule.exact_labels:
        REGISTERED_CONTROL_CHIPS[_label] = _rule.intent

ACTION_ALIASES: dict[str, str] = {}
FACTORY_MODE_TO_INTENT: dict[str, str] = {}
for _rule in INTENT_RULES:
    ACTION_ALIASES[_rule.intent] = _rule.intent
    for _alias in _rule.action_aliases:
        ACTION_ALIASES[_alias.lower().replace("-", "_")] = _rule.intent
    if _rule.system_mode:
        _token = _rule.system_mode.strip("[]").split(":")[-1].lower()
        if _token:
            ACTION_ALIASES[_token] = _rule.intent
    for _mode in _rule.factory_modes:
        FACTORY_MODE_TO_INTENT[_mode.lower()] = _rule.intent


def validate_intent_catalog() -> dict[str, Any]:
    """
    Integrity check for the SSOT before LanceDB sync / probe use.

    Raises ``ValueError`` when the registry is empty, duplicated, or overlay
    order is inverted (generic ``deep_analysis`` before specific L4/L5).
    """
    if not INTENT_RULES:
        raise ValueError("INTENT_RULES is empty")
    names = [rule.intent for rule in INTENT_RULES]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate intent in INTENT_RULES: {names}")
    index = {name: i for i, name in enumerate(names)}
    if "advanced_analysis" in index and "deep_analysis" in index:
        if index["advanced_analysis"] >= index["deep_analysis"]:
            raise ValueError(
                "advanced_analysis must precede deep_analysis in INTENT_RULES"
            )
    if "deep_design" in index and "deep_analysis" in index:
        if index["deep_design"] >= index["deep_analysis"]:
            raise ValueError(
                "deep_design must precede deep_analysis in INTENT_RULES"
            )
    phrase_count = 0
    for rule in INTENT_RULES:
        if not rule.cues:
            raise ValueError(f"intent {rule.intent!r} has empty cues")
        phrases = catalog_phrases(rule)
        if not phrases:
            raise ValueError(f"intent {rule.intent!r} has empty catalog phrases")
        phrase_count += len(phrases)
        for label in rule.exact_labels:
            if label not in phrases:
                raise ValueError(
                    f"exact label {label!r} missing from catalog of {rule.intent!r}"
                )
    return {
        "ok": True,
        "intents": len(INTENT_RULES),
        "phrases": phrase_count,
        "overlay_order": [
            name
            for name in ("advanced_analysis", "deep_design", "deep_analysis")
            if name in index
        ],
    }
