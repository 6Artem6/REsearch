"""
Ordered prompt blocks for Dense / Explain (static prefix → dynamic tail, cache-friendly).

RU: заголовки блоков для Gemini prompt cache; citation/layout — общие для dense и explain.
"""

from __future__ import annotations

BLOCK_STATIC_PRESET_HEADER = (
    "======================================================================\n"
    "[BLOCK 1: STATIC SYSTEM PRESET - КЭШИРУЕТСЯ 100%]\n"
    "======================================================================"
)
"""
RU (пояснение): BLOCK 1 — статический system preset (100% cache).
"""

BLOCK_SEMI_STATIC_HEADER = (
    "======================================================================\n"
    "[BLOCK 2: SEMI-STATIC NODE CONTEXT - КЭШИРУЕТСЯ В ПРЕДЕЛАХ НОДЫ]\n"
    "======================================================================"
)
"""
RU (пояснение): BLOCK 2 — контекст ноды (cache в рамках ноды).
"""

BLOCK_DYNAMIC_HEADER = (
    "======================================================================\n"
    "[BLOCK 3: DYNAMIC SESSION DATA - НЕ КЭШИРУЕТСЯ / ИЗМЕНЯЕТСЯ КАЖДЫЙ ХОД]\n"
    "======================================================================"
)
"""
RU (пояснение): BLOCK 3 — динамика сессии (не кэшируется).
"""

BLOCK_SEMI_STATIC_TAG = "[BLOCK 2: SEMI-STATIC NODE CONTEXT"
"""
RU (пояснение): якорь начала semi-static payload (cache tests / split).
"""

BLOCK_DYNAMIC_TAG = "[BLOCK 3: DYNAMIC SESSION DATA"
"""
RU (пояснение): якорь начала dynamic payload.
"""

BLOCK_RAG_TAG = "[RAG_AND_NODE_SOURCES]"
"""
RU (пояснение): тег секции RAG в user payload.
"""

BLOCK_USER_QUERY_TAG = "[CURRENT_USER_QUERY]"
"""
RU (пояснение): тег текущего запроса пользователя в payload.
"""

PINNED_REGISTRY_TAG = "[PINNED SOURCE REGISTRY]"
"""
RU (пояснение): закреплённый SOURCE REGISTRY в payload.
"""

PROMPT_CITATION_ID_RULES = (
    "=== CITATION ID RULES (OUTPUT ONLY REAL SOURCE IDS) ===\n"
    "- FORBIDDEN: cite payload section meta-tags ([SHARED_SESSION_CONTEXT], [context_paragraph], "
    "[RAG_AND_NODE_SOURCES], [BLOCK 1–3], fact_manifest as [MANIFEST], field names).\n"
    "- ALLOWED only real IDs: [R1], [R2], … (RAG chunks), [S1], [S2], … (whitelist/registry), "
    "[Diagram N] from DIAGRAM_CATALOG.\n"
    "- Tie lecture paragraphs to embedded [R*] and node-level [S*]; do not swap IDs or add decorative footnotes.\n"
)
"""
RU (пояснение): в output только [R*]/[S*]/[Diagram N], не мета-теги секций.
"""

LAYOUT_AND_TYPOGRAPHY_RULES = (
    "=== LAYOUT & TYPOGRAPHY (UX / READABILITY) ===\n"
    "Text must scan quickly in narrow web panels and on mobile.\n\n"
    "1. TABLE WIDTH (CRITICAL):\n"
    "   - NEVER use markdown tables with more than 3 columns. Wide tables break layout.\n"
    "   - For matrices or multi-axis data use one of:\n"
    "     a) ENTITY CARDS: vertical list with the same bold field labels per item.\n"
    '     b) TRANSPOSED TABLE (max 3 columns): "| Parameter | Option A | Option B |".\n'
    '     c) CRITERION GROUPS: split comparison by dimension ("Token cost", "Complexity").\n\n'
    "2. PARAGRAPHS & RHYTHM:\n"
    "   - No walls of text; cap paragraphs at 3–4 sentences.\n"
    "   - Put the main point first or bold the takeaway at the start of a paragraph.\n"
    "   - Use bullet lists to separate complex concepts.\n\n"
    "3. EMPHASIS & INLINE FORMATTING:\n"
    "   - Use **bold** for key terms, metrics, pattern names, and quantified facts "
    "(e.g. **−67% tokens**, **Stateless**).\n"
    "   - Wrap variables, methods, software terms, IDs, and CLI in inline code "
    "(`checkpointer`, `state_graph`, `[R6]`).\n\n"
    "4. LISTS:\n"
    "   - Enumerations of 3+ items MUST use bullets (`-`) or numbered lists (`1.`).\n"
    '   - Inside list items use "**Label:** description" '
    '(e.g. "- **Isolation:** The subagent does not accumulate stale state…").\n\n'
    "5. CODE BLOCKS & MATH:\n"
    "   - Multi-line code MUST use fenced blocks with an explicit language tag "
    "(`python`, `bash`, `json`, `yaml`).\n"
    "   - Use standard LaTeX for formulas: inline `$x_i$` or display `$$...$$`.\n"
)
"""
RU (пояснение): UX верстка — таблицы ≤3 колонок, списки, bold/code, fenced code.
"""


def prompt_block_header(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}\n"


def semi_static_user_prefix(payload: str) -> str:
    """Байт-идентичный префикс user payload до динамического блока (для cache tests)."""
    text = payload or ""
    idx = text.find(BLOCK_DYNAMIC_TAG)
    if idx < 0:
        return text
    return text[:idx]
