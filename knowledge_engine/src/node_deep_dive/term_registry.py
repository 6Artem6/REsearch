"""Реестр уже расшифрованных терминов (Term Registry) — anti-redundancy в диалоге."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

INTRODUCED_TERMS_MAX = 64
_TERM_MAX_LEN = 80
_TERM_RE = re.compile(r"[\wа-яёА-ЯЁ][\wа-яёА-ЯЁ\-\./]*", re.UNICODE)


def _canonical_term_key(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def normalize_introduced_terms(raw: Any) -> list[str]:
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, (set, frozenset)):
        items = list(raw)
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        t = raw.strip()
        if not t:
            return []
        if t.startswith("["):
            try:
                parsed = json.loads(t)
                items = parsed if isinstance(parsed, list) else [t]
            except Exception:
                items = [t]
        else:
            items = [t]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = str(item or "").strip()
        if not label or len(label) > _TERM_MAX_LEN:
            continue
        key = _canonical_term_key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= INTRODUCED_TERMS_MAX:
            break
    return out


def merge_introduced_terms(
    memory: SessionMemory,
    new_terms: list[str] | None,
    *,
    persist: Callable[[], None] | None = None,
) -> None:
    """Дополнить memory.introduced_terms после ответа тьютора / плотной лекции."""
    incoming = normalize_introduced_terms(new_terms or [])
    if not incoming:
        return
    current = normalize_introduced_terms(memory.introduced_terms or [])
    seen = {_canonical_term_key(t) for t in current}
    for term in incoming:
        key = _canonical_term_key(term)
        if key in seen:
            continue
        seen.add(key)
        current.append(term)
        if len(current) >= INTRODUCED_TERMS_MAX:
            break
    memory.introduced_terms = current
    if persist is not None:
        persist()


def format_already_explained_terms_block(memory: SessionMemory) -> str:
    """
    RU (пояснение): reusable prompt-rule блок (не просто данные) — попадает в
    DYNAMIC_SUFFIX диалогового payload'а тьютора, поэтому текст правила
    английский по конвенции проекта (`.cursor/rules/llm-system-prompts-
    english.mdc`); сам список терминов (ALREADY_EXPLAINED_TERMS) остаётся
    как есть — это данные сессии, не инструкция.
    """
    terms = normalize_introduced_terms(memory.introduced_terms or [])
    if not terms:
        return ""
    encoded = json.dumps(terms, ensure_ascii=False)
    return (
        "### ALREADY_EXPLAINED_TERMS (Term Registry — persistent, not reset on compact)\n"
        f"ALREADY_EXPLAINED_TERMS: {encoded}\n\n"
        "Anti-Redundancy Rule:\n"
        "FORBIDDEN to re-decode, re-translate, or give an intro explanation "
        "('in plain terms') for terms already in ALREADY_EXPLAINED_TERMS.\n"
        "Use them as known, already-mastered concepts.\n"
        "Decode and introduce ONLY genuinely NEW terms not yet in the list.\n"
        "In the JSON response, fill the `introduced_terms` field only with "
        "terms decoded for the first time in this reply."
    )


def carry_introduced_terms(
    preserved: list[str] | None,
    memory: SessionMemory,
) -> None:
    prev = normalize_introduced_terms(preserved or [])
    if not prev:
        return
    current = normalize_introduced_terms(memory.introduced_terms or [])
    merged = list(prev)
    seen = {_canonical_term_key(t) for t in merged}
    for term in current:
        key = _canonical_term_key(term)
        if key in seen:
            continue
        seen.add(key)
        merged.append(term)
    memory.introduced_terms = merged[:INTRODUCED_TERMS_MAX]


def load_introduced_terms_from_blob(blob: dict[str, Any]) -> list[str]:
    top = normalize_introduced_terms(blob.get("introduced_terms"))
    from knowledge_engine.src.node_deep_dive.tiered_memory import memory_from_blob

    mem = memory_from_blob(blob.get("memory"))
    from_mem = normalize_introduced_terms(
        (mem.introduced_terms if mem is not None else None) or []
    )
    if not from_mem:
        return top
    if not top:
        return from_mem
    merged = list(from_mem)
    seen = {_canonical_term_key(t) for t in merged}
    for term in top:
        key = _canonical_term_key(term)
        if key in seen:
            continue
        seen.add(key)
        merged.append(term)
    return merged[:INTRODUCED_TERMS_MAX]
