"""Post-hoc проверка цитат ``[A\\d+]`` — отдельное пространство от ``[S*]`` / ``[R*]``."""

from __future__ import annotations

import re

# Только [A123]. Не матчит [S1], [R1], arr[0], [ANCHOR: …], уже помеченные unverified.
_ANCHOR_CITE_RE = re.compile(r"\[A(\d+)\](?! \(\? unverified\))")
UNVERIFIED_ANCHOR_SUFFIX = " (? unverified)"


def _allowed_keys(valid_anchors: set[str] | None) -> set[str]:
    keys: set[str] = set()
    for raw in valid_anchors or set():
        item = str(raw).strip()
        if not item:
            continue
        keys.add(item)
        if item.startswith("A") and item[1:].isdigit():
            keys.add(item[1:])
            keys.add(f"[{item}]")
        elif item.isdigit():
            keys.add(f"A{item}")
            keys.add(f"[A{item}]")
        elif item.startswith("[A") and item.endswith("]") and item[2:-1].isdigit():
            keys.add(item[1:-1])
            keys.add(item[2:-1])
    return keys


def validate_and_annotate_anchors(
    text: str,
    valid_anchors: set[str],
) -> tuple[str, list[str]]:
    """Подменить неизвестные ``[A99]`` на ``[A99 (? unverified)]``.

    Текст не удаляется. No-op если ``ANCHOR_REGEX_VALIDATE=false``.
    ``[S*]``, ``[R*]`` и ``arr[0]`` регулярка не трогает.
    """
    from knowledge_engine import config as ke_config

    raw = text or ""
    if not ke_config.ANCHOR_REGEX_VALIDATE:
        return raw, []
    if not raw:
        return raw, []
    allowed = _allowed_keys(valid_anchors)
    unverified: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        num = match.group(1)
        key = f"A{num}"
        full = match.group(0)
        if key in allowed or num in allowed or full in allowed:
            return full
        unverified.append(key)
        return f"[A{num}{UNVERIFIED_ANCHOR_SUFFIX}]"

    annotated = _ANCHOR_CITE_RE.sub(_replace, raw)
    # stable unique order
    seen: set[str] = set()
    uniq: list[str] = []
    for item in unverified:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return annotated, uniq


def validate_anchor_citations(text: str, allowed_chunk_ids: set[str]) -> str:
    """Совместимость Phase 1: вернуть только аннотированный текст."""
    annotated, _unverified = validate_and_annotate_anchors(text, allowed_chunk_ids)
    return annotated
