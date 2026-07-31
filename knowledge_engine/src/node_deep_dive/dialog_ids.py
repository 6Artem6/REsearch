"""Стабильные msg_id для истории диалога тьютора (порядок и merge без потерь)."""

from __future__ import annotations

from typing import Any

MSG_ID_KEY = "msg_id"


def parse_msg_id(item: dict[str, Any] | None) -> int | None:
    if not item:
        return None
    raw = item.get(MSG_ID_KEY) or item.get("id")
    if raw is None:
        return None
    try:
        n = int(str(raw).strip())
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def max_msg_id(messages: list[dict[str, Any]] | None) -> int:
    best = 0
    for m in messages or []:
        mid = parse_msg_id(m)
        if mid is not None:
            best = max(best, mid)
    return best


def dialog_message(role: str, content: str, msg_id: int) -> dict[str, str]:
    r = role if role in ("user", "tutor") else "tutor"
    text = (content or "").strip()
    return {"role": r, "content": text, MSG_ID_KEY: str(msg_id)}


def ensure_msg_ids(
    messages: list[dict[str, Any]] | None,
    start_seq: int = 0,
) -> tuple[list[dict[str, str]], int]:
    """Присвоить msg_id legacy-репликам; возвращает список и следующий seq."""
    seq = max(0, start_seq)
    out: list[dict[str, str]] = []
    for raw in messages or []:
        role = str(raw.get("role") or "tutor").strip()
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        mid = parse_msg_id(raw)
        if mid is None:
            seq += 1
            mid = seq
        else:
            seq = max(seq, mid)
        row: dict[str, str] = {
            "role": role if role in ("user", "tutor") else "tutor",
            "content": content,
            MSG_ID_KEY: str(mid),
        }
        html = str(raw.get("content_html") or "").strip()
        if html:
            row["content_html"] = html
        out.append(row)
    return out, seq


def sort_by_msg_id(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    cleaned, _ = ensure_msg_ids(messages)
    cleaned.sort(key=lambda m: parse_msg_id(m) or 0)
    return cleaned


def merge_history_by_msg_id(
    history: list[dict[str, Any]] | None,
    window: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Объединение history + active_window по msg_id (окно перекрывает history)."""
    hist_ensured, seq = ensure_msg_ids(history)
    by_id: dict[int, dict[str, str]] = {}
    for m in hist_ensured:
        mid = parse_msg_id(m)
        if mid is not None:
            by_id[mid] = m
    window_ensured, _ = ensure_msg_ids(window, start_seq=seq)
    for m in window_ensured:
        mid = parse_msg_id(m)
        if mid is not None:
            by_id[mid] = m
    if not by_id:
        return sort_by_msg_id(history)
    return [by_id[k] for k in sorted(by_id)]


def clean_dialog_rows(
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Нормализация role/content без изменения порядка реплик."""
    out: list[dict[str, str]] = []
    for raw in messages or []:
        role = str(raw.get("role") or "tutor").strip()
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        if role not in ("user", "tutor"):
            role = "tutor"
        row: dict[str, str] = {"role": role, "content": content}
        mid = str(raw.get(MSG_ID_KEY) or raw.get("id") or "").strip()
        if mid:
            row[MSG_ID_KEY] = mid
        html = str(raw.get("content_html") or "").strip()
        if html:
            row["content_html"] = html
        out.append(row)
    return out


def _dialog_content_key(item: dict[str, Any]) -> tuple[str, str]:
    role = str(item.get("role") or "tutor").strip()
    if role not in ("user", "tutor"):
        role = "tutor"
    content = str(item.get("content") or "").strip()
    return role, content


def reconcile_dialog_history(
    history: list[dict[str, Any]] | None,
    window: list[dict[str, Any]] | None,
    *,
    start_seq: int = 0,
) -> tuple[list[dict[str, str]], int]:
    """
    Хронологический порядок: реплики из history, которых нет в window (например intro),
    затем active_window в порядке очереди. Переназначает msg_id без коллизий.
    """
    hist_rows = clean_dialog_rows(history)
    win_rows = clean_dialog_rows(window)

    win_keys = {_dialog_content_key(m) for m in win_rows}
    prefix = [m for m in hist_rows if _dialog_content_key(m) not in win_keys]
    ordered = prefix + win_rows
    if not ordered:
        return clean_dialog_rows(history), max(0, start_seq)
    out, seq = ensure_msg_ids(ordered, start_seq=max(0, start_seq))
    return out, seq


def next_msg_id(memory: Any, extra: list[dict[str, Any]] | None = None) -> int:
    seq = int(getattr(memory, "dialog_seq", 0) or 0)
    seq = max(seq, max_msg_id(getattr(memory, "active_window", None)))
    seq = max(seq, max_msg_id(extra))
    return seq + 1


def upsert_history_turn(
    history: list[dict[str, Any]],
    role: str,
    content: str,
    msg_id: int,
) -> list[dict[str, str]]:
    row = dialog_message(role, content, msg_id)
    ensured, _ = ensure_msg_ids(history)
    out: list[dict[str, str]] = []
    replaced = False
    for m in ensured:
        mid = parse_msg_id(m)
        if mid is not None and mid == msg_id:
            out.append(row)
            replaced = True
        else:
            out.append(m)
    if not replaced:
        out.append(row)
    return out


def sync_session_history_turns(
    history: list[dict[str, Any]] | None,
    memory: Any | None,
    *,
    user_message: str | None = None,
    tutor_message: str | None = None,
) -> list[dict[str, str]]:
    """Добавить user/tutor в history с msg_id из active_window или новым seq."""
    hist, seq = ensure_msg_ids(history)
    if memory is not None:
        seq = max(seq, int(getattr(memory, "dialog_seq", 0) or 0))
        seq = max(seq, max_msg_id(getattr(memory, "active_window", None)))

    window = list(getattr(memory, "active_window", None) or [])

    user_text = (user_message or "").strip()
    if user_text:
        uid: int | None = None
        for m in reversed(window):
            if m.get("role") == "user" and (m.get("content") or "").strip() == user_text:
                uid = parse_msg_id(m)
                break
        if uid is None:
            seq += 1
            uid = seq
        hist = upsert_history_turn(hist, "user", user_text, uid)

    tutor_text = (tutor_message or "").strip()
    if tutor_text:
        tid: int | None = None
        if window and (window[-1].get("role") or "") == "tutor":
            last = (window[-1].get("content") or "").strip()
            if last == tutor_text or not last:
                tid = parse_msg_id(window[-1])
        if tid is None:
            for m in reversed(window):
                if m.get("role") == "tutor" and (m.get("content") or "").strip() == tutor_text:
                    tid = parse_msg_id(m)
                    break
        if tid is None:
            seq = max(seq, max_msg_id(hist))
            seq += 1
            tid = seq
        hist = upsert_history_turn(hist, "tutor", tutor_text, tid)

    if memory is not None:
        memory.dialog_seq = max(seq, max_msg_id(hist))

    return hist
