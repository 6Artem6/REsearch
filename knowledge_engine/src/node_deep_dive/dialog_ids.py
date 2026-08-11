"""Стабильные msg_id для истории диалога тьютора (порядок и merge без потерь)."""

from __future__ import annotations

import re
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


def _normalize_user_dialog_content(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\[mode:\w+\]\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def _dialog_content_key(item: dict[str, Any]) -> tuple[str, str]:
    role = str(item.get("role") or "tutor").strip()
    if role not in ("user", "tutor"):
        role = "tutor"
    content = str(item.get("content") or "").strip()
    if role == "user":
        content = _normalize_user_dialog_content(content)
    return role, content


def _prefer_tutor_display_content(prev: str, new: str) -> str:
    """
    active_window хранит tutor без follow_up_question; history/UI — полная склейка полей.
    При reconcile не заменять длинный history-текст укорочённым window-текстом.
    """
    p = (prev or "").strip()
    n = (new or "").strip()
    if not p:
        return n
    if not n:
        return p
    if n in p or p.endswith(n):
        return p
    if p in n:
        return n
    return p if len(p) >= len(n) else n


def _merge_dialog_row_for_display(
    prev: dict[str, str], new: dict[str, str]
) -> dict[str, str]:
    role = str(new.get("role") or prev.get("role") or "tutor").strip()
    if role != "tutor":
        return new
    content = _prefer_tutor_display_content(
        str(prev.get("content") or ""),
        str(new.get("content") or ""),
    )
    out = {**new, "role": role, "content": content}
    prev_html = str(prev.get("content_html") or "").strip()
    new_html = str(new.get("content_html") or "").strip()
    if prev_html and len(str(prev.get("content") or "")) >= len(
        str(new.get("content") or "")
    ):
        out["content_html"] = prev_html
    elif new_html:
        out["content_html"] = new_html
    return out


def reconcile_dialog_history(
    history: list[dict[str, Any]] | None,
    window: list[dict[str, Any]] | None,
    *,
    start_seq: int = 0,
) -> tuple[list[dict[str, str]], int]:
    """
    Объединить history и active_window: одна реплика на msg_id; user-текст без [mode:*]
    сопоставляется с полным сообщением из API.
    """
    hist_rows = clean_dialog_rows(history)
    win_rows = clean_dialog_rows(window)
    if not win_rows:
        out, seq = ensure_msg_ids(hist_rows, start_seq=max(0, start_seq))
        return out, seq
    if not hist_rows:
        out, seq = ensure_msg_ids(win_rows, start_seq=max(0, start_seq))
        return out, seq

    by_id: dict[int, dict[str, str]] = {}
    order_ids: list[int] = []
    extra: list[dict[str, str]] = []

    def ingest(row: dict[str, str]) -> None:
        mid = parse_msg_id(row)
        if mid is not None:
            if mid not in by_id:
                order_ids.append(mid)
                by_id[mid] = row
            else:
                by_id[mid] = _merge_dialog_row_for_display(by_id[mid], row)
            return
        extra.append(row)

    for m in hist_rows:
        ingest(m)
    for m in win_rows:
        ingest(m)

    merged: list[dict[str, str]] = [by_id[i] for i in order_ids]
    seen_keys: set[tuple[str, str]] = {_dialog_content_key(m) for m in merged}
    for m in extra:
        key = _dialog_content_key(m)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(m)

    deduped: list[dict[str, str]] = []
    for m in merged:
        key = _dialog_content_key(m)
        if deduped and _dialog_content_key(deduped[-1]) == key:
            deduped[-1] = m
        else:
            deduped.append(m)

    out, seq = ensure_msg_ids(deduped, start_seq=max(0, start_seq))
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
        norm_user = _normalize_user_dialog_content(user_text)
        for m in reversed(window):
            if m.get("role") != "user":
                continue
            win_text = (m.get("content") or "").strip()
            if (
                win_text == user_text
                or _normalize_user_dialog_content(win_text) == norm_user
            ):
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
            tid = parse_msg_id(window[-1])
        if tid is None:
            for m in reversed(window):
                if (
                    m.get("role") == "tutor"
                    and (m.get("content") or "").strip() == tutor_text
                ):
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


def patch_last_tutor_history_content(
    history: list[dict[str, Any]] | None,
    tutor_text: str,
) -> list[dict[str, str]]:
    """Подставить полный UI-текст (с follow_up) в последнюю реплику тьютора."""
    text = (tutor_text or "").strip()
    if not text or not history:
        return clean_dialog_rows(history)
    rows = clean_dialog_rows(history)
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get("role") != "tutor":
            continue
        row = dict(rows[i])
        row["content"] = text
        row.pop("content_html", None)
        rows[i] = row
        return rows
    return rows
