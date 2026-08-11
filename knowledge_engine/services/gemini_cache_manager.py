"""Explicit Gemini context cache (layer1 + system) с локальным реестром."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.services.gemini_stateless import estimate_llm_tokens
from knowledge_engine.ui.run_log import trace

_REGISTRY_PATH = PACKAGE_ROOT / ".runs" / "gemini_explicit_cache_registry.json"
_lock = threading.RLock()


@dataclass(frozen=True)
class ExplicitCacheResult:
    mode: str
    cache_name: str = ""
    digest: str = ""
    est_layer1_tokens: int = 0
    error: str = ""


def explicit_cache_digest(
    model: str,
    system_instruction: str,
    layer1_body: str,
) -> str:
    payload = (
        f"{(model or '').strip()}\n---\n"
        f"{system_instruction or ''}\n---\n"
        f"{layer1_body or ''}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def explicit_cache_is_active(cache: ExplicitCacheResult | None) -> bool:
    return bool(cache and cache.cache_name and cache.mode in ("hit", "created"))


def is_cache_resource_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "not found" in msg or "404" in msg or "resourcenotfound" in msg:
        return True
    return "cached_content" in msg and ("invalid" in msg or "not found" in msg)


def apply_explicit_cache_to_generation_config(
    config_kwargs: dict[str, Any],
    *,
    system_instruction: str,
    cache: ExplicitCacheResult | None,
) -> dict[str, Any]:
    """
    При active cache — только cached_content (system уже в кэше).
    Иначе — system_instruction как раньше.
    """
    if explicit_cache_is_active(cache):
        config_kwargs["cached_content"] = cache.cache_name
        config_kwargs.pop("system_instruction", None)
    else:
        if (system_instruction or "").strip():
            config_kwargs["system_instruction"] = system_instruction
    return config_kwargs


def _load_registry() -> dict[str, dict[str, Any]]:
    with _lock:
        if not _REGISTRY_PATH.is_file():
            return {}
        try:
            raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}


def _save_registry(data: dict[str, dict[str, Any]]) -> None:
    with _lock:
        _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _REGISTRY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_REGISTRY_PATH)


def registry_delete(digest: str) -> None:
    d = (digest or "").strip()
    if not d:
        return
    data = _load_registry()
    if d in data:
        data.pop(d, None)
        _save_registry(data)


def registry_purge_for_anchor(anchor: str) -> int:
    """Удалить explicit-cache записи, привязанные к node_deep_dive anchor."""
    needle = (anchor or "").strip()
    if not needle:
        return 0
    data = _load_registry()
    if not data:
        return 0
    kept: dict[str, dict[str, Any]] = {}
    removed = 0
    for digest, row in data.items():
        if not isinstance(row, dict):
            kept[digest] = row
            continue
        key = str(row.get("node_session_key") or "")
        blob = json.dumps(row, ensure_ascii=False)
        if needle in key or needle in blob:
            removed += 1
            continue
        kept[digest] = row
    if removed:
        _save_registry(kept)
        trace(f"GEMINI_CACHE purge | anchor={needle[:72]} removed={removed}")
    return removed


def registry_clear_all() -> list[dict[str, Any]]:
    """Очистить локальный реестр explicit cache; вернуть удалённые записи."""
    with _lock:
        data = _load_registry()
        rows: list[dict[str, Any]] = []
        for digest, row in data.items():
            item: dict[str, Any] = {"digest": digest}
            if isinstance(row, dict):
                item.update(row)
            rows.append(item)
        if data:
            _save_registry({})
        return rows


def _delete_remote_cache(client: Any, cache_name: str) -> bool:
    name = (cache_name or "").strip()
    if not name:
        return False
    try:
        client.caches.delete(name=name)
        return True
    except Exception as exc:
        if _is_not_found_error(exc):
            return True
        trace(
            f"GEMINI explicit cache delete ✗ | {name[:80]} | {type(exc).__name__}: {exc}"
        )
        return False


def invalidate_all_explicit_caches(
    client: Any | None = None,
    *,
    delete_remote: bool = True,
) -> dict[str, int]:
    """Удалить cached_content на Gemini (по реестру) и очистить локальный реестр."""
    rows = registry_clear_all()
    remote_ok = 0
    remote_fail = 0
    if delete_remote and client is not None and rows:
        for row in rows:
            name = str(row.get("cache_name") or "")
            if _delete_remote_cache(client, name):
                remote_ok += 1
            else:
                remote_fail += 1
    trace(
        f"GEMINI_CACHE invalidate_all ✓ | registry={len(rows)} "
        f"remote_deleted={remote_ok} remote_failed={remote_fail} "
        f"delete_remote={delete_remote}"
    )
    return {
        "registry_entries": len(rows),
        "remote_deleted": remote_ok,
        "remote_failed": remote_fail,
    }


def _registry_get(digest: str) -> dict[str, Any] | None:
    return _load_registry().get((digest or "").strip())


def _registry_upsert(
    digest: str,
    *,
    model: str,
    cache_name: str,
    ttl_sec: int,
    node_session_key: str = "",
    label: str = "",
) -> None:
    data = _load_registry()
    data[digest] = {
        "model": model,
        "cache_name": cache_name,
        "expires_at": time.time() + max(60, int(ttl_sec)),
        "node_session_key": node_session_key,
        "label": label,
        "updated_at": time.time(),
    }
    _save_registry(data)


def _is_not_found_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return "not found" in msg or "404" in msg


def verify_cache_on_server(client: Any, cache_name: str, digest: str) -> bool:
    """True если кэш жив; при 404 — удаляет запись из реестра."""
    name = (cache_name or "").strip()
    if not name:
        return False
    try:
        client.caches.get(name=name)
        return True
    except Exception as exc:
        if _is_not_found_error(exc):
            trace(
                f"GEMINI explicit cache stale (404) | {name} | drop registry {digest[:12]}"
            )
            registry_delete(digest)
        return False


def get_or_create_explicit_cache(
    client: Any,
    *,
    model: str,
    system_instruction: str,
    layer1_body: str,
    node_session_key: str = "",
    label: str = "node_deep_dive/tutor",
) -> ExplicitCacheResult:
    from google.genai import types

    from knowledge_engine.config import (
        ENABLE_GEMINI_EXPLICIT_CACHE,
        GEMINI_CACHE_MIN_EST_TOKENS,
        GEMINI_CACHE_TTL_SECONDS,
    )

    if not ENABLE_GEMINI_EXPLICIT_CACHE:
        return ExplicitCacheResult(mode="disabled")

    layer1 = (layer1_body or "").strip()
    system = (system_instruction or "").strip()
    mdl = (model or "").strip()
    if not mdl or not layer1:
        return ExplicitCacheResult(mode="skipped_empty")

    est = estimate_llm_tokens(system, mdl) + estimate_llm_tokens(layer1, mdl)
    if est < GEMINI_CACHE_MIN_EST_TOKENS:
        return ExplicitCacheResult(
            mode="skipped_below_threshold",
            est_layer1_tokens=est,
        )

    digest = explicit_cache_digest(mdl, system, layer1)
    entry = _registry_get(digest)
    if entry:
        cache_name = str(entry.get("cache_name") or "")
        expires_at = float(entry.get("expires_at") or 0)
        entry_model = str(entry.get("model") or "")
        if (
            cache_name
            and expires_at > time.time()
            and entry_model == mdl
            and verify_cache_on_server(client, cache_name, digest)
        ):
            trace(
                f"GEMINI explicit cache hit | {label} | {cache_name} | "
                f"digest={digest[:12]} | est≈{est} tok"
            )
            return ExplicitCacheResult(
                mode="hit",
                cache_name=cache_name,
                digest=digest,
                est_layer1_tokens=est,
            )
        registry_delete(digest)

    try:
        trace(
            f"GEMINI explicit cache create ▶ | {label} | model={mdl} | "
            f"digest={digest[:12]} | est≈{est} tok | ttl={GEMINI_CACHE_TTL_SECONDS}s"
        )
        cache = client.caches.create(
            model=mdl,
            config=types.CreateCachedContentConfig(
                display_name=f"ke-node-dive-{digest[:12]}",
                system_instruction=system,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=layer1)],
                    )
                ],
                ttl=f"{int(GEMINI_CACHE_TTL_SECONDS)}s",
            ),
        )
        cache_name = str(getattr(cache, "name", "") or "")
        if not cache_name:
            raise RuntimeError("caches.create вернул пустой name")
        _registry_upsert(
            digest,
            model=mdl,
            cache_name=cache_name,
            ttl_sec=GEMINI_CACHE_TTL_SECONDS,
            node_session_key=node_session_key,
            label=label,
        )
        trace(f"GEMINI explicit cache create ✓ | {cache_name}")
        return ExplicitCacheResult(
            mode="created",
            cache_name=cache_name,
            digest=digest,
            est_layer1_tokens=est,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        trace(f"GEMINI explicit cache create ✗ | {label} | {err[:240]}")
        return ExplicitCacheResult(
            mode="error_fallback",
            digest=digest,
            est_layer1_tokens=est,
            error=err,
        )
