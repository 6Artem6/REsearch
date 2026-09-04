"""Redis-backed hot/cold Gemini explicit context cache (cloud-safe registry).

Отдельно от `gemini_cache_manager.py`: тот держит реестр в локальном JSON-файле
(один процесс/диск), этот — в Redis, чтобы cache_name был виден всем воркерам
в облачном деплое. Cold — стабильный system/preset (TTL часы), Hot — контекст
конкретной сессии/ноды (TTL минуты).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from knowledge_engine.config import (
    GEMINI_CLOUD_CACHE_COLD_TTL_SECONDS,
    GEMINI_CLOUD_CACHE_HOT_TTL_SECONDS,
)
from knowledge_engine.services.redis_client import get_redis, redis_enabled
from knowledge_engine.ui.run_log import trace

_COLD_KEY_PREFIX = "cache:cold:"
_HOT_KEY_PREFIX = "cache:hot:"


class CacheMetadata(BaseModel):
    cache_name: str
    created_at: datetime
    ttl_seconds: int
    is_hot: bool


def preset_hash(model: str, system_instruction: str, preset_body: str) -> str:
    payload = (
        f"{(model or '').strip()}\n---\n"
        f"{system_instruction or ''}\n---\n"
        f"{preset_body or ''}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_not_found_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return "not found" in msg or "404" in msg


class CloudCacheManager:
    """Redis registry поверх `client.caches` (Gemini explicit context cache)."""

    def _get_metadata(self, key: str) -> CacheMetadata | None:
        if not redis_enabled():
            return None
        raw = get_redis().get(key)
        if not raw:
            return None
        try:
            return CacheMetadata.model_validate_json(raw)
        except Exception:
            return None

    def _save_metadata(self, key: str, meta: CacheMetadata) -> None:
        if not redis_enabled():
            return
        get_redis().set(key, meta.model_dump_json(), ex=max(60, meta.ttl_seconds))

    def _verify_remote(self, client: Any, cache_name: str) -> bool:
        try:
            client.caches.get(name=cache_name)
            return True
        except Exception as exc:
            if _is_not_found_error(exc):
                return False
            # Unknown/transient error — не хороним валидный кэш из-за сетевого сбоя.
            return True

    def _create_cache(
        self,
        client: Any,
        *,
        key: str,
        model: str,
        system_instruction: str,
        body: str,
        ttl_seconds: int,
        is_hot: bool,
        label: str,
    ) -> CacheMetadata | None:
        from google.genai import types

        digest = key.rsplit(":", 1)[-1]
        try:
            cache = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    display_name=f"ke-cloud-{digest[:16]}",
                    system_instruction=system_instruction or None,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=body)],
                        )
                    ],
                    ttl=f"{int(ttl_seconds)}s",
                ),
            )
        except Exception as exc:
            trace(f"CLOUD_CACHE create ✗ | {label} | {type(exc).__name__}: {exc}")
            return None

        cache_name = str(getattr(cache, "name", "") or "")
        if not cache_name:
            trace(f"CLOUD_CACHE create ✗ | {label} | empty cache name")
            return None

        meta = CacheMetadata(
            cache_name=cache_name,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=int(ttl_seconds),
            is_hot=is_hot,
        )
        self._save_metadata(key, meta)
        trace(f"CLOUD_CACHE create ✓ | {label} | {cache_name} | ttl={ttl_seconds}s")
        return meta

    def get_or_create_cold_cache(
        self,
        client: Any,
        *,
        model: str,
        system_instruction: str,
        preset_body: str,
        ttl_seconds: int = GEMINI_CLOUD_CACHE_COLD_TTL_SECONDS,
        label: str = "cloud_cache/cold",
    ) -> CacheMetadata | None:
        """Стабильный system/preset (например статический BLOCK 1). Ключ = хэш пресета."""
        if not redis_enabled():
            trace(f"CLOUD_CACHE cold skip | redis disabled | {label}")
            return None
        mdl = (model or "").strip()
        preset = (preset_body or "").strip()
        if not mdl or not preset:
            return None

        digest = preset_hash(mdl, system_instruction, preset)
        key = f"{_COLD_KEY_PREFIX}{digest}"
        existing = self._get_metadata(key)
        if existing and self._verify_remote(client, existing.cache_name):
            trace(f"CLOUD_CACHE cold hit | {label} | {existing.cache_name}")
            return existing

        return self._create_cache(
            client,
            key=key,
            model=mdl,
            system_instruction=(system_instruction or "").strip(),
            body=preset,
            ttl_seconds=ttl_seconds,
            is_hot=False,
            label=label,
        )

    def get_or_create_hot_session_cache(
        self,
        client: Any,
        *,
        session_id: str,
        model: str,
        node_context: str,
        system_instruction: str = "",
        ttl_seconds: int = GEMINI_CLOUD_CACHE_HOT_TTL_SECONDS,
        label: str = "cloud_cache/hot",
    ) -> CacheMetadata | None:
        """Контекст текущей ноды/сессии. Ключ = session_id, короткий TTL."""
        if not redis_enabled():
            trace(f"CLOUD_CACHE hot skip | redis disabled | {label}")
            return None
        sid = (session_id or "").strip()
        mdl = (model or "").strip()
        context = (node_context or "").strip()
        if not sid or not mdl or not context:
            return None

        key = f"{_HOT_KEY_PREFIX}{sid}"
        existing = self._get_metadata(key)
        if existing and self._verify_remote(client, existing.cache_name):
            trace(f"CLOUD_CACHE hot hit | {label} | {existing.cache_name}")
            return existing

        return self._create_cache(
            client,
            key=key,
            model=mdl,
            system_instruction=(system_instruction or "").strip(),
            body=context,
            ttl_seconds=ttl_seconds,
            is_hot=True,
            label=label,
        )
