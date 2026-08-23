"""Gemini explicit cache for ingest REDUCE — isolated from tutor sessions."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from knowledge_engine.config import PACKAGE_ROOT

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_REGISTRY_PATH = PACKAGE_ROOT / ".runs" / "gemini_ingest_cache_registry.json"
_lock = threading.RLock()
_INGEST_KEY_PREFIX = "ingest:"


@dataclass(frozen=True)
class IngestCacheResult:
    mode: str
    cache_name: str = ""
    registry_key: str = ""
    error: str = ""

    @property
    def is_active(self) -> bool:
        return bool(self.cache_name and self.mode in ("hit", "created"))


def ingest_cache_registry_key(doc_id: str, content: str) -> str:
    """Ключ реестра ``ingest:{doc_id}:{sha256(content)}`` — не пересекается с тьютором."""
    digest = hashlib.sha256((content or "").encode("utf-8")).hexdigest()
    did = (doc_id or "doc").strip() or "doc"
    return f"{_INGEST_KEY_PREFIX}{did}:{digest}"


def _is_fatal_cache_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return True
    code = None
    for attr in ("status_code", "code"):
        raw = getattr(exc, attr, None)
        if raw is not None:
            try:
                code = int(raw)
                break
            except (TypeError, ValueError):
                pass
    if code in {400, 403, 404}:
        return True
    return any(tok in msg for tok in (" 400", " 403", " 404", "not found", "permission", "invalid"))


def _is_not_found_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return "not found" in msg or "404" in msg or "resourcenotfound" in msg


class IngestContextCacheManager:
    """Кэш Map Summaries для REDUCE через ``client.caches.create``."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        registry_path: Any | None = None,
        model: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._client = client
        self._registry_path = registry_path or _REGISTRY_PATH
        self._model_override = (model or "").strip()
        self._ttl_override = ttl_seconds

    def _resolve_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        from knowledge_engine.config import GEMINI_CLIENT

        return GEMINI_CLIENT

    def _resolve_model(self) -> str:
        if self._model_override:
            return self._model_override
        from knowledge_engine.config import GEMINI_LITE_MODEL

        return (GEMINI_LITE_MODEL or "").strip()

    def _ttl(self) -> int:
        if self._ttl_override is not None:
            return max(60, int(self._ttl_override))
        from knowledge_engine.config import INGEST_CACHE_TTL_SECONDS

        return max(60, int(INGEST_CACHE_TTL_SECONDS or 86400))

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        with _lock:
            path = self._registry_path
            if not path.is_file():
                return {}
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return raw if isinstance(raw, dict) else {}
            except (json.JSONDecodeError, OSError):
                return {}

    def _save_registry(self, data: dict[str, dict[str, Any]]) -> None:
        with _lock:
            path = self._registry_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def _registry_get(self, key: str) -> dict[str, Any] | None:
        row = self._load_registry().get((key or "").strip())
        return row if isinstance(row, dict) else None

    def _registry_delete(self, key: str) -> None:
        k = (key or "").strip()
        if not k:
            return
        data = self._load_registry()
        if k in data:
            data.pop(k, None)
            self._save_registry(data)

    def _registry_upsert(self, key: str, *, model: str, cache_name: str, ttl_sec: int) -> None:
        data = self._load_registry()
        data[key] = {
            "model": model,
            "cache_name": cache_name,
            "expires_at": time.time() + max(60, int(ttl_sec)),
            "updated_at": time.time(),
            "namespace": "ingest",
        }
        self._save_registry(data)

    def _verify_remote(self, client: Any, cache_name: str, registry_key: str) -> bool:
        name = (cache_name or "").strip()
        if not name:
            return False
        try:
            client.caches.get(name=name)
            return True
        except Exception as exc:
            if _is_not_found_error(exc):
                self._registry_delete(registry_key)
            return False

    def get_or_create(
        self,
        *,
        doc_id: str,
        content: str,
        system_instruction: str,
        model: str | None = None,
    ) -> IngestCacheResult:
        from knowledge_engine import config as ke_config

        if not ke_config.MIGRATION_USE_CONTEXT_CACHING:
            return IngestCacheResult(mode="disabled")
        body = (content or "").strip()
        system = (system_instruction or "").strip()
        mdl = (model or self._resolve_model()).strip()
        if not body or not mdl:
            return IngestCacheResult(mode="skipped_empty")
        client = self._resolve_client()
        if client is None:
            return IngestCacheResult(mode="error_fallback", error="no_gemini_client")

        key = ingest_cache_registry_key(doc_id, body)
        ttl = self._ttl()
        entry = self._registry_get(key)
        if entry:
            cache_name = str(entry.get("cache_name") or "")
            expires_at = float(entry.get("expires_at") or 0)
            entry_model = str(entry.get("model") or "")
            if (
                cache_name
                and expires_at > time.time()
                and entry_model == mdl
                and self._verify_remote(client, cache_name, key)
            ):
                return IngestCacheResult(
                    mode="hit", cache_name=cache_name, registry_key=key
                )
            self._registry_delete(key)

        try:
            from google.genai import types

            digest = key.rsplit(":", 1)[-1]
            cache = client.caches.create(
                model=mdl,
                config=types.CreateCachedContentConfig(
                    display_name=f"ke-ingest-{digest[:12]}",
                    system_instruction=system,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=body)],
                        )
                    ],
                    ttl=f"{int(ttl)}s",
                ),
            )
            cache_name = str(getattr(cache, "name", "") or "")
            if not cache_name:
                raise RuntimeError("caches.create вернул пустой name")
            self._registry_upsert(key, model=mdl, cache_name=cache_name, ttl_sec=ttl)
            return IngestCacheResult(
                mode="created", cache_name=cache_name, registry_key=key
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("Ingest context cache create failed: %s", err[:240])
            return IngestCacheResult(
                mode="error_fallback",
                registry_key=key,
                error=err,
            )

    def generate_structured(
        self,
        *,
        doc_id: str,
        system_instruction: str,
        cache_content: str,
        user_prompt: str,
        schema: type[T],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> T | None:
        """REDUCE по ``cached_content``; при 400/403/404/timeout вернуть None (caller fallback)."""
        from google.genai import types

        from knowledge_engine.ui.run_log import trace

        cache = self.get_or_create(
            doc_id=doc_id,
            content=cache_content,
            system_instruction=system_instruction,
            model=model,
        )
        if not cache.is_active:
            if cache.error:
                trace(f"INGEST_CACHE skip | {cache.mode} | {cache.error[:160]}")
            return None
        client = self._resolve_client()
        mdl = (model or self._resolve_model()).strip()
        if client is None or not mdl:
            return None
        config_kwargs: dict[str, Any] = {
            "cached_content": cache.cache_name,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0.1,
        }
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_tokens)
        try:
            response = client.models.generate_content(
                model=mdl,
                contents=(user_prompt or "").strip() or "Return the REDUCE JSON.",
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise RuntimeError("empty generate_content from ingest cache")
            try:
                return schema.model_validate_json(text)
            except Exception:
                import json

                return schema.model_validate(json.loads(text))
        except Exception as exc:
            # 400 / 403 / 404 / timeout / parse — прозрачный fallback на GemmaCloudClient
            trace(f"INGEST_CACHE generate ✗ | {type(exc).__name__}: {exc}")
            logger.warning(
                "Ingest cached REDUCE failed (%s); fallback to Gemma",
                type(exc).__name__,
            )
            return None
