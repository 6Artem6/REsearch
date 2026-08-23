"""Ollama, пути данных, Gemini и поисковые провайдеры.

Все переменные окружения читаются здесь (после `_load_dotenv()`).
Остальной код импортирует константы из этого модуля, не `os.getenv`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT: Path = Path(__file__).resolve().parent

# Ключи из .env всегда перекрывают export в shell (типичный случай: GRAPH_VERSION=0.8)
_DOTENV_FORCE_OVERRIDE_KEYS = frozenset(
    {
        "GRAPH_VERSION",
        "SEMANTIC_SCHOLAR_ENABLED",
    }
)


def _load_dotenv() -> None:
    """Подхват .env из корня репо и knowledge_engine/."""
    candidates = [
        PACKAGE_ROOT.parent / ".env",
        PACKAGE_ROOT / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            if key in _DOTENV_FORCE_OVERRIDE_KEYS:
                os.environ[key] = val
            else:
                os.environ.setdefault(key, val)


def get_graph_version() -> str:
    """Актуальный GRAPH_VERSION после подхвата .env."""
    _load_dotenv()
    return (os.getenv("GRAPH_VERSION", "0.4") or "0.4").strip()


_load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_AUTO_START: bool = _env_bool("OLLAMA_AUTO_START", False)
ROUTER_MODEL: str = "qwen2.5-coder:1.5b"
MAIN_MODEL: str = "qwen2.5-coder:7b"
LOCAL_ROUTER_MODEL: str = os.getenv("LOCAL_ROUTER_MODEL", ROUTER_MODEL)
LOCAL_HEAVY_MODEL: str = os.getenv("LOCAL_HEAVY_MODEL", MAIN_MODEL)
# Smart Selection Prompts (v0.8) — быстрые подсказки при выделении текста
SELECTION_PROMPTS_OLLAMA_MODEL: str = os.getenv(
    "SELECTION_PROMPTS_OLLAMA_MODEL",
    os.getenv("LOCAL_ROUTER_MODEL", ROUTER_MODEL),
)
SELECTION_PROMPTS_TIMEOUT_SEC: float = float(
    os.getenv("SELECTION_PROMPTS_TIMEOUT_SEC", "60")
)
SELECTION_PROMPTS_NUM_PREDICT: int = int(
    os.getenv("SELECTION_PROMPTS_NUM_PREDICT", "256")
)
SELECTION_PROMPTS_KEEP_ALIVE: str = os.getenv(
    "SELECTION_PROMPTS_KEEP_ALIVE",
    os.getenv("OLLAMA_ROUTER_KEEP_ALIVE", "5m"),
)
# Pre-filter картинок перед VLM (article ingest)
ARTICLE_DIAGRAM_FILTER_OLLAMA_MODEL: str = os.getenv(
    "ARTICLE_DIAGRAM_FILTER_OLLAMA_MODEL",
    os.getenv("MAIN_MODEL", "qwen2.5-coder:7b"),
)
ARTICLE_DIAGRAM_FILTER_TIMEOUT_SEC: float = float(
    os.getenv("ARTICLE_DIAGRAM_FILTER_TIMEOUT_SEC", "45")
)
ARTICLE_DIAGRAM_FILTER_NUM_PREDICT: int = int(
    os.getenv("ARTICLE_DIAGRAM_FILTER_NUM_PREDICT", "256")
)
ARTICLE_DIAGRAM_FILTER_NUM_CTX: int = int(
    os.getenv("ARTICLE_DIAGRAM_FILTER_NUM_CTX", "4096")
)
ARTICLE_MAX_DIAGRAMS_PER_ARTICLE: int = int(
    os.getenv("ARTICLE_MAX_DIAGRAMS_PER_ARTICLE", "4")
)
BLOG_SPATIAL_SUMMARIZER_MODEL: str = os.getenv(
    "BLOG_SPATIAL_SUMMARIZER_MODEL",
    os.getenv("MAIN_MODEL", "qwen2.5-coder:7b"),
)
BLOG_SPATIAL_NUM_CTX: int = int(os.getenv("BLOG_SPATIAL_NUM_CTX", "16384"))
BLOG_SPATIAL_NUM_PREDICT: int = int(os.getenv("BLOG_SPATIAL_NUM_PREDICT", "4096"))
BLOG_SPATIAL_TIMEOUT_SEC: float = float(os.getenv("BLOG_SPATIAL_TIMEOUT_SEC", "180"))
# MAP window size — fixed for every provider/model (Prompt Caching + TPM budget).
BLOG_SPATIAL_MAP_MAX_TOKENS: int = 2800
BLOG_SPATIAL_MAP_USER_OVERHEAD_TOKENS: int = int(
    os.getenv("BLOG_SPATIAL_MAP_USER_OVERHEAD_TOKENS", "192")
)
BLOG_SPATIAL_OVERLAP_TOKENS: int = int(os.getenv("BLOG_SPATIAL_OVERLAP_TOKENS", "400"))
# Set after MAX_CONCURRENT_MAP_REQUESTS (see below) — placeholder overwritten there.
BLOG_SPATIAL_MAP_CONCURRENCY: int = 4
BLOG_SPATIAL_TRIAGE_ENABLED: bool = os.getenv(
    "BLOG_SPATIAL_TRIAGE_ENABLED", "true"
).lower() in (
    "1",
    "true",
    "yes",
)
# После LLM-подрезки текста сохранять все FIG (схемы) для spatial/VLM
BLOG_SPATIAL_TRIAGE_KEEP_FIGURES: bool = os.getenv(
    "BLOG_SPATIAL_TRIAGE_KEEP_FIGURES",
    "true",
).lower() in ("1", "true", "yes")
# MAP не выбрал схемы — отправить все FIG из разметки в VLM (arxiv и др.)
BLOG_SPATIAL_VLM_FALLBACK_ALL_FIGURES: bool = os.getenv(
    "BLOG_SPATIAL_VLM_FALLBACK_ALL_FIGURES",
    "true",
).lower() in ("1", "true", "yes")
BLOG_SPATIAL_MAP_PROVIDER: str = (
    os.getenv("BLOG_SPATIAL_MAP_PROVIDER", "gemma_cloud").strip().lower()
)
# VLM diagram extraction (Gemini Lite multimodal, отдельный запрос на изображение)
VLM_GEMINI_MODEL: str = os.getenv(
    "VLM_GEMINI_MODEL",
    "gemini-3.5-flash-lite",
).strip()
# Единый пул VLM (round-robin + failover). Пусто → VLM_GEMINI_MODEL + GEMINI_LITE_FALLBACK_MODELS
VLM_GEMINI_MODELS: str = os.getenv("VLM_GEMINI_MODELS", "").strip()
# --- Shared Flash Lite quotas (3.5 + 3.1): VLM, tutor, map overflow, curriculum… ---
# Official free tier ≈ 15 RPM / 250k TPM / 500 RPD per model → local hard caps below.
GEMINI_FLASH_LITE_MAX_RPM: int = int(os.getenv("GEMINI_FLASH_LITE_MAX_RPM", "14"))
GEMINI_FLASH_LITE_MAX_TPM: int = int(os.getenv("GEMINI_FLASH_LITE_MAX_TPM", "250000"))
GEMINI_FLASH_LITE_MAX_RPD: int = int(os.getenv("GEMINI_FLASH_LITE_MAX_RPD", "490"))
# VLM pool aliases the same shared ceilings (override only if you must diverge)
VLM_GEMINI_MAX_RPM: int = int(
    os.getenv("VLM_GEMINI_MAX_RPM", str(GEMINI_FLASH_LITE_MAX_RPM))
)
VLM_GEMINI_MAX_TPM: int = int(
    os.getenv("VLM_GEMINI_MAX_TPM", str(GEMINI_FLASH_LITE_MAX_TPM))
)
VLM_GEMINI_MAX_RPD: int = int(
    os.getenv("VLM_GEMINI_MAX_RPD", str(GEMINI_FLASH_LITE_MAX_RPD))
)
VLM_GEMINI_CONCURRENCY: int = int(os.getenv("VLM_GEMINI_CONCURRENCY", "3"))
VLM_GEMINI_EST_INPUT_TOKENS: int = int(
    os.getenv("VLM_GEMINI_EST_INPUT_TOKENS", "12000")
)
VLM_GEMINI_EST_OUTPUT_TOKENS: int = int(
    os.getenv("VLM_GEMINI_EST_OUTPUT_TOKENS", "1024")
)
VLM_GEMINI_QUOTA_TRACK: bool = os.getenv("VLM_GEMINI_QUOTA_TRACK", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def flash_lite_rate_limits_live() -> tuple[int, int, int]:
    """Shared per-model Flash Lite caps (all project roles, not VLM-only)."""
    return (
        int(os.getenv("GEMINI_FLASH_LITE_MAX_RPM", str(GEMINI_FLASH_LITE_MAX_RPM))),
        int(os.getenv("GEMINI_FLASH_LITE_MAX_TPM", str(GEMINI_FLASH_LITE_MAX_TPM))),
        int(os.getenv("GEMINI_FLASH_LITE_MAX_RPD", str(GEMINI_FLASH_LITE_MAX_RPD))),
    )


def vlm_gemini_rate_limits_live() -> tuple[int, int, int]:
    """VLM pool uses shared Flash Lite ceilings (env aliases kept for compat)."""
    rpm, tpm, rpd = flash_lite_rate_limits_live()
    return (
        int(os.getenv("VLM_GEMINI_MAX_RPM", str(rpm))),
        int(os.getenv("VLM_GEMINI_MAX_TPM", str(tpm))),
        int(os.getenv("VLM_GEMINI_MAX_RPD", str(rpd))),
    )


def vlm_gemini_concurrency_live() -> int:
    return max(1, int(os.getenv("VLM_GEMINI_CONCURRENCY", str(VLM_GEMINI_CONCURRENCY))))


def vlm_gemini_est_tokens_live() -> tuple[int, int]:
    inp = int(
        os.getenv("VLM_GEMINI_EST_INPUT_TOKENS", str(VLM_GEMINI_EST_INPUT_TOKENS))
    )
    out = int(
        os.getenv("VLM_GEMINI_EST_OUTPUT_TOKENS", str(VLM_GEMINI_EST_OUTPUT_TOKENS))
    )
    return inp, out


def vlm_gemini_model_live() -> str:
    return (os.getenv("VLM_GEMINI_MODEL", VLM_GEMINI_MODEL) or VLM_GEMINI_MODEL).strip()


def vlm_gemini_quota_track_live() -> bool:
    raw = os.getenv("VLM_GEMINI_QUOTA_TRACK")
    if raw is None:
        return VLM_GEMINI_QUOTA_TRACK
    return raw.lower() in ("1", "true", "yes", "on")


def refresh_vlm_gemini_env_from_dotenv() -> None:
    """Перечитать VLM_GEMINI_* из .env (перекрывает export без рестарта worker)."""
    candidates = [
        PACKAGE_ROOT.parent / ".env",
        PACKAGE_ROOT / ".env",
    ]
    prefix = "VLM_GEMINI_"
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key.startswith(prefix):
                os.environ[key] = val


GEMMA_API_BASE: str = os.getenv("GEMMA_API_BASE", "").strip() or (
    "https://generativelanguage.googleapis.com/v1beta/openai"
    if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    and not os.getenv("GEMMA_API_KEY", "").strip()
    else "https://api.openai.com/v1"
)
GEMMA_API_KEY: str = (
    os.getenv("GEMMA_API_KEY", "").strip()
    or (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
)


def gemma_cloud_api_key_available() -> bool:
    """Gemma 4 cloud (MAP, Mermaid repair): достаточно GEMINI_API_KEY / GOOGLE_API_KEY."""
    return bool((GEMMA_API_KEY or "").strip())


# Legacy alias → primary
GEMMA_MODEL_NAME: str = os.getenv("GEMMA_MODEL_NAME", "gemma-4-31b-it").strip()
GEMMA_PRIMARY_MODEL: str = os.getenv(
    "GEMMA_PRIMARY_MODEL",
    GEMMA_MODEL_NAME,
).strip()
GEMMA_FALLBACK_MODEL: str = os.getenv(
    "GEMMA_FALLBACK_MODEL",
    "gemma-4-26b-a4b-it",
).strip()
# Google AI Studio: Gemma 4 (31B + 26B) — общая категория «Other models» на ключ.
GEMMA_MAX_RPM: int = int(os.getenv("GEMMA_MAX_RPM", "30"))
GEMMA_MAX_TPM: int = int(os.getenv("GEMMA_MAX_TPM", "16000"))
GEMMA_MAX_RPD: int = int(os.getenv("GEMMA_MAX_RPD", "14400"))
GEMMA_QUOTA_SHARED: bool = os.getenv("GEMMA_QUOTA_SHARED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
GEMMA_PRIMARY_MAX_RPM: int = int(os.getenv("GEMMA_PRIMARY_MAX_RPM", str(GEMMA_MAX_RPM)))
GEMMA_PRIMARY_MAX_TPM: int = int(os.getenv("GEMMA_PRIMARY_MAX_TPM", str(GEMMA_MAX_TPM)))
GEMMA_PRIMARY_MAX_RPD: int = int(os.getenv("GEMMA_PRIMARY_MAX_RPD", str(GEMMA_MAX_RPD)))
GEMMA_FALLBACK_MAX_RPM: int = int(
    os.getenv("GEMMA_FALLBACK_MAX_RPM", str(GEMMA_MAX_RPM))
)
GEMMA_FALLBACK_MAX_TPM: int = int(
    os.getenv("GEMMA_FALLBACK_MAX_TPM", str(GEMMA_MAX_TPM))
)
GEMMA_FALLBACK_MAX_RPD: int = int(
    os.getenv("GEMMA_FALLBACK_MAX_RPD", str(GEMMA_MAX_RPD))
)
# Оценка in+out на один MAP-запрос (документация / legacy budget tooling).
GEMMA_EST_REQUEST_TOKENS: int = int(os.getenv("GEMMA_EST_REQUEST_TOKENS", "4000"))

# Fixed MAP parallelism for all providers/models (16k TPM budget).
# 4 × ~2.8–3k input ≈ 11–12k TPM, ~4k headroom for completion tokens.
MAX_CONCURRENT_MAP_REQUESTS: int = 4
# Keep Ollama/env alias in lockstep with the unified MAP semaphore.
BLOG_SPATIAL_MAP_CONCURRENCY = MAX_CONCURRENT_MAP_REQUESTS


def gemma_map_concurrency_live() -> int:
    """MAP parallel slots — always ``MAX_CONCURRENT_MAP_REQUESTS`` (all models)."""
    return MAX_CONCURRENT_MAP_REQUESTS


def map_pipeline_concurrency() -> int:
    """Unified MAP in-flight cap for Gemma cloud, Ollama, and any MAP backend."""
    return MAX_CONCURRENT_MAP_REQUESTS


GEMMA_CONCURRENCY: int = MAX_CONCURRENT_MAP_REQUESTS
# MAP completion cap (fixed for every model / window size).
GEMMA_MAP_MAX_OUTPUT_TOKENS: int = 4096
GEMMA_REDUCE_MAX_OUTPUT_TOKENS: int = int(
    os.getenv("GEMMA_REDUCE_MAX_OUTPUT_TOKENS", "4096")
)
# REDUCE strategy: "two_phase" (dedup atoms → synthesis) or "legacy" (single call).
_REDUCE_STRATEGY_RAW = (
    (os.getenv("REDUCE_STRATEGY", "two_phase") or "two_phase").strip().lower()
)
REDUCE_STRATEGY: str = (
    _REDUCE_STRATEGY_RAW
    if _REDUCE_STRATEGY_RAW in ("two_phase", "legacy")
    else "two_phase"
)
_CLAIM_DEDUP_MODE_RAW = (os.getenv("CLAIM_DEDUP_MODE", "none") or "none").strip().lower()
_CLAIM_DEDUP_ALLOWED = ("none", "exact", "entity_consensus", "llm", "claim_mmr")
CLAIM_DEDUP_MODE: str = (
    _CLAIM_DEDUP_MODE_RAW if _CLAIM_DEDUP_MODE_RAW in _CLAIM_DEDUP_ALLOWED else "none"
)
CLAIM_MMR_LAMBDA: float = float(os.getenv("CLAIM_MMR_LAMBDA", "0.7"))
SPO_CLUSTER_THRESHOLD: float = float(os.getenv("SPO_CLUSTER_THRESHOLD", "0.85"))
SPO_RERANKER_DUPLICATE_THRESHOLD: float = float(
    os.getenv("SPO_RERANKER_DUPLICATE_THRESHOLD", "0.88")
)
MAX_CONSENSUS_BATCH_TOKENS: int = int(os.getenv("MAX_CONSENSUS_BATCH_TOKENS", "3072"))
MAX_CONSENSUS_NODES_PER_BATCH: int = int(os.getenv("MAX_CONSENSUS_NODES_PER_BATCH", "10"))
MAX_PRIMARY_ANCHORS: int = int(os.getenv("MAX_PRIMARY_ANCHORS", "3"))
GEMMA_EST_OUTPUT_TOKENS: int = int(
    os.getenv("GEMMA_EST_OUTPUT_TOKENS", str(GEMMA_MAP_MAX_OUTPUT_TOKENS))
)
GEMMA_API_TIMEOUT_SEC: float = float(os.getenv("GEMMA_API_TIMEOUT_SEC", "120"))
GEMMA_FALLBACK_MAX_WAIT_SEC: float = float(
    os.getenv("GEMMA_FALLBACK_MAX_WAIT_SEC", "180")
)
# MAP: выравнивание по UTC :00 и greedy-пакет до safety cap (AI Studio TPM reset).
GEMMA_MAP_FIXED_MINUTE_PACING: bool = os.getenv(
    "GEMMA_MAP_FIXED_MINUTE_PACING", "true"
).strip().lower() in ("1", "true", "yes", "on")
GEMMA_TARGET_TPM_SAFETY_CAP: int = int(
    os.getenv("GEMMA_TARGET_TPM_SAFETY_CAP", "15200")
)
# Два независимых TPM (16k+16k) на один пул MAP-задач; игнорирует GEMMA_QUOTA_SHARED для MAP.
GEMMA_MAP_FORCE_PER_MODEL_LIMITS: bool = os.getenv(
    "GEMMA_MAP_FORCE_PER_MODEL_LIMITS", "true"
).strip().lower() in ("1", "true", "yes", "on")
# Параллельный MAP: на сервере Ollama задайте OLLAMA_NUM_PARALLEL >= concurrency.
# Фоновая экстракция компетенций (Node Deep-Dive, router 1.5B)
COMPETENCY_EXTRACT_OLLAMA_MODEL: str = os.getenv(
    "COMPETENCY_EXTRACT_OLLAMA_MODEL",
    SELECTION_PROMPTS_OLLAMA_MODEL,
)
COMPETENCY_EXTRACT_TIMEOUT_SEC: float = float(
    os.getenv("COMPETENCY_EXTRACT_TIMEOUT_SEC", "45")
)
COMPETENCY_EXTRACT_NUM_PREDICT: int = int(
    os.getenv("COMPETENCY_EXTRACT_NUM_PREDICT", "320")
)
# v0.7 guardrails (Stage 0/1) — structured JSON через Ollama 7B
GUARDRAILS_OLLAMA_MODEL: str = os.getenv(
    "GUARDRAILS_OLLAMA_MODEL",
    os.getenv("GUARDRAILS_MODEL", "qwen2.5-coder:7b"),
)
# Модель для галочек контекста (1.5B — меньше UMA; 7B через CONTEXT_EVAL_MODEL)
CONTEXT_EVAL_MODEL: str = os.getenv("CONTEXT_EVAL_MODEL", ROUTER_MODEL)
CONTEXT_EVAL_NUM_PREDICT: int = int(os.getenv("CONTEXT_EVAL_NUM_PREDICT", "2048"))
# System-wide Bi-Encoder (LanceDB). Cross-Encoder is RAG_CROSS_ENCODER_MODEL only.
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip() or "BAAI/bge-m3"
# Pinned Hub commit: reuse cached pytorch_model.bin; do not follow floating main.
EMBED_MODEL_REVISION: str = os.getenv(
    "EMBED_MODEL_REVISION",
    "5617a9f61b028005a4858fdac845db406aefb181",
).strip()
# Semantic control-chip routing (BGE-M3 cosine vs reference phrases)
VECTOR_INTENT_THRESHOLD: float = float(os.getenv("VECTOR_INTENT_THRESHOLD", "0.82"))
VECTOR_INTENT_ENABLED: bool = os.getenv("VECTOR_INTENT_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# deep_analysis digest ranking: edge/bottleneck/trade-off thesis via embeddings
EDGE_CASE_VECTOR_THRESHOLD: float = float(
    os.getenv("EDGE_CASE_VECTOR_THRESHOLD", "0.48")
)
EDGE_CASE_VECTOR_ENABLED: bool = os.getenv(
    "EDGE_CASE_VECTOR_ENABLED", "true"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Контекст KV: router (1.5B) vs heavy (7B). OLLAMA_NUM_CTX — legacy alias для heavy.
OLLAMA_ROUTER_NUM_CTX: int = int(os.getenv("OLLAMA_ROUTER_NUM_CTX", "2048"))
OLLAMA_HEAVY_NUM_CTX: int = int(
    os.getenv(
        "OLLAMA_HEAVY_NUM_CTX",
        os.getenv("OLLAMA_NUM_CTX", "4096"),
    )
)
OLLAMA_NUM_CTX: int = OLLAMA_HEAVY_NUM_CTX
OLLAMA_ROUTER_KEEP_ALIVE: str = os.getenv("OLLAMA_ROUTER_KEEP_ALIVE", "5m")
OLLAMA_HEAVY_KEEP_ALIVE: str = os.getenv("OLLAMA_HEAVY_KEEP_ALIVE", "5m")
OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
# Guardrails JSON (короткий ValidatedQuerySpec / PersonalContext)
OLLAMA_GUARDRAILS_NUM_PREDICT: int = int(
    os.getenv("OLLAMA_GUARDRAILS_NUM_PREDICT", "1536")
)
# AnalysisReport (3 options + abstractions) — при 1024 JSON обрезается на 3-м варианте
OLLAMA_STRUCTURE_NUM_PREDICT: int = int(
    os.getenv("OLLAMA_STRUCTURE_NUM_PREDICT", "3072")
)
GRAPH_VERSION: str = get_graph_version()


def _gemini_api_key_pool() -> tuple[str, ...]:
    """GEMINI_API_KEYS (comma-separated) или один GEMINI_API_KEY / GOOGLE_API_KEY."""
    raw = (os.getenv("GEMINI_API_KEYS") or "").strip()
    if raw:
        keys = tuple(k.strip() for part in raw.split(",") for k in [part.strip()] if k)
        if keys:
            return keys
    single = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return (single,) if single else ()


GEMINI_API_KEYS: tuple[str, ...] = _gemini_api_key_pool()
GEMINI_API_KEY: str = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# v0.7 analytics: Lite = chunking/step_analysis, Flash = L2a–L2c / tutor
GEMINI_LITE_MODEL: str = os.getenv("GEMINI_LITE_MODEL", "gemini-3.5-flash-lite")
# Частые вызовы (v0.7 L2 / Lite eval): Flash-Lite tier (не reasoner 3.6)
GEMINI_FLASH_MODEL: str = os.getenv("GEMINI_FLASH_MODEL", GEMINI_LITE_MODEL)
# Node Deep-Dive / тьютор / dense в панели — Lite + high-quota fallbacks (не GEMINI_MODEL)
GEMINI_TUTOR_MODEL: str = os.getenv("GEMINI_TUTOR_MODEL", GEMINI_LITE_MODEL)
# Резерв с большим free-tier RPD (например gemma-4-31b)
GEMINI_HIGH_QUOTA_MODEL: str = os.getenv("GEMINI_HIGH_QUOTA_MODEL", "gemma-4-31b-it")


def _parse_model_list_env(key: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(key, default)
    return tuple(m.strip() for part in raw.split(",") for m in [part.strip()] if m)


# Основной поток / fallback при 429 (≤14 RPM Lite tier)
GEMINI_LITE_FALLBACK_MODELS: tuple[str, ...] = _parse_model_list_env(
    "GEMINI_LITE_FALLBACK_MODELS",
    "gemini-3.5-flash-lite,gemini-3.1-flash-lite",
)

# Graph construction only (5 RPM Flash tier)
GEMINI_REASONER_FALLBACK_MODELS: tuple[str, ...] = _parse_model_list_env(
    "GEMINI_REASONER_FALLBACK_MODELS",
    "gemini-3.6-flash,gemini-3.5-flash",
)


def _gemini_fallback_models() -> tuple[str, ...]:
    """Legacy env: по умолчанию Lite chain (не тянуть 3.6 в общий поток)."""
    raw = (
        os.getenv("GEMINI_FALLBACK_MODELS") or os.getenv("GEMINI_FALLBACK_MODEL") or ""
    )
    if raw.strip():
        return tuple(m.strip() for part in raw.split(",") for m in [part.strip()] if m)
    return GEMINI_LITE_FALLBACK_MODELS


GEMINI_FALLBACK_MODELS: tuple[str, ...] = _gemini_fallback_models()


def _gemini_tutor_fallback_models() -> tuple[str, ...]:
    raw = os.getenv("GEMINI_TUTOR_FALLBACK_MODELS", "")
    if raw.strip():
        return _parse_model_list_env("GEMINI_TUTOR_FALLBACK_MODELS", raw)
    return GEMINI_LITE_FALLBACK_MODELS


GEMINI_TUTOR_FALLBACK_MODELS: tuple[str, ...] = _gemini_tutor_fallback_models()
# Макс. длительность одного HTTP-запроса к API (не «пауза»). Reasoner / curriculum.
GEMINI_API_TIMEOUT_SEC: float = float(os.getenv("GEMINI_API_TIMEOUT_SEC", "120"))
# Тьютор / Skill Tree: align with API default — large dialogue prompts often need >45s
# under load (otherwise httpx.ReadTimeout aborts tutor_generate).
GEMINI_TUTOR_TIMEOUT_SEC: float = float(os.getenv("GEMINI_TUTOR_TIMEOUT_SEC", "120"))
KE_NODE_DIVE_TIMEOUT_SEC: float = float(os.getenv("KE_NODE_DIVE_TIMEOUT_SEC", "3600"))
KE_NODE_DIVE_ASYNC_TIMEOUT_SEC: float = float(
    os.getenv(
        "KE_NODE_DIVE_ASYNC_TIMEOUT_SEC",
        "600",
    )
)
KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC: float = float(
    os.getenv("KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC", "3600")
)
# Init с lazy grounding (search + map-reduce): минимум для init_timeout в graph.
KE_NODE_DIVE_INIT_GROUNDING_MIN_TIMEOUT_SEC: float = float(
    os.getenv("KE_NODE_DIVE_INIT_GROUNDING_MIN_TIMEOUT_SEC", "3600")
)
# Быстрый ping перед основным запросом (переключение chain без полного payload).
GEMINI_PROBE_BEFORE_USE: bool = _env_bool("GEMINI_PROBE_BEFORE_USE", False)
GEMINI_PROBE_TIMEOUT_SEC: float = float(os.getenv("GEMINI_PROBE_TIMEOUT_SEC", "12"))
GEMINI_RETRY_BACKOFF_SEC: tuple[float, ...] = tuple(
    float(x.strip())
    for x in os.getenv("GEMINI_RETRY_BACKOFF_SEC", "2,4,8,16").split(",")
    if x.strip()
)
GEMINI_RPM_PAUSE_SEC: float = float(os.getenv("GEMINI_RPM_PAUSE_SEC", "1.5"))
GEMINI_RPM_JITTER_SEC: float = float(os.getenv("GEMINI_RPM_JITTER_SEC", "2"))
# Локальная пауза после 429 без RPD (минутный лимит)
GEMINI_RPM_BLOCK_SEC: float = float(os.getenv("GEMINI_RPM_BLOCK_SEC", "45"))
# Локальный guard: переключать chain до HTTP 429 (доля от hard RPM/TPM)
# Для Flash Lite hard=14 → soft≈12 при 0.9
GEMINI_QUOTA_SAFETY_RATIO: float = float(os.getenv("GEMINI_QUOTA_SAFETY_RATIO", "0.9"))
KE_RAG_TIMEOUT_SEC: float = float(os.getenv("KE_RAG_TIMEOUT_SEC", "60"))
KE_INGEST_URL_CONCURRENCY: int = max(
    1, int(os.getenv("KE_INGEST_URL_CONCURRENCY", "4"))
)
ACADEMIC_INGEST_MAX_BODY_CHARS: int = int(
    os.getenv("ACADEMIC_INGEST_MAX_BODY_CHARS", "80000")
)
# Two-pass inbound gate (Flash Lite structure + parametric credibility) before Gemma MAP.
INGEST_GATE_ENABLED: bool = _env_bool("INGEST_GATE_ENABLED", True)
INGEST_GATE_BLOG_QUALITY_MIN: float = float(
    os.getenv("INGEST_GATE_BLOG_QUALITY_MIN", "0.65")
)
GEMMA_BUDGET_MAX_TPM: int = int(os.getenv("GEMMA_BUDGET_MAX_TPM", "14400"))
GEMMA_BUDGET_MAX_RPM: int = int(os.getenv("GEMMA_BUDGET_MAX_RPM", "27"))
GEMMA_BUDGET_OVERFLOW_WAIT_SEC: float = float(
    os.getenv("GEMMA_BUDGET_OVERFLOW_WAIT_SEC", "10")
)
UNPAYWALL_EMAIL: str = os.getenv(
    "UNPAYWALL_EMAIL",
    "dev@knowledge-engine.local",
)
# OpenAlex polite pool (mailto) + arXiv trust_score enrichment for rag_chunks
OPENALEX_MAILTO: str = os.getenv("OPENALEX_MAILTO", "").strip() or UNPAYWALL_EMAIL
OPENALEX_TIMEOUT_SEC: float = float(os.getenv("OPENALEX_TIMEOUT_SEC", "4"))
OPENALEX_TRUST_ENABLED: bool = _env_bool("OPENALEX_TRUST_ENABLED", True)
OPENALEX_DAILY_LIMIT: int = int(os.getenv("OPENALEX_DAILY_LIMIT", "100000"))
OPENALEX_CONCURRENCY: int = int(os.getenv("OPENALEX_CONCURRENCY", "10"))
# RAG hard cutoff: drop if trust < min AND vector_sim < min_sim
RAG_TRUST_HARD_CUTOFF: bool = _env_bool("RAG_TRUST_HARD_CUTOFF", True)
RAG_TRUST_HARD_MIN_TRUST: float = float(os.getenv("RAG_TRUST_HARD_MIN_TRUST", "0.2"))
RAG_TRUST_HARD_MIN_SIM: float = float(os.getenv("RAG_TRUST_HARD_MIN_SIM", "0.85"))

# Hybrid academic rerank: score = α·sim + β·trust + γ·log_cites + δ·recency
ACADEMIC_RERANK_ENABLED: bool = _env_bool("ACADEMIC_RERANK_ENABLED", False)
ACADEMIC_RERANK_WEIGHTS: str = os.getenv(
    "ACADEMIC_RERANK_WEIGHTS",
    "0.45,0.25,0.20,0.10",
).strip()
ACADEMIC_RERANK_C_SAT: float = float(os.getenv("ACADEMIC_RERANK_C_SAT", "40"))
ACADEMIC_RERANK_RECENCY_HALF_LIFE_YEARS: float = float(
    os.getenv("ACADEMIC_RERANK_RECENCY_HALF_LIFE_YEARS", "6")
)
# Relaxation cascade when academic hits < MIN_HITS
ACADEMIC_RELAXATION_ENABLED: bool = _env_bool("ACADEMIC_RELAXATION_ENABLED", True)
ACADEMIC_RELAXATION_MIN_HITS: int = int(os.getenv("ACADEMIC_RELAXATION_MIN_HITS", "3"))
ACADEMIC_RELAX_L0_MIN_TRUST: float = float(
    os.getenv("ACADEMIC_RELAX_L0_MIN_TRUST", "0.35")
)
ACADEMIC_RELAX_L0_MIN_CITATIONS: int = int(
    os.getenv("ACADEMIC_RELAX_L0_MIN_CITATIONS", "5")
)
ACADEMIC_RELAX_L1_MIN_TRUST: float = float(
    os.getenv("ACADEMIC_RELAX_L1_MIN_TRUST", "0.2")
)
ACADEMIC_RELAX_L1_MIN_CITATIONS: int = int(
    os.getenv("ACADEMIC_RELAX_L1_MIN_CITATIONS", "0")
)
ACADEMIC_RELAX_L1_YEAR_PAD: int = int(os.getenv("ACADEMIC_RELAX_L1_YEAR_PAD", "3"))

# Re-Act evaluator — локально (не Gemini)
REACT_EVAL_MODEL: str = os.getenv("REACT_EVAL_MODEL", ROUTER_MODEL)
LOCAL_L2_MODEL: str = os.getenv("LOCAL_L2_MODEL", MAIN_MODEL)
MIN_PAGE_CHARS_FOR_EXTRACTION: int = int(
    os.getenv("MIN_PAGE_CHARS_FOR_EXTRACTION", "120")
)


# Docker / API: логи в stdout (docker compose logs), без Rich Live-панели
KE_TRACE_STDOUT: bool = _env_bool("KE_TRACE_STDOUT", False)
KE_LOG_PLAIN: bool = _env_bool("KE_LOG_PLAIN", False)
# Полные промпты и сырые ответы каждого LLM-вызова в run log
KE_LLM_FULL_TRACE: bool = _env_bool("KE_LLM_FULL_TRACE", False)
# Файловые Markdown-дампы промптов node_deep_dive (logs/session_traces/)
ENABLE_PROMPT_TRACE_LOGS: bool = _env_bool(
    "ENABLE_PROMPT_TRACE_LOGS",
    _env_bool("KE_API_RELOAD", False),
)
_prompt_trace_dir_raw = (
    os.getenv("PROMPT_TRACE_DIR", "logs/session_traces") or "logs/session_traces"
).strip()
_prompt_trace_path = Path(_prompt_trace_dir_raw)
PROMPT_TRACE_DIR: Path = (
    _prompt_trace_path
    if _prompt_trace_path.is_absolute()
    else (PACKAGE_ROOT.parent / _prompt_trace_path)
).resolve()
# Логировать все Gemini-вызовы (не только node_deep_dive)
PROMPT_TRACE_ALL_LLM: bool = _env_bool("PROMPT_TRACE_ALL_LLM", False)

# Explicit Gemini context cache (layer1 + system_instruction)
ENABLE_GEMINI_EXPLICIT_CACHE: bool = _env_bool("ENABLE_GEMINI_EXPLICIT_CACHE", True)
GEMINI_CACHE_TTL_SECONDS: int = int(os.getenv("GEMINI_CACHE_TTL_SECONDS", "3600"))
GEMINI_CACHE_MIN_EST_TOKENS: int = int(
    os.getenv("GEMINI_CACHE_MIN_EST_TOKENS", "32000")
)

REDIS_URL: str = (os.getenv("REDIS_URL") or "").strip()
KE_USE_REDIS: bool = _env_bool("KE_USE_REDIS", bool(REDIS_URL)) and bool(REDIS_URL)
KE_REDIS_LOGS: bool = _env_bool("KE_REDIS_LOGS", KE_USE_REDIS)
KE_TASKS_CHANNEL: str = os.getenv("KE_TASKS_CHANNEL", "ke:tasks")
KE_REDIS_LOG_MAX_LINES: int = int(os.getenv("KE_REDIS_LOG_MAX_LINES", "20000"))
REDIS_SOCKET_TIMEOUT_SEC: float = float(os.getenv("REDIS_SOCKET_TIMEOUT_SEC", "120"))

KE_API_HOST: str = os.getenv("KE_API_HOST", "127.0.0.1")
KE_API_PORT: int = int(os.getenv("KE_API_PORT", "8765"))
KE_API_RELOAD: bool = _env_bool("KE_API_RELOAD", False)
_raw_ke_api_base = (os.getenv("KE_API_BASE") or "").strip()
KE_API_BASE: str = _raw_ke_api_base or f"http://{KE_API_HOST}:{KE_API_PORT}"

KE_WORKER_POLL_SEC: float = float(os.getenv("KE_WORKER_POLL_SEC", "0.4"))
KE_WORKER_HEARTBEAT_SEC: float = float(os.getenv("KE_WORKER_HEARTBEAT_SEC", "10"))
KE_WORKER_STALE_RUNNING_SEC: float = float(
    os.getenv("KE_WORKER_STALE_RUNNING_SEC", "7200")
)
KE_WORKER_INLINE_FALLBACK: bool = _env_bool("KE_WORKER_INLINE_FALLBACK", False)
KE_WORKER_RELOAD_DEBOUNCE_SEC: float = float(
    os.getenv("KE_WORKER_RELOAD_DEBOUNCE_SEC", "1.0")
)
KE_WORKER_STOP_TIMEOUT_SEC: float = float(os.getenv("KE_WORKER_STOP_TIMEOUT_SEC", "30"))

GEMINI_QUOTA_TRACK: bool = _env_bool("GEMINI_QUOTA_TRACK", True)

PLAYWRIGHT_BROWSERS_PATH: str = (os.getenv("PLAYWRIGHT_BROWSERS_PATH") or "").strip()


def _init_gemini_client() -> Any | None:
    if not GEMINI_API_KEY:
        return None
    if os.getenv("SKIP_GEMINI", "false").lower() in ("1", "true", "yes"):
        return None
    try:
        from google import genai
        from google.genai import types

        timeout_ms = max(1, int(GEMINI_API_TIMEOUT_SEC * 1000))
        return genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
    except ImportError:
        return None


GEMINI_CLIENT: Any | None = _init_gemini_client()
SUMMARIZER_MAX_INPUT_CHARS: int = int(os.getenv("SUMMARIZER_MAX_INPUT_CHARS", "4500"))
SUMMARIZER_MAX_PROFILE_CHARS: int = int(
    os.getenv("SUMMARIZER_MAX_PROFILE_CHARS", "2500")
)
MATRIX_MAX_SUMMARY_CHARS: int = int(os.getenv("MATRIX_MAX_SUMMARY_CHARS", "1200"))

_REPO_REL_LANCE = "knowledge_engine/.lancedb"
_REPO_REL_BROWSER = "knowledge_engine/.browser_state"
_REPO_REL_PROFILE = "knowledge_engine/user_profile.md"

LANCE_DB_PATH: Path = (PACKAGE_ROOT / ".lancedb").resolve()
_ARTICLE_DB_FILE = (PACKAGE_ROOT / ".runs" / "article_diagrams.db").resolve()
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_ARTICLE_DB_FILE.as_posix()}",
)
DOMAIN_TRUST_DB_PATH: Path = (
    Path(
        os.getenv(
            "DOMAIN_TRUST_DB_PATH",
            str(PACKAGE_ROOT / ".domain_trust" / "domains.sqlite"),
        )
    )
    .expanduser()
    .resolve()
)
DOMAIN_TRUST_ENABLED: bool = _env_bool("DOMAIN_TRUST_ENABLED", True)
DOMAIN_TRUST_MIN_SCORE: float = float(os.getenv("DOMAIN_TRUST_MIN_SCORE", "0.4"))
DOMAIN_TRUST_BATCH_SIZE: int = int(os.getenv("DOMAIN_TRUST_BATCH_SIZE", "24"))
DOMAIN_TRUST_HIGH_SCORE: float = float(os.getenv("DOMAIN_TRUST_HIGH_SCORE", "0.75"))
SOURCE_ARCHIVE_ENABLED: bool = _env_bool("SOURCE_ARCHIVE_ENABLED", True)
SOURCE_ARCHIVE_DB_PATH: Path = (
    Path(
        os.getenv(
            "SOURCE_ARCHIVE_DB_PATH",
            str(PACKAGE_ROOT / ".source_archive" / "links.sqlite"),
        )
    )
    .expanduser()
    .resolve()
)
# search | cache_first — сначала доверенные ссылки из архива, затем SearXNG
DISCOVERY_MODE: str = os.getenv("DISCOVERY_MODE", "search").strip().lower()
SMART_QUERY_SYNTAX_ENABLED: bool = _env_bool("SMART_QUERY_SYNTAX_ENABLED", True)
_SEARXNG_DISCOVERY_CATEGORIES_RAW = os.getenv(
    "SEARXNG_DISCOVERY_CATEGORIES",
    "it,science,general",
).strip()
SEARXNG_DISCOVERY_CATEGORIES: tuple[str, ...] = tuple(
    c.strip() for c in _SEARXNG_DISCOVERY_CATEGORIES_RAW.split(",") if c.strip()
)
BROWSER_STATE_PATH: Path = (PACKAGE_ROOT / ".browser_state").resolve()
USER_PROFILE_PATH: Path = (PACKAGE_ROOT / "user_profile.md").resolve()

# chromium | firefox — свой профиль в .browser_state/<engine>/
_raw_pw_browser = os.getenv("PLAYWRIGHT_BROWSER", "chromium").strip().lower()
PLAYWRIGHT_BROWSER: str = (
    _raw_pw_browser if _raw_pw_browser in ("chromium", "firefox") else "chromium"
)
BROWSER_PROFILE_PATH: Path = (BROWSER_STATE_PATH / PLAYWRIGHT_BROWSER).resolve()

LANCE_DB_PATH_SPEC: str = _REPO_REL_LANCE
BROWSER_STATE_PATH_SPEC: str = _REPO_REL_BROWSER
USER_PROFILE_PATH_SPEC: str = _REPO_REL_PROFILE

GRAPH_THREAD_ID: str = os.getenv("GRAPH_THREAD_ID", "knowledge-engine-session")

# --- Consensus.app (Playwright, stateful v0.8) ---
CONSENSUS_START_URL: str = os.getenv(
    "CONSENSUS_START_URL", "https://consensus.app/home"
).rstrip("/")
CONSENSUS_INPUT_SELECTOR: str = os.getenv(
    "CONSENSUS_INPUT_SELECTOR",
    "textarea[data-testid='new-thread-input'],"
    "textarea[data-testid='search-input'],"
    "textarea, div[contenteditable='true']",
)
CONSENSUS_RESPONSE_SELECTOR: str = os.getenv(
    "CONSENSUS_RESPONSE_SELECTOR",
    "[data-testid='answer'], [data-testid*='message'], [class*='Answer'], article .prose, main article",
)
CONSENSUS_SEND_SELECTORS: tuple[str, ...] = tuple(
    s.strip()
    for s in os.getenv(
        "CONSENSUS_SEND_SELECTORS",
        "button[data-testid='search-button'],"
        "[data-testid='search-input-form'] button[type='submit'],"
        "button[aria-label='Submit search'],"
        "button[aria-label*='Submit search'],"
        "button[type='submit']",
    ).split(",")
    if s.strip()
)
CONSENSUS_BROWSER_HEADLESS: bool = os.getenv(
    "CONSENSUS_BROWSER_HEADLESS", "false"
).lower() in (
    "1",
    "true",
    "yes",
)
# Playwright: headed + сохранённый profile (consensus-login). Headless часто login wall / пустая SPA.
CONSENSUS_FORCE_HEADED: bool = _env_bool("CONSENSUS_FORCE_HEADED", True)
if CONSENSUS_FORCE_HEADED:
    CONSENSUS_BROWSER_HEADLESS = False
CONSENSUS_MAX_RETRIES: int = int(os.getenv("CONSENSUS_MAX_RETRIES", "2"))
CONSENSUS_RESPONSE_MAX_SEC: float = float(
    os.getenv("CONSENSUS_RESPONSE_MAX_SEC", "300")
)
CONSENSUS_RESPONSE_FIRST_TIMEOUT_SEC: float = float(
    os.getenv("CONSENSUS_RESPONSE_FIRST_TIMEOUT_SEC", "90")
)
CONSENSUS_STREAM_POLL_SEC: float = float(os.getenv("CONSENSUS_STREAM_POLL_SEC", "1.5"))
CONSENSUS_STREAM_STABLE_ROUNDS: int = int(
    os.getenv("CONSENSUS_STREAM_STABLE_ROUNDS", "4")
)
CONSENSUS_BOOTSTRAP_INPUT_TIMEOUT_SEC: float = float(
    os.getenv("CONSENSUS_BOOTSTRAP_INPUT_TIMEOUT_SEC", "45")
)
CONSENSUS_MIN_RESPONSE_CHARS: int = int(
    os.getenv("CONSENSUS_MIN_RESPONSE_CHARS", "200")
)
# При Sign in / login wall: goto → restart browser → goto (цикл), по умолчанию 2
CONSENSUS_AUTH_RECOVERY_CYCLES: int = int(
    os.getenv("CONSENSUS_AUTH_RECOVERY_CYCLES", "2")
)
# Один Playwright persistent profile между прогонами (логин в cookies локально).
CONSENSUS_REUSE_BROWSER_SESSION: bool = _env_bool(
    "CONSENSUS_REUSE_BROWSER_SESSION", True
)
# Каждый анализ — новый тред на Consensus; RETRY внутри прогона — тот же тред.
CONSENSUS_NEW_THREAD_EACH_RUN: bool = _env_bool("CONSENSUS_NEW_THREAD_EACH_RUN", True)
# Закрывать Chromium после каждого v0.8 harvest (свободная RAM; profile/cookies сохраняются).
CONSENSUS_CLOSE_AFTER_EACH_HARVEST: bool = _env_bool(
    "CONSENSUS_CLOSE_AFTER_EACH_HARVEST", True
)
CONSENSUS_NEW_DIALOG_MAX_WAIT_SEC: float = float(
    os.getenv("CONSENSUS_NEW_DIALOG_MAX_WAIT_SEC", "28")
)
CONSENSUS_UI_POLL_SEC: float = float(os.getenv("CONSENSUS_UI_POLL_SEC", "1.0"))
CONSENSUS_PAPER_HARVEST_PASSES: int = int(
    os.getenv("CONSENSUS_PAPER_HARVEST_PASSES", "10")
)
CONSENSUS_PAPER_HARVEST_PAUSE_SEC: float = float(
    os.getenv("CONSENSUS_PAPER_HARVEST_PAUSE_SEC", "1.2")
)
# Базовый paper search (без Pro AI): /quick/?q=… + модал «Find papers» при лимите.
CONSENSUS_USE_QUICK_PAPER_SEARCH: bool = _env_bool(
    "CONSENSUS_USE_QUICK_PAPER_SEARCH", True
)
CONSENSUS_QUICK_BASE_URL: str = os.getenv(
    "CONSENSUS_QUICK_BASE_URL", "https://consensus.app/quick"
).rstrip("/")
CONSENSUS_QUICK_OPEN_ACCESS: bool = _env_bool("CONSENSUS_QUICK_OPEN_ACCESS", True)
CONSENSUS_QUICK_LOAD_MORE_CLICKS: int = int(
    os.getenv("CONSENSUS_QUICK_LOAD_MORE_CLICKS", "6")
)
CONSENSUS_QUICK_RESULTS_MAX_WAIT_SEC: float = float(
    os.getenv("CONSENSUS_QUICK_RESULTS_MAX_WAIT_SEC", "45")
)
# HAR + verbose JSON traffic (для reverse-engineering API без DOM).
CONSENSUS_RECORD_HAR: bool = _env_bool("CONSENSUS_RECORD_HAR", False)
CONSENSUS_HAR_PATH: Path = Path(
    os.getenv(
        "CONSENSUS_HAR_PATH",
        str(Path(__file__).resolve().parent.parent / "consensus_network_trace.har"),
    )
).expanduser()
CONSENSUS_LOG_JSON_TRAFFIC: bool = _env_bool(
    "CONSENSUS_LOG_JSON_TRAFFIC", CONSENSUS_RECORD_HAR
)
# Hybrid Direct API (curl_cffi + Playwright warmup) вместо DOM/кликов.
CONSENSUS_USE_DIRECT_API: bool = _env_bool("CONSENSUS_USE_DIRECT_API", True)
CONSENSUS_DIRECT_SESSION_MAX_AGE_SEC: float = float(
    os.getenv("CONSENSUS_DIRECT_SESSION_MAX_AGE_SEC", "45")
)
CONSENSUS_DIRECT_WARMUP_URL: str = (
    os.getenv("CONSENSUS_DIRECT_WARMUP_URL", "https://consensus.app/").rstrip("/")
    or "https://consensus.app"
)
# Reasoner — только построение графа учебного плана (не Lite eval / не ingest)
GEMINI_REASONER_MODEL: str = os.getenv("GEMINI_REASONER_MODEL", "gemini-3.6-flash")

AI_CHAT_START_URL: str = "https://gemini.google.com/app"
AI_CHAT_PROVIDER_NAME: str = "Gemini"

GEMINI_INPUT_SELECTOR: str = "div[contenteditable='true'], textarea"
GEMINI_RESPONSE_SELECTOR: str = "message-content, .model-response-text"
GEMINI_SEND_SELECTORS: tuple[str, ...] = tuple(
    s.strip()
    for s in os.getenv(
        "GEMINI_SEND_SELECTORS",
        "button[aria-label*='Send'],button[aria-label*='Отправ'],"
        "[data-test-id='send-button'],button.send-button",
    ).split(",")
    if s.strip()
)

# headless=False снижает риск блокировок Google при первом входе и в диалоге
GEMINI_BROWSER_HEADLESS: bool = os.getenv(
    "GEMINI_BROWSER_HEADLESS", "false"
).lower() in (
    "1",
    "true",
    "yes",
)
GEMINI_RESPONSE_WAIT_SEC: float = 4.0
GEMINI_RESPONSE_MAX_SEC: float = float(os.getenv("GEMINI_RESPONSE_MAX_SEC", "300"))
GEMINI_RESPONSE_FIRST_TIMEOUT_SEC: float = float(
    os.getenv("GEMINI_RESPONSE_FIRST_TIMEOUT_SEC", "120")
)
GEMINI_STREAM_POLL_SEC: float = float(os.getenv("GEMINI_STREAM_POLL_SEC", "1.5"))
GEMINI_STREAM_STABLE_ROUNDS: int = int(os.getenv("GEMINI_STREAM_STABLE_ROUNDS", "4"))
GEMINI_MIN_RESPONSE_CHARS: int = int(os.getenv("GEMINI_MIN_RESPONSE_CHARS", "400"))
GEMINI_PAYLOAD_MAX_CHARS: int = int(os.getenv("GEMINI_PAYLOAD_MAX_CHARS", "14000"))
ROLLING_SUMMARY_MAX_CHARS: int = int(os.getenv("ROLLING_SUMMARY_MAX_CHARS", "1200"))

# --- Провайдеры поиска ---
SEARXNG_BASE_URL: str = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080").rstrip(
    "/"
)
SEARXNG_ENABLED: bool = os.getenv("SEARXNG_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
SEARXNG_TIMEOUT_SEC: float = float(os.getenv("SEARXNG_TIMEOUT_SEC", "15"))
# Движки SearXNG (в дефолте bing disabled, google inactive; в settings.yml включены явно)
SEARXNG_DEFAULT_ENGINES: str = os.getenv("SEARXNG_DEFAULT_ENGINES", "bing,google")
_SEARXNG_CLIENT_IP: str = os.getenv("SEARXNG_CLIENT_IP", "127.0.0.1")
SEARXNG_REQUEST_HEADERS: dict[str, str] = {
    "X-Forwarded-For": _SEARXNG_CLIENT_IP,
    "X-Real-IP": _SEARXNG_CLIENT_IP,
}

SEMANTIC_SCHOLAR_API_URL: str = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_LIMIT: int = int(os.getenv("SEMANTIC_SCHOLAR_LIMIT", "7"))
SEMANTIC_SCHOLAR_TIMEOUT_SEC: float = float(
    os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_SEC", "20")
)
# SS API: 1 request/s cumulative across endpoints (official limit)
SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC: float = float(
    os.getenv("SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC", "1.25")
)
SEMANTIC_SCHOLAR_429_BACKOFF_SEC: float = float(
    os.getenv("SEMANTIC_SCHOLAR_429_BACKOFF_SEC", "1.25")
)
SEMANTIC_SCHOLAR_ENRICH_TIMEOUT_SEC: float = float(
    os.getenv("SEMANTIC_SCHOLAR_ENRICH_TIMEOUT_SEC", "2.0")
)
# SS API часто 429/503 — по умолчанию выключен; v0.7 → arXiv, v0.8 → только карточки Consensus
SEMANTIC_SCHOLAR_ENABLED: bool = _env_bool("SEMANTIC_SCHOLAR_ENABLED", False)
LIGHT_RAG_MIN_COSINE_SIM: float = float(os.getenv("LIGHT_RAG_MIN_COSINE_SIM", "0.42"))
LIGHT_RAG_PROFILE_LIMIT: int = int(os.getenv("LIGHT_RAG_PROFILE_LIMIT", "5"))
# Cross-Encoder: Inbound Gate / RAG rerank ONLY (not domain_registry embeddings).
RAG_CROSS_ENCODER_MODEL: str = os.getenv(
    "RAG_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"
)
RAG_CROSS_ENCODER_REVISION: str = os.getenv(
    "RAG_CROSS_ENCODER_REVISION",
    "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
).strip()
# Cross-Encoder memory: fp16 on MPS, optional idle unload (cross_encoder.py)
RAG_CE_TORCH_DTYPE: str = os.getenv("RAG_CE_TORCH_DTYPE", "auto").strip().lower()
RAG_CE_AUTO_UNLOAD: bool = _env_bool("RAG_CE_AUTO_UNLOAD", False)
RAG_CE_AUTO_UNLOAD_IDLE_SEC: float = float(
    os.getenv("RAG_CE_AUTO_UNLOAD_IDLE_SEC", "300")
)
RAG_DEFAULT_MIN_RELEVANCE: float = float(os.getenv("RAG_DEFAULT_MIN_RELEVANCE", "0.55"))
RAG_DEFAULT_MAX_FACTS: int = int(os.getenv("RAG_DEFAULT_MAX_FACTS", "4"))
RAG_RETRIEVAL_PER_DIRECTION: int = int(os.getenv("RAG_RETRIEVAL_PER_DIRECTION", "5"))
RAG_LATENCY_WARN_MS: float = float(os.getenv("RAG_LATENCY_WARN_MS", "100"))
# Gemma-сжатие длинных фактов в RAG Gateway (должно укладываться в KE_RAG_TIMEOUT_SEC).
RAG_FACT_COMPRESS_GEMMA_TIMEOUT_SEC: float = float(
    os.getenv("RAG_FACT_COMPRESS_GEMMA_TIMEOUT_SEC", "12")
)
RAG_GATEWAY_FINISH_MARGIN_SEC: float = float(
    os.getenv("RAG_GATEWAY_FINISH_MARGIN_SEC", "2")
)
# Опциональные оверрайды в Node Dive / Reasoner / Summarizer (не в DAG/RAG Gateway).
KE_PROMPT_CONTEXT_OVERRIDE_LOCAL_MAC: bool = _env_bool(
    "KE_PROMPT_CONTEXT_OVERRIDE_LOCAL_MAC",
    sys.platform == "darwin",
)
KE_PROMPT_CONTEXT_OVERRIDE_FULLSTACK: bool = _env_bool(
    "KE_PROMPT_CONTEXT_OVERRIDE_FULLSTACK",
    False,
)
ARXIV_API_URL: str = "https://export.arxiv.org/api/query"
# Official arXiv API guidance: ≥3s between successive calls; default 3.25 for margin.
ARXIV_MIN_INTERVAL_SEC: float = float(os.getenv("ARXIV_MIN_INTERVAL_SEC", "3.25"))
ARXIV_MAX_RETRIES: int = int(os.getenv("ARXIV_MAX_RETRIES", "3"))
ARXIV_BACKOFF_BASE_SEC: float = float(os.getenv("ARXIV_BACKOFF_BASE_SEC", "3.0"))
ARXIV_ID_LIST_CHUNK: int = int(os.getenv("ARXIV_ID_LIST_CHUNK", "50"))
ARXIV_TIMEOUT_SEC: float = float(os.getenv("ARXIV_TIMEOUT_SEC", "25"))
CROSSREF_API_URL: str = "https://api.crossref.org/works"
HABR_API_URL: str = "https://habr.com/kairos/v1/articles"

# Exa API (exa-py) — whitelist search (см. services/search/exa_client.py)
EXA_API_KEY: str = (os.getenv("EXA_API_KEY") or "").strip()
CURRICULUM_PRACTICAL_EXA_LIMIT: int = int(
    os.getenv("CURRICULUM_PRACTICAL_EXA_LIMIT", "12")
)
EXA_SEARCH_ENABLED: bool = _env_bool("EXA_SEARCH_ENABLED", True)
EXA_DOMAIN_CAP_PER_HOST: int = int(os.getenv("EXA_DOMAIN_CAP_PER_HOST", "1"))
EXA_RERANK_LITE_THRESHOLD: int = int(os.getenv("EXA_RERANK_LITE_THRESHOLD", "5"))
EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN: int = int(
    os.getenv("EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN", "1")
)
EXA_RECALL_MAX_PER_DOMAIN: int = int(os.getenv("EXA_RECALL_MAX_PER_DOMAIN", "2"))
EXA_FETCH_NUM_RESULTS: int = int(os.getenv("EXA_FETCH_NUM_RESULTS", "20"))
EXA_MAX_CONCURRENT_SEARCH: int = int(os.getenv("EXA_MAX_CONCURRENT_SEARCH", "3"))
EXA_DUAL_QUERY_EN_RATIO: float = float(os.getenv("EXA_DUAL_QUERY_EN_RATIO", "0.7"))
# Exa API: excludeText — одна фраза, до 5 слов (без запятых / нескольких phrase).
EXA_EXCLUDE_TEXT: str = os.getenv(
    "EXA_EXCLUDE_TEXT",
    "api reference documentation sdk classes",
).strip()
EXA_PRACTICAL_HIGHLIGHT_QUERY: str = os.getenv(
    "EXA_PRACTICAL_HIGHLIGHT_QUERY",
    "Engineering blog deep dive: system architecture, implementation trade-offs, "
    "failure modes, benchmarks — not API parameter lists or SDK setup steps.",
).strip()
EXCLUDED_SOURCES_BLACKLIST: tuple[str, ...] = tuple(
    d.strip().lower()
    for d in (
        os.getenv(
            "EXCLUDED_SOURCES_BLACKLIST",
            "medium.com,dev.to,twitter.com,reddit.com,linkedin.com,youtube.com",
        )
    ).split(",")
    if d.strip()
)

# Bi-Encoder alias (must stay BAAI/bge-m3; same space as EMBED_MODEL).
DOMAIN_REGISTRY_EMBED_MODEL: str = (
    os.getenv("DOMAIN_REGISTRY_EMBED_MODEL", EMBED_MODEL).strip() or EMBED_MODEL
)
DOMAIN_REGISTRY_COSINE_MIN: float = float(
    os.getenv("DOMAIN_REGISTRY_COSINE_MIN", "0.82")
)
DOMAIN_REGISTRY_SEARCH_LIMIT: int = int(os.getenv("DOMAIN_REGISTRY_SEARCH_LIMIT", "8"))

# Имена провайдеров в SearchRegistry (можно сузить список)
SEARCH_ACTIVE_PROVIDERS: tuple[str, ...] = (
    "google_meta",
    "semantic_scholar",
    "habr",
    "consensus",
    "arxiv",
    "crossref",
)


def resolved_search_active_providers() -> tuple[str, ...]:
    """Добавляет exa в начало списка при наличии EXA_API_KEY."""
    names = list(SEARCH_ACTIVE_PROVIDERS)
    if EXA_API_KEY and EXA_SEARCH_ENABLED and "exa" not in names:
        names.insert(0, "exa")
    return tuple(names)


# Search-First curriculum (предпоиск перед Flash)
CURRICULUM_SEARCH_TARGET_HITS: int = int(
    os.getenv("CURRICULUM_SEARCH_TARGET_HITS", "15")
)
CURRICULUM_SEARCH_MIN_HITS: int = int(os.getenv("CURRICULUM_SEARCH_MIN_HITS", "8"))
CURRICULUM_OPEN_SEARCH_QUERY_CONCURRENCY: int = int(
    os.getenv("CURRICULUM_OPEN_SEARCH_QUERY_CONCURRENCY", "6")
)
CURRICULUM_SEARCH_PROBE_URLS: bool = _env_bool("CURRICULUM_SEARCH_PROBE_URLS", True)
CURRICULUM_SEARCH_FIRST_ENABLED: bool = _env_bool(
    "CURRICULUM_SEARCH_FIRST_ENABLED", False
)
CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED: bool = _env_bool(
    "CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED", True
)
CURRICULUM_MODEL_FIRST_MIN_NODES: int = int(
    os.getenv("CURRICULUM_MODEL_FIRST_MIN_NODES", "8")
)
CURRICULUM_MODEL_FIRST_TARGET_NODES: int = int(
    os.getenv("CURRICULUM_MODEL_FIRST_TARGET_NODES", "10")
)
CURRICULUM_LITE_BATCH_STRICT: bool = _env_bool("CURRICULUM_LITE_BATCH_STRICT", True)
CURRICULUM_DEEP_NODE_MAX_HITS: int = int(
    os.getenv("CURRICULUM_DEEP_NODE_MAX_HITS", "4")
)
CURRICULUM_DEEP_NODE_REPLENISH_POOL: int = int(
    os.getenv("CURRICULUM_DEEP_NODE_REPLENISH_POOL", "15")
)
CURRICULUM_ACADEMIC_MIN_VALID_REUSE_AFTER_LITE: int = int(
    os.getenv("CURRICULUM_ACADEMIC_MIN_VALID_REUSE_AFTER_LITE", "2")
)
# hybrid на DEEP-ноде: сначала практика (Exa/SearXNG) + Lite; академика только если пусто
CURRICULUM_DEEP_HYBRID_PRACTICAL_FIRST: bool = _env_bool(
    "CURRICULUM_DEEP_HYBRID_PRACTICAL_FIRST", True
)
# Практический SearXNG: только category general (веб: Google/Bing).
CURRICULUM_PRACTICAL_SEARXNG_ENGINES: str = os.getenv(
    "CURRICULUM_PRACTICAL_SEARXNG_ENGINES", "google,bing"
)
# Игнорируется для HTTP: в коде всегда categories=["general"].
CURRICULUM_PRACTICAL_SEARXNG_CATEGORIES: str = os.getenv(
    "CURRICULUM_PRACTICAL_SEARXNG_CATEGORIES", "general"
)
CURRICULUM_ACADEMIC_SEARXNG_LIMIT: int = int(
    os.getenv("CURRICULUM_ACADEMIC_SEARXNG_LIMIT", "8")
)
CURRICULUM_USE_V08_CONSENSUS: bool = _env_bool("CURRICULUM_USE_V08_CONSENSUS", True)
CURRICULUM_CONSENSUS_MIN_APPROVED_ACADEMIC: int = int(
    os.getenv("CURRICULUM_CONSENSUS_MIN_APPROVED_ACADEMIC", "2")
)
# Legacy: игнорируется, режим задаётся UI generation_mode (fast | consensus)
CURRICULUM_CONSENSUS_PRIMARY: bool = _env_bool("CURRICULUM_CONSENSUS_PRIMARY", False)
CURRICULUM_V08_MAX_PAPERS: int = int(os.getenv("CURRICULUM_V08_MAX_PAPERS", "10"))
# Сколько карточек накапливаем из API/DOM перед отбором в MAX_PAPERS (Lite может отсеять часть).
CURRICULUM_V08_PAPER_POOL_SIZE: int = int(
    os.getenv("CURRICULUM_V08_PAPER_POOL_SIZE", "75")
)
# Lazy grounding / node/init: лёгкий academic path
CURRICULUM_ON_DEMAND_V08_MAX_PAPERS: int = int(
    os.getenv("CURRICULUM_ON_DEMAND_V08_MAX_PAPERS", "3")
)
CURRICULUM_ON_DEMAND_V08_POOL_SIZE: int = int(
    os.getenv("CURRICULUM_ON_DEMAND_V08_POOL_SIZE", "15")
)
CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC: float = float(
    os.getenv("CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC", "5")
)
# Extra wait after practical completes when on_demand fast-return applies (SOTA+Consensus uses max(env, 40s)).
CURRICULUM_ON_DEMAND_MIN_PRACTICAL_FOR_FAST_RETURN: int = int(
    os.getenv("CURRICULUM_ON_DEMAND_MIN_PRACTICAL_FOR_FAST_RETURN", "3")
)
ACADEMIC_FAST_FETCH_TIMEOUT_SEC: float = float(
    os.getenv("ACADEMIC_FAST_FETCH_TIMEOUT_SEC", "8.0")
)
ACADEMIC_SCIHUB_TIMEOUT_SEC: float = float(
    os.getenv("ACADEMIC_SCIHUB_TIMEOUT_SEC", "1.5")
)
CURRICULUM_GEMINI_GROUNDING_ENABLED: bool = _env_bool(
    "CURRICULUM_GEMINI_GROUNDING_ENABLED", False
)
CURRICULUM_GEMINI_GROUNDING_MAX_URLS: int = int(
    os.getenv("CURRICULUM_GEMINI_GROUNDING_MAX_URLS", "8")
)
# Playwright gemini.google.com (отдельный профиль .browser_state, не ручной Chrome)
CURRICULUM_GEMINI_WEB_HARVEST_ENABLED: bool = _env_bool(
    "CURRICULUM_GEMINI_WEB_HARVEST_ENABLED", True
)
CURRICULUM_GEMINI_WEB_HARVEST_TIMEOUT_SEC: float = float(
    os.getenv("CURRICULUM_GEMINI_WEB_HARVEST_TIMEOUT_SEC", "120")
)
CURRICULUM_GEMINI_WEB_RESPONSE_MAX_SEC: float = float(
    os.getenv("CURRICULUM_GEMINI_WEB_RESPONSE_MAX_SEC", "90")
)
CURRICULUM_GEMINI_WEB_RESPONSE_FIRST_TIMEOUT_SEC: float = float(
    os.getenv("CURRICULUM_GEMINI_WEB_RESPONSE_FIRST_TIMEOUT_SEC", "45")
)
CURRICULUM_GEMINI_WEB_URL_RETRY_MAX: int = int(
    os.getenv("CURRICULUM_GEMINI_WEB_URL_RETRY_MAX", "3")
)
CURRICULUM_URL_VALIDATE_TIMEOUT_SEC: float = float(
    os.getenv("CURRICULUM_URL_VALIDATE_TIMEOUT_SEC", "10")
)
# Google Custom Search (практические блоги)
GOOGLE_CSE_API_KEY: str = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
# По умолчанию выкл. (GCP Custom Search часто требует billing card)
CURRICULUM_GOOGLE_CSE_ENABLED: bool = _env_bool("CURRICULUM_GOOGLE_CSE_ENABLED", False)
GOOGLE_CSE_DAILY_LIMIT: int = int(os.getenv("GOOGLE_CSE_DAILY_LIMIT", "100"))
SEMANTIC_SCHOLAR_DAILY_LIMIT: int = int(
    os.getenv("SEMANTIC_SCHOLAR_DAILY_LIMIT", "100")
)
CURRICULUM_API_QUOTA_TRACK: bool = _env_bool("CURRICULUM_API_QUOTA_TRACK", True)
CURRICULUM_PRACTICAL_CSE_LIMIT: int = int(
    os.getenv("CURRICULUM_PRACTICAL_CSE_LIMIT", "8")
)
CURRICULUM_PRACTICAL_DDGS_LIMIT: int = int(
    os.getenv("CURRICULUM_PRACTICAL_DDGS_LIMIT", "5")
)
CURRICULUM_PRACTICAL_DDGS_ENABLED: bool = _env_bool(
    "CURRICULUM_PRACTICAL_DDGS_ENABLED", False
)
CURRICULUM_PRACTICAL_SEARXNG_LIMIT: int = int(
    os.getenv("CURRICULUM_PRACTICAL_SEARXNG_LIMIT", "12")
)
CURRICULUM_PRACTICAL_SEARXNG_QUERIES: int = int(
    os.getenv("CURRICULUM_PRACTICAL_SEARXNG_QUERIES", "6")
)
CURRICULUM_LITE_BATCH_EVAL_FALLBACK_N: int = int(
    os.getenv("CURRICULUM_LITE_BATCH_EVAL_FALLBACK_N", "3")
)
# Дублирует Lite Search Query Architect (build_search_queries); по умолчанию выключен.
CURRICULUM_LITE_SITE_SUGGEST_ENABLED: bool = os.getenv(
    "CURRICULUM_LITE_SITE_SUGGEST_ENABLED", "false"
).lower() in ("1", "true", "yes", "on")
CURRICULUM_PRACTICAL_SNIPPET_MIN_CHARS: int = int(
    os.getenv("CURRICULUM_PRACTICAL_SNIPPET_MIN_CHARS", "300")
)
CURRICULUM_ACADEMIC_SS_LIMIT: int = int(os.getenv("CURRICULUM_ACADEMIC_SS_LIMIT", "5"))
CURRICULUM_ACADEMIC_ARXIV_LIMIT: int = int(
    os.getenv("CURRICULUM_ACADEMIC_ARXIV_LIMIT", "3")
)
CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS: int = int(
    os.getenv("CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS", "300")
)
# Search grounding (Google Search tool) — отключено по умолчанию (не используется в прод-пайплайне).
GEMINI_GROUNDING_ENABLED: bool = _env_bool("GEMINI_GROUNDING_ENABLED", False)
_GROUNDING_MODEL_ALIASES: dict[str, str] = {
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
}


def _normalize_grounding_model_id(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return m
    return _GROUNDING_MODEL_ALIASES.get(m.lower(), m)


def _grounding_model_from_env(key: str, default: str) -> str:
    raw = (os.getenv(key) or default).strip()
    if not raw or raw.lower() in ("none", "disabled", "off"):
        return ""
    return _normalize_grounding_model_id(raw)


GEMINI_GROUNDING_MODEL: str = _grounding_model_from_env("GEMINI_GROUNDING_MODEL", "")
CURRICULUM_GEMINI_GROUNDING_MODEL: str = _normalize_grounding_model_id(
    os.getenv("CURRICULUM_GEMINI_GROUNDING_MODEL", GEMINI_GROUNDING_MODEL or "")
)
_GROUNDING_FALLBACK_RAW: str = (
    os.getenv("CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS") or ""
).strip()
CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS: tuple[str, ...] = (
    tuple(
        _normalize_grounding_model_id(m)
        for part in _GROUNDING_FALLBACK_RAW.split(",")
        for m in [part.strip()]
        if m
    )
    if _GROUNDING_FALLBACK_RAW
    else ()
)

MAX_SEARCH_ITERATIONS: int = 3
MAX_AI_DIALOGUE_TURNS: int = int(os.getenv("MAX_AI_DIALOGUE_TURNS", "3"))
# multi_search: меньше URL = меньше Playwright + 7B summarizer (главный cost)
MAX_FETCH_URLS: int = int(os.getenv("MAX_FETCH_URLS", "3"))
MULTI_SEARCH_SKIP_VISION: bool = os.getenv(
    "MULTI_SEARCH_SKIP_VISION", "true"
).lower() in (
    "1",
    "true",
    "yes",
)
SKIP_GEMINI: bool = os.getenv("SKIP_GEMINI", "false").lower() in ("1", "true", "yes")
# После успешного Gemini: API-горизонты + 1 индекс в LanceDB, без 7B×N URL
GEMINI_PRIMARY: bool = os.getenv("GEMINI_PRIMARY", "true").lower() in (
    "1",
    "true",
    "yes",
)
REQUIRE_GEMINI: bool = os.getenv("REQUIRE_GEMINI", "false").lower() in (
    "1",
    "true",
    "yes",
)
MAX_LANCE_INDEX_URLS: int = int(os.getenv("MAX_LANCE_INDEX_URLS", "1"))
# Deep Researcher loop
MAX_RESEARCH_SOURCES: int = int(os.getenv("MAX_RESEARCH_SOURCES", "5"))
MAX_URLS: int = int(os.getenv("MAX_URLS", str(MAX_RESEARCH_SOURCES)))
MIN_VALIDATED_SOURCES: int = int(os.getenv("MIN_VALIDATED_SOURCES", "2"))
MAX_RESEARCH_FIND_ROUNDS: int = int(os.getenv("MAX_RESEARCH_FIND_ROUNDS", "2"))
MAX_RESEARCH_DEPTH: int = int(os.getenv("MAX_RESEARCH_DEPTH", "2"))
# LangGraph supersteps (узлы); по умолчанию хватает для MAX_URLS × fetch-loop + Re-Act
_GRAPH_RECURSION_EXPLICIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "0"))
if _GRAPH_RECURSION_EXPLICIT > 0:
    GRAPH_RECURSION_LIMIT: int = _GRAPH_RECURSION_EXPLICIT
else:
    _per_url_loop = 5
    GRAPH_RECURSION_LIMIT = (
        12
        + MAX_URLS * _per_url_loop
        + MAX_RESEARCH_DEPTH * (14 + MAX_URLS * _per_url_loop)
    )
# Уточнение ТТЖ: через Gemini (Deep Researcher) или interrupt пользователя
CLARIFY_VIA_GEMINI: bool = os.getenv("CLARIFY_VIA_GEMINI", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Подстроки URL — не парсить (шум из Bing, не RAG)
URL_BLOCKLIST_SUBSTR: tuple[str, ...] = tuple(
    s.strip()
    for s in os.getenv(
        "URL_BLOCKLIST_SUBSTR",
        "youtube.com,youtu.be,microsoft.com,geeksforgeeks.org,wikipedia.org",
    ).split(",")
    if s.strip()
)
URL_PRIORITY_SUBSTR: tuple[str, ...] = (
    "arxiv.org",
    "doi.org",
    "semanticscholar.org",
    "habr.com",
    "peerj.com",
    "crossref",
)
RAG_HYBRID_LIMIT: int = int(os.getenv("RAG_HYBRID_LIMIT", "3"))
RAG_MIN_RELEVANT_HITS: int = int(os.getenv("RAG_MIN_RELEVANT_HITS", "2"))
LECTURE_RAG_TOP_K: int = int(os.getenv("LECTURE_RAG_TOP_K", "3"))
# Lecture dense: расширенный пул → CE rerank → MMR (services/lecture_context_rerank.py)
LECTURE_RAG_CANDIDATE_LIMIT: int = int(os.getenv("LECTURE_RAG_CANDIDATE_LIMIT", "8"))
LECTURE_RAG_MMR_TOP_K: int = int(os.getenv("LECTURE_RAG_MMR_TOP_K", "3"))
LECTURE_RAG_CE_MIN_SCORE: float = float(os.getenv("LECTURE_RAG_CE_MIN_SCORE", "0.50"))
LECTURE_RAG_CONTEXT_MAX_CHARS: int = int(
    os.getenv("LECTURE_RAG_CONTEXT_MAX_CHARS", "9000")
)
LECTURE_RAG_PROMPT_MAX_CHARS: int = int(
    os.getenv("LECTURE_RAG_PROMPT_MAX_CHARS", "9000")
)
CHAT_SESSION_API_TURNS_MAX: int = int(os.getenv("CHAT_SESSION_API_TURNS_MAX", "8"))
LECTURE_RAG_MMR_LAMBDA: float = float(os.getenv("LECTURE_RAG_MMR_LAMBDA", "0.62"))
# Chunk cross-attention + MMR перед Reduce (services/chunk_cross_attention_mmr.py)
LECTURE_CHUNK_CA_TOP_K: int = int(os.getenv("LECTURE_CHUNK_CA_TOP_K", "10"))
LECTURE_CHUNK_CA_ALPHA: float = float(os.getenv("LECTURE_CHUNK_CA_ALPHA", "0.7"))
LECTURE_CHUNK_CA_BETA: float = float(os.getenv("LECTURE_CHUNK_CA_BETA", "0.3"))
LECTURE_CHUNK_CA_GAMMA: float = float(os.getenv("LECTURE_CHUNK_CA_GAMMA", "0.55"))
LECTURE_CHUNK_CA_MAX_PER_SOURCE: int = int(
    os.getenv("LECTURE_CHUNK_CA_MAX_PER_SOURCE", "2")
)
LECTURE_CHUNK_CA_ENABLED: bool = os.getenv("LECTURE_CHUNK_CA_ENABLED", "1").strip() in (
    "1",
    "true",
    "yes",
)
# Doc-level gate before fine-chunk CA/MMR (cosine doc_meta vs query)
DOC_GATE_THRESHOLD: float = float(os.getenv("DOC_GATE_THRESHOLD", "0.40"))
# Fine chunking for rag_chunks LanceDB table
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "600"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
RAG_CHUNK_SEARCH_LIMIT: int = int(os.getenv("RAG_CHUNK_SEARCH_LIMIT", "64"))
MAX_CHUNKS_PER_DOC: int = int(
    os.getenv("MAX_CHUNKS_PER_DOC", os.getenv("LECTURE_CHUNK_CA_MAX_PER_SOURCE", "2"))
)
# Relative gap + anchor / multi-source selection (chunk_cross_attention_mmr.py)
RAG_SCORE_MIN_FLOOR: float = float(os.getenv("RAG_SCORE_MIN_FLOOR", "0.30"))
RAG_KNEE_DROP_RATIO: float = float(os.getenv("RAG_KNEE_DROP_RATIO", "0.12"))
RAG_ANCHOR_THRESHOLD: float = float(os.getenv("RAG_ANCHOR_THRESHOLD", "0.70"))
RAG_CHUNK_SEMANTIC_DEDUP: float = float(os.getenv("RAG_CHUNK_SEMANTIC_DEDUP", "0.85"))
RAG_ANCHOR_SUPPLEMENT_MAX: int = int(os.getenv("RAG_ANCHOR_SUPPLEMENT_MAX", "2"))
LECTURE_RAG_PREFILTER_MIN_PRIMARY_CHUNKS: int = int(
    os.getenv("LECTURE_RAG_PREFILTER_MIN_PRIMARY_CHUNKS", "3")
)
LECTURE_RAG_SECONDARY_SCORE_FLOOR: float = float(
    os.getenv("LECTURE_RAG_SECONDARY_SCORE_FLOOR", "0.40")
)
LECTURE_RAG_SCOPE_SECONDARY_PENALTY: float = float(
    os.getenv("LECTURE_RAG_SCOPE_SECONDARY_PENALTY", "0.75")
)
LECTURE_RAG_RERANK_TIMEOUT_SEC: float = float(
    os.getenv("LECTURE_RAG_RERANK_TIMEOUT_SEC", "60")
)
LECTURE_RAG_COLLECT_TIMEOUT_SEC: float = float(
    os.getenv("LECTURE_RAG_COLLECT_TIMEOUT_SEC", "90")
)
LECTURE_RAG_LIGHT_TIMEOUT_SEC: float = float(
    os.getenv("LECTURE_RAG_LIGHT_TIMEOUT_SEC", "45")
)
NODE_DIVE_LECTURE_RAG_TIMEOUT_SEC: float = float(
    os.getenv("NODE_DIVE_LECTURE_RAG_TIMEOUT_SEC", "120")
)
NODE_DIVE_LECTURE_SEARCH_TIMEOUT_SEC: float = float(
    os.getenv("NODE_DIVE_LECTURE_SEARCH_TIMEOUT_SEC", "10")
)
LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC: float = float(
    os.getenv("LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC", "12")
)
LECTURE_RAG_KNODE_CANDIDATE_LIMIT: int = int(
    os.getenv("LECTURE_RAG_KNODE_CANDIDATE_LIMIT", "4")
)
LECTURE_EXTERNAL_SEARCH_ENABLED: bool = os.getenv(
    "LECTURE_EXTERNAL_SEARCH_ENABLED", "true"
).lower() in ("1", "true", "yes", "on")
LECTURE_EXTERNAL_SEARCH_TOP_K: int = int(
    os.getenv("LECTURE_EXTERNAL_SEARCH_TOP_K", "3")
)
# Alias used by lecture external-search waterfall (Exa early-exit cap).
MAX_EXTERNAL_SOURCES: int = int(
    os.getenv("MAX_EXTERNAL_SOURCES", str(LECTURE_EXTERNAL_SEARCH_TOP_K))
)
LECTURE_MIN_LOCAL_SOURCES: int = int(os.getenv("LECTURE_MIN_LOCAL_SOURCES", "3"))
# Avg local RAG retrieval_score below this → allow external search (with count gate).
LECTURE_LOCAL_QUALITY_THRESHOLD: float = float(
    os.getenv("LECTURE_LOCAL_QUALITY_THRESHOLD", "0.50")
)
LOCAL_QUALITY_THRESHOLD: float = float(
    os.getenv("LOCAL_QUALITY_THRESHOLD", str(LECTURE_LOCAL_QUALITY_THRESHOLD))
)
# Dialog tutor: per-turn knowledge_atoms retrieval (no parent-chunk expand).
DIALOG_ATOMS_TOP_K: int = int(os.getenv("DIALOG_ATOMS_TOP_K", "6"))
DIALOG_ATOMS_MIN_SCORE: float = float(os.getenv("DIALOG_ATOMS_MIN_SCORE", "0.35"))
DIALOG_ATOMS_ENABLED: bool = os.getenv("DIALOG_ATOMS_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Selection Explainer: Target Anchor + PRINCIPLE/MECHANIC atoms + causal facts.
EXPLAIN_ATOMS_ENABLED: bool = os.getenv("EXPLAIN_ATOMS_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
EXPLAIN_ATOMS_TOP_K: int = int(os.getenv("EXPLAIN_ATOMS_TOP_K", "3"))
EXPLAIN_ATOMS_MIN_SCORE: float = float(os.getenv("EXPLAIN_ATOMS_MIN_SCORE", "0.35"))
EXPLAIN_CAUSAL_FACTS_TOP_K: int = int(os.getenv("EXPLAIN_CAUSAL_FACTS_TOP_K", "5"))
EXPLAIN_ANCHOR_FALLBACK_ENABLED: bool = os.getenv(
    "EXPLAIN_ANCHOR_FALLBACK_ENABLED", "true"
).lower() in ("1", "true", "yes", "on")
EXPLAIN_ANCHOR_FALLBACK_TOP_K: int = int(
    os.getenv("EXPLAIN_ANCHOR_FALLBACK_TOP_K", "2")
)
LECTURE_MAX_OUTPUT_TOKENS: int = int(os.getenv("LECTURE_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_TUTOR_MAX_OUTPUT_TOKENS: int = int(
    os.getenv("GEMINI_TUTOR_MAX_OUTPUT_TOKENS", "8192")
)
GEMINI_INTRO_MAX_OUTPUT_TOKENS: int = int(
    os.getenv("GEMINI_INTRO_MAX_OUTPUT_TOKENS", "2048")
)
GEMINI_LITE_MAX_OUTPUT_TOKENS: int = int(
    os.getenv("GEMINI_LITE_MAX_OUTPUT_TOKENS", "4096")
)
LECTURE_GENERATION_TEMPERATURE: float = float(
    os.getenv("LECTURE_GENERATION_TEMPERATURE", "0.3")
)
LECTURE_GENERATION_TIMEOUT_SEC: float = float(
    os.getenv("LECTURE_GENERATION_TIMEOUT_SEC", "180")
)
LECTURE_MIN_WORDS_TARGET: int = int(os.getenv("LECTURE_MIN_WORDS_TARGET", "1200"))


def set_api_server(host: str, port: int, reload: bool) -> None:
    """CLI / main: переключить HTTP API без os.environ."""
    global KE_API_HOST, KE_API_PORT, KE_API_RELOAD, KE_API_BASE
    KE_API_HOST = host
    KE_API_PORT = int(port)
    KE_API_RELOAD = bool(reload)
    KE_API_BASE = f"http://{KE_API_HOST}:{KE_API_PORT}"


def apply_cli_gemini_research_mode() -> None:
    """CLI --gemini-research: Gemini bulk + локальный Re-Act."""
    global SKIP_GEMINI, REQUIRE_GEMINI, GEMINI_PRIMARY, SEARXNG_ENABLED
    SKIP_GEMINI = False
    REQUIRE_GEMINI = True
    GEMINI_PRIMARY = True
    SEARXNG_ENABLED = False
