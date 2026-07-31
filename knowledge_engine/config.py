"""Ollama, пути данных, Gemini и поисковые провайдеры.

Все переменные окружения читаются здесь (после `_load_dotenv()`).
Остальной код импортирует константы из этого модуля, не `os.getenv`.
"""

from __future__ import annotations

import os
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
    os.getenv("SELECTION_PROMPTS_TIMEOUT_SEC", "3")
)
SELECTION_PROMPTS_NUM_PREDICT: int = int(
    os.getenv("SELECTION_PROMPTS_NUM_PREDICT", "256")
)
# v0.7 guardrails (Stage 0/1) — structured JSON через Ollama 7B
GUARDRAILS_OLLAMA_MODEL: str = os.getenv(
    "GUARDRAILS_OLLAMA_MODEL",
    os.getenv("GUARDRAILS_MODEL", "qwen2.5-coder:7b"),
)
# Модель для галочек контекста (1.5B — меньше UMA; 7B через CONTEXT_EVAL_MODEL)
CONTEXT_EVAL_MODEL: str = os.getenv("CONTEXT_EVAL_MODEL", ROUTER_MODEL)
CONTEXT_EVAL_NUM_PREDICT: int = int(os.getenv("CONTEXT_EVAL_NUM_PREDICT", "2048"))
EMBED_MODEL: str = "nomic-embed-text"
# Контекст KV: router (1.5B) vs heavy (7B). OLLAMA_NUM_CTX — legacy alias для heavy.
OLLAMA_ROUTER_NUM_CTX: int = int(os.getenv("OLLAMA_ROUTER_NUM_CTX", "2048"))
OLLAMA_HEAVY_NUM_CTX: int = int(
    os.getenv(
        "OLLAMA_HEAVY_NUM_CTX",
        os.getenv("OLLAMA_NUM_CTX", "4096"),
    )
)
OLLAMA_NUM_CTX: int = OLLAMA_HEAVY_NUM_CTX
OLLAMA_ROUTER_KEEP_ALIVE: str = os.getenv("OLLAMA_ROUTER_KEEP_ALIVE", "2m")
OLLAMA_HEAVY_KEEP_ALIVE: str = os.getenv("OLLAMA_HEAVY_KEEP_ALIVE", "2m")
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
GEMINI_LITE_MODEL: str = os.getenv("GEMINI_LITE_MODEL", "gemini-3.1-flash-lite")
# Частые вызовы (v0.7 L2): Lite, не 3.5 Flash (5 RPM)
GEMINI_FLASH_MODEL: str = os.getenv("GEMINI_FLASH_MODEL", GEMINI_LITE_MODEL)
# Node Deep-Dive / тьютор / dense в панели — Lite + high-quota fallbacks (не GEMINI_MODEL)
GEMINI_TUTOR_MODEL: str = os.getenv("GEMINI_TUTOR_MODEL", GEMINI_LITE_MODEL)
# Резерв с большим free-tier RPD (например gemma-4-31b)
GEMINI_HIGH_QUOTA_MODEL: str = os.getenv("GEMINI_HIGH_QUOTA_MODEL", "gemma-4-31b-it")


def _parse_model_list_env(key: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(key, default)
    return tuple(m.strip() for part in raw.split(",") for m in [part.strip()] if m)


# Основной поток / fallback при 429 (15 RPM Lite tier)
GEMINI_LITE_FALLBACK_MODELS: tuple[str, ...] = _parse_model_list_env(
    "GEMINI_LITE_FALLBACK_MODELS",
    "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemma-4-26b-a4b-it,gemma-4-31b-it",
)

# Только reasoner / curriculum (5 RPM Flash tier)
GEMINI_REASONER_FALLBACK_MODELS: tuple[str, ...] = _parse_model_list_env(
    "GEMINI_REASONER_FALLBACK_MODELS",
    "gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview",
)


def _gemini_fallback_models() -> tuple[str, ...]:
    """Legacy env: по умолчанию Lite chain (не тянуть 3.6 в общий поток)."""
    raw = os.getenv("GEMINI_FALLBACK_MODELS") or os.getenv("GEMINI_FALLBACK_MODEL") or ""
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
# Тьютор / Skill Tree — короче, чтобы быстрее падать при зависании API.
GEMINI_TUTOR_TIMEOUT_SEC: float = float(os.getenv("GEMINI_TUTOR_TIMEOUT_SEC", "45"))
KE_NODE_DIVE_TIMEOUT_SEC: float = float(os.getenv("KE_NODE_DIVE_TIMEOUT_SEC", "900"))
KE_NODE_DIVE_ASYNC_TIMEOUT_SEC: float = float(
    os.getenv(
        "KE_NODE_DIVE_ASYNC_TIMEOUT_SEC",
        str(max(GEMINI_TUTOR_TIMEOUT_SEC * 3, 120)),
    )
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
KE_RAG_TIMEOUT_SEC: float = float(os.getenv("KE_RAG_TIMEOUT_SEC", "45"))
UNPAYWALL_EMAIL: str = os.getenv(
    "UNPAYWALL_EMAIL",
    "dev@knowledge-engine.local",
)
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

REDIS_URL: str = (os.getenv("REDIS_URL") or "").strip()
KE_USE_REDIS: bool = _env_bool(
    "KE_USE_REDIS", bool(REDIS_URL)
) and bool(REDIS_URL)
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
    os.getenv("KE_WORKER_STALE_RUNNING_SEC", "300")
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
# Reasoner / Heavy (флагман; по умолчанию GEMINI_MODEL)
GEMINI_REASONER_MODEL: str = os.getenv("GEMINI_REASONER_MODEL", GEMINI_MODEL)

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
# Модуль 3 — Directional RAG Gateway (без LLM)
RAG_CROSS_ENCODER_MODEL: str = os.getenv(
    "RAG_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"
)
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
ARXIV_API_URL: str = "https://export.arxiv.org/api/query"
CROSSREF_API_URL: str = "https://api.crossref.org/works"
HABR_API_URL: str = "https://habr.com/kairos/v1/articles"

# Exa API (exa-py) — whitelist search (см. services/search/exa_client.py)
EXA_API_KEY: str = (os.getenv("EXA_API_KEY") or "").strip()
CURRICULUM_PRACTICAL_EXA_LIMIT: int = int(os.getenv("CURRICULUM_PRACTICAL_EXA_LIMIT", "12"))
EXA_SEARCH_ENABLED: bool = _env_bool("EXA_SEARCH_ENABLED", True)
EXA_DOMAIN_CAP_PER_HOST: int = int(os.getenv("EXA_DOMAIN_CAP_PER_HOST", "1"))
EXA_RERANK_LITE_THRESHOLD: int = int(os.getenv("EXA_RERANK_LITE_THRESHOLD", "5"))
EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN: int = int(
    os.getenv("EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN", "1")
)
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
CURRICULUM_SEARCH_TARGET_HITS: int = int(os.getenv("CURRICULUM_SEARCH_TARGET_HITS", "15"))
CURRICULUM_SEARCH_MIN_HITS: int = int(os.getenv("CURRICULUM_SEARCH_MIN_HITS", "8"))
CURRICULUM_SEARCH_PROBE_URLS: bool = _env_bool("CURRICULUM_SEARCH_PROBE_URLS", True)
CURRICULUM_SEARCH_FIRST_ENABLED: bool = _env_bool("CURRICULUM_SEARCH_FIRST_ENABLED", False)
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
CURRICULUM_GOOGLE_CSE_ENABLED: bool = _env_bool(
    "CURRICULUM_GOOGLE_CSE_ENABLED", False
)
GOOGLE_CSE_DAILY_LIMIT: int = int(os.getenv("GOOGLE_CSE_DAILY_LIMIT", "100"))
SEMANTIC_SCHOLAR_DAILY_LIMIT: int = int(os.getenv("SEMANTIC_SCHOLAR_DAILY_LIMIT", "100"))
CURRICULUM_API_QUOTA_TRACK: bool = _env_bool("CURRICULUM_API_QUOTA_TRACK", True)
CURRICULUM_PRACTICAL_CSE_LIMIT: int = int(os.getenv("CURRICULUM_PRACTICAL_CSE_LIMIT", "8"))
CURRICULUM_PRACTICAL_DDGS_LIMIT: int = int(os.getenv("CURRICULUM_PRACTICAL_DDGS_LIMIT", "5"))
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
CURRICULUM_LITE_SITE_SUGGEST_ENABLED: bool = (
    os.getenv("CURRICULUM_LITE_SITE_SUGGEST_ENABLED", "false").lower()
    in ("1", "true", "yes", "on")
)
CURRICULUM_PRACTICAL_SNIPPET_MIN_CHARS: int = int(
    os.getenv("CURRICULUM_PRACTICAL_SNIPPET_MIN_CHARS", "300")
)
CURRICULUM_ACADEMIC_SS_LIMIT: int = int(os.getenv("CURRICULUM_ACADEMIC_SS_LIMIT", "5"))
CURRICULUM_ACADEMIC_ARXIV_LIMIT: int = int(os.getenv("CURRICULUM_ACADEMIC_ARXIV_LIMIT", "3"))
CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS: int = int(
    os.getenv("CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS", "300")
)
# Search grounding (Google Search tool). gemini-2.5-flash-lite — неверное имя API (404).
_GROUNDING_MODEL_ALIASES: dict[str, str] = {
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
}


def _normalize_grounding_model_id(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return m
    return _GROUNDING_MODEL_ALIASES.get(m.lower(), m)


def _grounding_model_from_env(key: str, default: str) -> str:
    return _normalize_grounding_model_id(os.getenv(key, default))


GEMINI_GROUNDING_MODEL: str = _grounding_model_from_env(
    "GEMINI_GROUNDING_MODEL", "gemini-2.5-flash"
)
CURRICULUM_GEMINI_GROUNDING_MODEL: str = _normalize_grounding_model_id(
    os.getenv("CURRICULUM_GEMINI_GROUNDING_MODEL", GEMINI_GROUNDING_MODEL)
)
CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS: tuple[str, ...] = tuple(
    _normalize_grounding_model_id(m)
    for m in _parse_model_list_env(
        "CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS",
        "gemini-3.1-flash-lite,gemini-1.5-flash",
    )
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
LECTURE_RAG_CANDIDATE_LIMIT: int = int(os.getenv("LECTURE_RAG_CANDIDATE_LIMIT", "15"))
LECTURE_RAG_MMR_TOP_K: int = int(os.getenv("LECTURE_RAG_MMR_TOP_K", "5"))
LECTURE_RAG_CE_MIN_SCORE: float = float(os.getenv("LECTURE_RAG_CE_MIN_SCORE", "0.38"))
LECTURE_RAG_MMR_LAMBDA: float = float(os.getenv("LECTURE_RAG_MMR_LAMBDA", "0.62"))
LECTURE_RAG_RERANK_TIMEOUT_SEC: float = float(
    os.getenv("LECTURE_RAG_RERANK_TIMEOUT_SEC", "60")
)
LECTURE_RAG_KNODE_CANDIDATE_LIMIT: int = int(
    os.getenv("LECTURE_RAG_KNODE_CANDIDATE_LIMIT", "4")
)


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
