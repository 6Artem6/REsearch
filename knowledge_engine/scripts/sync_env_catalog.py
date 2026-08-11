"""Синхронизация каталога env: дефолты из config.py → .env.example и блок каталога в .env."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KE = REPO / "knowledge_engine"
CONFIG = KE / "config.py"
ENV_EXAMPLE = REPO / ".env.example"
ENV_FILE = REPO / ".env"

_MARKER_START = "# --- catalog keys not set locally (sync_env_catalog.py) ---"
_MARKER_END = "# --- end sync_env_catalog.py ---"

_PAT_KEY = re.compile(
    r"os\.getenv\(\s*['\"]([^'\"]+)['\"]|_env_bool\(\s*['\"]([^'\"]+)['\"]"
)
_PAT_STR_DEFAULT = re.compile(
    r"os\.getenv\(\s*['\"]([A-Z0-9_]+)['\"]\s*,\s*['\"]([^'\"]*)['\"]"
)
_PAT_BOOL_DEFAULT = re.compile(
    r"_env_bool\(\s*['\"]([A-Z0-9_]+)['\"]\s*,\s*(True|False)"
)
_PAT_BOOL_OS = re.compile(
    r"os\.getenv\(\s*['\"]([A-Z0-9_]+)['\"]\s*,\s*['\"](true|false)['\"]\)\.lower\(\)"
)

ROUTER = "qwen2.5-coder:1.5b"
MAIN = "qwen2.5-coder:7b"

SECRET_KEYS = frozenset(
    {
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_API_KEY",
        "EXA_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "GEMMA_API_KEY",
        "DATABASE_URL",
    }
)

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Core / API / graph",
        [
            "GRAPH_VERSION",
            "GRAPH_THREAD_ID",
            "GRAPH_RECURSION_LIMIT",
            "SEARXNG_BASE_URL",
            "SEARXNG_ENABLED",
            "SEARXNG_TIMEOUT_SEC",
            "SEARXNG_DEFAULT_ENGINES",
            "SEARXNG_CLIENT_IP",
            "SEARXNG_DISCOVERY_CATEGORIES",
            "KE_API_HOST",
            "KE_API_PORT",
            "KE_API_BASE",
            "KE_API_RELOAD",
            "KE_NODE_DIVE_TIMEOUT_SEC",
            "KE_NODE_DIVE_ASYNC_TIMEOUT_SEC",
        ],
    ),
    (
        "Redis / worker",
        [
            "REDIS_URL",
            "REDIS_SOCKET_TIMEOUT_SEC",
            "KE_USE_REDIS",
            "KE_REDIS_LOGS",
            "KE_TASKS_CHANNEL",
            "KE_REDIS_LOG_MAX_LINES",
            "KE_WORKER_POLL_SEC",
            "KE_WORKER_HEARTBEAT_SEC",
            "KE_WORKER_STALE_RUNNING_SEC",
            "KE_WORKER_INLINE_FALLBACK",
            "KE_WORKER_RELOAD_DEBOUNCE_SEC",
            "KE_WORKER_STOP_TIMEOUT_SEC",
        ],
    ),
    (
        "Logging / trace",
        ["KE_TRACE_STDOUT", "KE_LOG_PLAIN", "KE_LLM_FULL_TRACE"],
    ),
    (
        "Ollama local",
        [
            "OLLAMA_BASE_URL",
            "OLLAMA_AUTO_START",
            "LOCAL_ROUTER_MODEL",
            "LOCAL_HEAVY_MODEL",
            "LOCAL_L2_MODEL",
            "REACT_EVAL_MODEL",
            "MAIN_MODEL",
            "GUARDRAILS_OLLAMA_MODEL",
            "GUARDRAILS_MODEL",
            "CONTEXT_EVAL_MODEL",
            "CONTEXT_EVAL_NUM_PREDICT",
            "OLLAMA_ROUTER_NUM_CTX",
            "OLLAMA_HEAVY_NUM_CTX",
            "OLLAMA_NUM_CTX",
            "OLLAMA_ROUTER_KEEP_ALIVE",
            "OLLAMA_HEAVY_KEEP_ALIVE",
            "OLLAMA_NUM_PARALLEL",
            "OLLAMA_NUM_PREDICT",
            "OLLAMA_GUARDRAILS_NUM_PREDICT",
            "OLLAMA_STRUCTURE_NUM_PREDICT",
            "SELECTION_PROMPTS_OLLAMA_MODEL",
            "SELECTION_PROMPTS_TIMEOUT_SEC",
            "SELECTION_PROMPTS_KEEP_ALIVE",
            "SELECTION_PROMPTS_NUM_PREDICT",
            "ARTICLE_DIAGRAM_FILTER_OLLAMA_MODEL",
            "ARTICLE_DIAGRAM_FILTER_TIMEOUT_SEC",
            "ARTICLE_DIAGRAM_FILTER_NUM_PREDICT",
            "ARTICLE_DIAGRAM_FILTER_NUM_CTX",
            "ARTICLE_MAX_DIAGRAMS_PER_ARTICLE",
            "COMPETENCY_EXTRACT_OLLAMA_MODEL",
            "COMPETENCY_EXTRACT_TIMEOUT_SEC",
            "COMPETENCY_EXTRACT_NUM_PREDICT",
        ],
    ),
    (
        "Blog spatial / triage / Gemma cloud MAP",
        [
            "BLOG_SPATIAL_MAP_PROVIDER",
            "BLOG_SPATIAL_SUMMARIZER_MODEL",
            "BLOG_SPATIAL_NUM_CTX",
            "BLOG_SPATIAL_NUM_PREDICT",
            "BLOG_SPATIAL_TIMEOUT_SEC",
            "BLOG_SPATIAL_MAP_MAX_TOKENS",
            "BLOG_SPATIAL_OVERLAP_TOKENS",
            "BLOG_SPATIAL_MAP_CONCURRENCY",
            "BLOG_SPATIAL_TRIAGE_ENABLED",
            "GEMMA_API_BASE",
            "GEMMA_API_KEY",
            "GEMMA_MODEL_NAME",
            "GEMMA_PRIMARY_MODEL",
            "GEMMA_FALLBACK_MODEL",
            "GEMMA_MAX_RPM",
            "GEMMA_MAX_TPM",
            "GEMMA_MAX_RPD",
            "GEMMA_PRIMARY_MAX_RPM",
            "GEMMA_PRIMARY_MAX_TPM",
            "GEMMA_PRIMARY_MAX_RPD",
            "GEMMA_FALLBACK_MAX_RPM",
            "GEMMA_FALLBACK_MAX_TPM",
            "GEMMA_FALLBACK_MAX_RPD",
            "GEMMA_CONCURRENCY",
            "GEMMA_MAP_MAX_OUTPUT_TOKENS",
            "GEMMA_EST_OUTPUT_TOKENS",
            "GEMMA_API_TIMEOUT_SEC",
            "GEMMA_FALLBACK_MAX_WAIT_SEC",
            "VLM_GEMINI_MODEL",
            "VLM_GEMINI_MODELS",
            "GEMINI_FLASH_LITE_MAX_RPM",
            "GEMINI_FLASH_LITE_MAX_TPM",
            "GEMINI_FLASH_LITE_MAX_RPD",
            "VLM_GEMINI_MAX_RPM",
            "VLM_GEMINI_MAX_TPM",
            "VLM_GEMINI_MAX_RPD",
            "VLM_GEMINI_CONCURRENCY",
            "VLM_GEMINI_EST_INPUT_TOKENS",
            "VLM_GEMINI_EST_OUTPUT_TOKENS",
            "VLM_GEMINI_QUOTA_TRACK",
        ],
    ),
    (
        "Gemini API",
        [
            "GEMINI_API_KEY",
            "GEMINI_API_KEYS",
            "GOOGLE_API_KEY",
            "SKIP_GEMINI",
            "GEMINI_PRIMARY",
            "REQUIRE_GEMINI",
            "GEMINI_MODEL",
            "GEMINI_REASONER_MODEL",
            "GEMINI_LITE_MODEL",
            "GEMINI_FLASH_MODEL",
            "GEMINI_TUTOR_MODEL",
            "GEMINI_HIGH_QUOTA_MODEL",
            "GEMINI_LITE_FALLBACK_MODELS",
            "GEMINI_FALLBACK_MODELS",
            "GEMINI_FALLBACK_MODEL",
            "GEMINI_TUTOR_FALLBACK_MODELS",
            "GEMINI_GROUNDING_MODEL",
            "CURRICULUM_GEMINI_GROUNDING_MODEL",
            "GEMINI_API_TIMEOUT_SEC",
            "GEMINI_TUTOR_TIMEOUT_SEC",
            "GEMINI_PROBE_TIMEOUT_SEC",
            "GEMINI_PROBE_BEFORE_USE",
            "GEMINI_QUOTA_TRACK",
            "GEMINI_RPM_PAUSE_SEC",
            "GEMINI_RPM_JITTER_SEC",
            "GEMINI_RPM_BLOCK_SEC",
            "GEMINI_RETRY_BACKOFF_SEC",
            "KE_RAG_TIMEOUT_SEC",
            "GEMINI_RESPONSE_MAX_SEC",
            "GEMINI_RESPONSE_FIRST_TIMEOUT_SEC",
            "GEMINI_STREAM_POLL_SEC",
            "GEMINI_STREAM_STABLE_ROUNDS",
            "GEMINI_MIN_RESPONSE_CHARS",
            "GEMINI_PAYLOAD_MAX_CHARS",
            "GEMINI_BROWSER_HEADLESS",
            "GEMINI_SEND_SELECTORS",
            "ROLLING_SUMMARY_MAX_CHARS",
        ],
    ),
    (
        "Curriculum / search / Exa",
        [
            "EXA_API_KEY",
            "EXA_SEARCH_ENABLED",
            "CURRICULUM_PRACTICAL_EXA_LIMIT",
            "EXA_DOMAIN_CAP_PER_HOST",
            "EXA_RERANK_LITE_THRESHOLD",
            "EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN",
            "EXA_RECALL_MAX_PER_DOMAIN",
            "EXA_FETCH_NUM_RESULTS",
            "EXA_MAX_CONCURRENT_SEARCH",
            "EXA_DUAL_QUERY_EN_RATIO",
            "EXA_EXCLUDE_TEXT",
            "EXA_PRACTICAL_HIGHLIGHT_QUERY",
            "EXCLUDED_SOURCES_BLACKLIST",
            "CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED",
            "CURRICULUM_SEARCH_FIRST_ENABLED",
            "CURRICULUM_SEARCH_TARGET_HITS",
            "CURRICULUM_SEARCH_MIN_HITS",
            "CURRICULUM_SEARCH_PROBE_URLS",
            "CURRICULUM_OPEN_SEARCH_QUERY_CONCURRENCY",
            "CURRICULUM_MODEL_FIRST_MIN_NODES",
            "CURRICULUM_MODEL_FIRST_TARGET_NODES",
            "CURRICULUM_LITE_BATCH_STRICT",
            "CURRICULUM_DEEP_NODE_MAX_HITS",
            "CURRICULUM_DEEP_HYBRID_PRACTICAL_FIRST",
            "CURRICULUM_USE_V08_CONSENSUS",
            "CURRICULUM_CONSENSUS_PRIMARY",
            "CURRICULUM_CONSENSUS_MIN_APPROVED_ACADEMIC",
            "CURRICULUM_V08_MAX_PAPERS",
            "CURRICULUM_V08_PAPER_POOL_SIZE",
            "CURRICULUM_ON_DEMAND_V08_MAX_PAPERS",
            "CURRICULUM_ON_DEMAND_V08_POOL_SIZE",
            "CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC",
            "CURRICULUM_ON_DEMAND_MIN_PRACTICAL_FOR_FAST_RETURN",
            "ACADEMIC_FAST_FETCH_TIMEOUT_SEC",
            "ACADEMIC_SCIHUB_TIMEOUT_SEC",
            "CURRICULUM_GEMINI_GROUNDING_ENABLED",
            "CURRICULUM_GEMINI_GROUNDING_MAX_URLS",
            "CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS",
            "CURRICULUM_GEMINI_WEB_HARVEST_ENABLED",
            "CURRICULUM_GEMINI_WEB_HARVEST_TIMEOUT_SEC",
            "CURRICULUM_GEMINI_WEB_RESPONSE_MAX_SEC",
            "CURRICULUM_GEMINI_WEB_RESPONSE_FIRST_TIMEOUT_SEC",
            "CURRICULUM_GEMINI_WEB_URL_RETRY_MAX",
            "CURRICULUM_URL_VALIDATE_TIMEOUT_SEC",
            "GOOGLE_CSE_API_KEY",
            "GOOGLE_CSE_ID",
            "CURRICULUM_GOOGLE_CSE_ENABLED",
            "GOOGLE_CSE_DAILY_LIMIT",
            "SEMANTIC_SCHOLAR_DAILY_LIMIT",
            "CURRICULUM_API_QUOTA_TRACK",
            "CURRICULUM_PRACTICAL_CSE_LIMIT",
            "CURRICULUM_PRACTICAL_DDGS_LIMIT",
            "CURRICULUM_PRACTICAL_DDGS_ENABLED",
            "CURRICULUM_PRACTICAL_SEARXNG_LIMIT",
            "CURRICULUM_PRACTICAL_SEARXNG_QUERIES",
            "CURRICULUM_PRACTICAL_SEARXNG_ENGINES",
            "CURRICULUM_PRACTICAL_SEARXNG_CATEGORIES",
            "CURRICULUM_ACADEMIC_SEARXNG_LIMIT",
            "CURRICULUM_LITE_BATCH_EVAL_FALLBACK_N",
            "CURRICULUM_LITE_SITE_SUGGEST_ENABLED",
            "CURRICULUM_PRACTICAL_SNIPPET_MIN_CHARS",
            "CURRICULUM_ACADEMIC_SS_LIMIT",
            "CURRICULUM_ACADEMIC_ARXIV_LIMIT",
            "CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS",
            "ARXIV_MIN_INTERVAL_SEC",
            "ARXIV_MAX_RETRIES",
            "ARXIV_BACKOFF_BASE_SEC",
            "ARXIV_ID_LIST_CHUNK",
        ],
    ),
    (
        "Consensus Playwright",
        [
            "CONSENSUS_START_URL",
            "CONSENSUS_MAX_RETRIES",
            "CONSENSUS_BROWSER_HEADLESS",
            "CONSENSUS_FORCE_HEADED",
            "CONSENSUS_REUSE_BROWSER_SESSION",
            "CONSENSUS_NEW_THREAD_EACH_RUN",
            "CONSENSUS_CLOSE_AFTER_EACH_HARVEST",
            "CONSENSUS_AUTH_RECOVERY_CYCLES",
            "CONSENSUS_BOOTSTRAP_INPUT_TIMEOUT_SEC",
            "CONSENSUS_NEW_DIALOG_MAX_WAIT_SEC",
            "CONSENSUS_UI_POLL_SEC",
            "CONSENSUS_PAPER_HARVEST_PASSES",
            "CONSENSUS_PAPER_HARVEST_PAUSE_SEC",
            "CONSENSUS_USE_QUICK_PAPER_SEARCH",
            "CONSENSUS_QUICK_BASE_URL",
            "CONSENSUS_QUICK_OPEN_ACCESS",
            "CONSENSUS_QUICK_LOAD_MORE_CLICKS",
            "CONSENSUS_QUICK_RESULTS_MAX_WAIT_SEC",
            "CONSENSUS_INPUT_SELECTOR",
            "CONSENSUS_RESPONSE_SELECTOR",
            "CONSENSUS_SEND_SELECTORS",
            "CONSENSUS_RESPONSE_MAX_SEC",
            "CONSENSUS_RESPONSE_FIRST_TIMEOUT_SEC",
            "CONSENSUS_STREAM_POLL_SEC",
            "CONSENSUS_STREAM_STABLE_ROUNDS",
            "CONSENSUS_MIN_RESPONSE_CHARS",
        ],
    ),
    (
        "RAG / lecture / personal",
        [
            "RAG_CROSS_ENCODER_MODEL",
            "RAG_CE_TORCH_DTYPE",
            "RAG_CE_AUTO_UNLOAD",
            "RAG_CE_AUTO_UNLOAD_IDLE_SEC",
            "RAG_DEFAULT_MIN_RELEVANCE",
            "RAG_DEFAULT_MAX_FACTS",
            "RAG_RETRIEVAL_PER_DIRECTION",
            "RAG_LATENCY_WARN_MS",
            "RAG_FACT_COMPRESS_GEMMA_TIMEOUT_SEC",
            "RAG_GATEWAY_FINISH_MARGIN_SEC",
            "KE_PROMPT_CONTEXT_OVERRIDE_LOCAL_MAC",
            "KE_PROMPT_CONTEXT_OVERRIDE_FULLSTACK",
            "RAG_HYBRID_LIMIT",
            "RAG_MIN_RELEVANT_HITS",
            "LIGHT_RAG_MIN_COSINE_SIM",
            "LIGHT_RAG_PROFILE_LIMIT",
            "LECTURE_RAG_TOP_K",
            "LECTURE_RAG_CANDIDATE_LIMIT",
            "LECTURE_RAG_MMR_TOP_K",
            "LECTURE_RAG_CE_MIN_SCORE",
            "LECTURE_RAG_CONTEXT_MAX_CHARS",
            "LECTURE_RAG_MMR_LAMBDA",
            "LECTURE_RAG_RERANK_TIMEOUT_SEC",
            "LECTURE_RAG_COLLECT_TIMEOUT_SEC",
            "LECTURE_RAG_LIGHT_TIMEOUT_SEC",
            "LECTURE_RAG_KNODE_CANDIDATE_LIMIT",
            "CHAT_SESSION_API_TURNS_MAX",
            "LECTURE_EXTERNAL_SEARCH_ENABLED",
            "LECTURE_EXTERNAL_SEARCH_TOP_K",
            "MAX_EXTERNAL_SOURCES",
            "LECTURE_MIN_LOCAL_SOURCES",
            "LECTURE_LOCAL_QUALITY_THRESHOLD",
            "LOCAL_QUALITY_THRESHOLD",
            "DIALOG_ATOMS_ENABLED",
            "DIALOG_ATOMS_TOP_K",
            "DIALOG_ATOMS_MIN_SCORE",
            "LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC",
            "NODE_DIVE_LECTURE_RAG_TIMEOUT_SEC",
            "NODE_DIVE_LECTURE_SEARCH_TIMEOUT_SEC",
            "LECTURE_MAX_OUTPUT_TOKENS",
            "GEMINI_TUTOR_MAX_OUTPUT_TOKENS",
            "GEMINI_INTRO_MAX_OUTPUT_TOKENS",
            "GEMINI_LITE_MAX_OUTPUT_TOKENS",
            "LECTURE_GENERATION_TEMPERATURE",
            "LECTURE_GENERATION_TIMEOUT_SEC",
            "LECTURE_MIN_WORDS_TARGET",
            "KE_POOL_LIGHT_WORKERS",
            "KE_POOL_RAG_IO_WORKERS",
            "KE_POOL_RAG_CE_WORKERS",
            "KE_POOL_LLM_SYNC_WORKERS",
            "KE_POOL_LLM_LECTURE_WORKERS",
            "KE_POOL_NET_SYNC_WORKERS",
        ],
    ),
    (
        "Analyze pipeline / discovery",
        [
            "MAX_FETCH_URLS",
            "MULTI_SEARCH_SKIP_VISION",
            "MAX_LANCE_INDEX_URLS",
            "MAX_AI_DIALOGUE_TURNS",
            "MAX_RESEARCH_SOURCES",
            "MAX_URLS",
            "MIN_VALIDATED_SOURCES",
            "MAX_RESEARCH_FIND_ROUNDS",
            "MAX_RESEARCH_DEPTH",
            "CLARIFY_VIA_GEMINI",
            "MIN_PAGE_CHARS_FOR_EXTRACTION",
            "DISCOVERY_MODE",
            "SMART_QUERY_SYNTAX_ENABLED",
            "DOMAIN_TRUST_ENABLED",
            "DOMAIN_TRUST_MIN_SCORE",
            "DOMAIN_TRUST_BATCH_SIZE",
            "DOMAIN_TRUST_HIGH_SCORE",
            "DOMAIN_TRUST_DB_PATH",
            "SOURCE_ARCHIVE_ENABLED",
            "SOURCE_ARCHIVE_DB_PATH",
            "UNPAYWALL_EMAIL",
            "OPENALEX_MAILTO",
            "OPENALEX_TIMEOUT_SEC",
            "OPENALEX_TRUST_ENABLED",
            "OPENALEX_DAILY_LIMIT",
            "OPENALEX_CONCURRENCY",
            "RAG_TRUST_HARD_CUTOFF",
            "RAG_TRUST_HARD_MIN_TRUST",
            "RAG_TRUST_HARD_MIN_SIM",
            "ACADEMIC_RERANK_ENABLED",
            "ACADEMIC_RERANK_WEIGHTS",
            "ACADEMIC_RERANK_C_SAT",
            "ACADEMIC_RERANK_RECENCY_HALF_LIFE_YEARS",
            "ACADEMIC_RELAXATION_ENABLED",
            "ACADEMIC_RELAXATION_MIN_HITS",
            "ACADEMIC_RELAX_L0_MIN_TRUST",
            "ACADEMIC_RELAX_L0_MIN_CITATIONS",
            "ACADEMIC_RELAX_L1_MIN_TRUST",
            "ACADEMIC_RELAX_L1_MIN_CITATIONS",
            "ACADEMIC_RELAX_L1_YEAR_PAD",
            "URL_BLOCKLIST_SUBSTR",
            "SEMANTIC_SCHOLAR_ENABLED",
            "SEMANTIC_SCHOLAR_API_KEY",
            "SEMANTIC_SCHOLAR_LIMIT",
            "SEMANTIC_SCHOLAR_TIMEOUT_SEC",
            "SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC",
            "SEMANTIC_SCHOLAR_429_BACKOFF_SEC",
            "SEMANTIC_SCHOLAR_ENRICH_TIMEOUT_SEC",
            "SUMMARIZER_MAX_INPUT_CHARS",
            "SUMMARIZER_MAX_PROFILE_CHARS",
            "MATRIX_MAX_SUMMARY_CHARS",
            "PLAYWRIGHT_BROWSERS_PATH",
            "PLAYWRIGHT_BROWSER",
        ],
    ),
]

# Ручные дефолты (сложные выражения / не строковые литералы в config.py)
MANUAL_DEFAULTS: dict[str, str] = {
    "GRAPH_VERSION": "0.4",
    "LOCAL_ROUTER_MODEL": ROUTER,
    "LOCAL_HEAVY_MODEL": MAIN,
    "LOCAL_L2_MODEL": MAIN,
    "REACT_EVAL_MODEL": ROUTER,
    "MAIN_MODEL": MAIN,
    "CONTEXT_EVAL_MODEL": ROUTER,
    "SELECTION_PROMPTS_OLLAMA_MODEL": ROUTER,
    "SELECTION_PROMPTS_KEEP_ALIVE": "5m",
    "BLOG_SPATIAL_SUMMARIZER_MODEL": MAIN,
    "ARTICLE_DIAGRAM_FILTER_OLLAMA_MODEL": MAIN,
    "COMPETENCY_EXTRACT_OLLAMA_MODEL": ROUTER,
    "GUARDRAILS_OLLAMA_MODEL": MAIN,
    "GEMINI_FLASH_MODEL": "gemini-3.1-flash-lite",
    "GEMINI_TUTOR_MODEL": "gemini-3.1-flash-lite",
    "GEMINI_REASONER_MODEL": "gemini-3.6-flash",
    "CURRICULUM_GEMINI_GROUNDING_MODEL": "gemini-2.5-flash",
    "GEMMA_PRIMARY_MODEL": "gemma-4-31b-it",
    "GEMMA_PRIMARY_MAX_RPM": "30",
    "GEMMA_PRIMARY_MAX_TPM": "16000",
    "GEMMA_PRIMARY_MAX_RPD": "14400",
    "GEMMA_FALLBACK_MAX_RPM": "30",
    "GEMMA_FALLBACK_MAX_TPM": "16000",
    "GEMMA_FALLBACK_MAX_RPD": "14400",
    "GEMMA_EST_OUTPUT_TOKENS": "2048",
    "VLM_GEMINI_MODEL": "gemini-3.5-flash-lite",
    "VLM_GEMINI_MODELS": "gemini-3.5-flash-lite,gemini-3.1-flash-lite",
    "GEMINI_FLASH_LITE_MAX_RPM": "14",
    "GEMINI_FLASH_LITE_MAX_TPM": "250000",
    "GEMINI_FLASH_LITE_MAX_RPD": "490",
    "VLM_GEMINI_MAX_RPM": "14",
    "VLM_GEMINI_MAX_TPM": "250000",
    "VLM_GEMINI_MAX_RPD": "490",
    "VLM_GEMINI_CONCURRENCY": "3",
    "VLM_GEMINI_EST_INPUT_TOKENS": "12000",
    "VLM_GEMINI_EST_OUTPUT_TOKENS": "1024",
    "VLM_GEMINI_QUOTA_TRACK": "true",
    "KE_USE_REDIS": "false",
    "KE_REDIS_LOGS": "false",
    "KE_NODE_DIVE_ASYNC_TIMEOUT_SEC": "600",
    "KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC": "900",
    "KE_INGEST_URL_CONCURRENCY": "4",
    "ACADEMIC_INGEST_MAX_BODY_CHARS": "80000",
    "KE_API_BASE": "http://127.0.0.1:8765",
    "MAX_URLS": "5",
    "GRAPH_RECURSION_LIMIT": "0",
    "OLLAMA_NUM_PARALLEL": "1",
    "LECTURE_RAG_CE_MIN_SCORE": "0.48",
    "CURRICULUM_GEMINI_WEB_HARVEST_ENABLED": "true",
    "CONSENSUS_INPUT_SELECTOR": (
        "textarea[data-testid='new-thread-input'],textarea[data-testid='search-input'],"
        "textarea, div[contenteditable='true']"
    ),
    "CONSENSUS_RESPONSE_SELECTOR": (
        "[data-testid='answer'], [data-testid*='message'], [class*='Answer'], "
        "article .prose, main article"
    ),
    "CONSENSUS_SEND_SELECTORS": (
        "button[data-testid='search-button'],"
        "[data-testid='search-input-form'] button[type='submit'],"
        "button[aria-label='Submit search'],"
        "button[aria-label*='Submit search'],"
        "button[type='submit']"
    ),
    "GEMINI_SEND_SELECTORS": (
        "button[aria-label*='Send'],button[aria-label*='Отправ'],"
        "[data-test-id='send-button'],button.send-button"
    ),
    "EXA_EXCLUDE_TEXT": "api reference documentation sdk classes",
    "EXA_PRACTICAL_HIGHLIGHT_QUERY": (
        "Engineering blog deep dive: system architecture, implementation trade-offs, "
        "failure modes, benchmarks — not API parameter lists or SDK setup steps."
    ),
    "DOMAIN_TRUST_DB_PATH": "knowledge_engine/.domain_trust/domains.sqlite",
    "SOURCE_ARCHIVE_DB_PATH": "knowledge_engine/.source_archive/links.sqlite",
}


def scan_env_keys() -> set[str]:
    keys: set[str] = set()
    for p in KE.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _PAT_KEY.finditer(text):
            k = m.group(1) or m.group(2)
            if k:
                keys.add(k)
    keys.add("GRAPH_VERSION")
    keys.add("OLLAMA_NUM_PARALLEL")
    return keys


def parse_config_defaults() -> dict[str, str]:
    text = CONFIG.read_text(encoding="utf-8")
    out: dict[str, str] = dict(MANUAL_DEFAULTS)
    for m in _PAT_STR_DEFAULT.finditer(text):
        out[m.group(1)] = m.group(2)
    for m in _PAT_BOOL_DEFAULT.finditer(text):
        out[m.group(1)] = m.group(2).lower()
    for m in _PAT_BOOL_OS.finditer(text):
        out[m.group(1)] = m.group(2).lower()
    # os.getenv("KEY", "true").lower() in (...)
    for m in re.finditer(
        r"os\.getenv\(\s*['\"]([A-Z0-9_]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\)\.lower\(\)\s+in",
        text,
    ):
        out[m.group(1)] = m.group(2).lower()
    for key in SECRET_KEYS:
        if key not in out:
            out[key] = ""
    return out


def default_for(key: str, defaults: dict[str, str]) -> str:
    if key in defaults:
        return defaults[key]
    return ""


def format_line(key: str, value: str, commented: bool) -> str:
    body = f"{key}={value}"
    if commented:
        return f"# {body}"
    return body


def build_example_text(all_keys: set[str], defaults: dict[str, str]) -> str:
    lines = [
        "# Knowledge Engine — каталог переменных (дефолты из config.py)",
        "# Документация: knowledge_engine/docs/ENV_VARIABLES.md",
        "# Синхронизация: python knowledge_engine/scripts/sync_env_catalog.py --write-example",
        "# Секреты: пустые значения; задайте в .env (не коммитить).",
        "",
    ]
    seen: set[str] = set()
    for title, keys in SECTIONS:
        lines.append(f"# --- {title} ---")
        for key in keys:
            if key not in all_keys:
                continue
            seen.add(key)
            val = default_for(key, defaults)
            if key in SECRET_KEYS:
                lines.append(format_line(key, "", commented=False))
            else:
                lines.append(format_line(key, val, commented=False))
        lines.append("")

    extra = sorted(all_keys - seen)
    if extra:
        lines.append("# --- Other (scan) ---")
        for key in extra:
            val = default_for(key, defaults)
            if key in SECRET_KEYS:
                lines.append(format_line(key, "", commented=False))
            else:
                lines.append(format_line(key, val, commented=False))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_active_env_keys(path: Path) -> set[str]:
    active: set[str] = set()
    if not path.is_file():
        return active
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            active.add(s.split("=", 1)[0].strip())
    return active


def strip_catalog_block(text: str) -> str:
    if _MARKER_START not in text:
        return text.rstrip()
    before, rest = text.split(_MARKER_START, 1)
    if _MARKER_END in rest:
        _, after = rest.split(_MARKER_END, 1)
        return (before.rstrip() + after).rstrip()
    return before.rstrip()


def build_catalog_block(
    miss_keys: list[str],
    defaults: dict[str, str],
) -> str:
    lines = ["", _MARKER_START]
    for k in miss_keys:
        val = default_for(k, defaults)
        lines.append(f"# {k}={val}")
    lines.append(_MARKER_END)
    return "\n".join(lines) + "\n"


def enrich_commented_defaults_in_body(text: str, defaults: dict[str, str]) -> str:
    """# KEY= без значения → # KEY=default (не трогаем активные строки)."""
    out_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") and "=" in s:
            body = s.lstrip("#").strip()
            key, _, val = body.partition("=")
            key = key.strip()
            if key and not val.strip() and key in defaults:
                out_lines.append(f"# {key}={defaults[key]}")
                continue
        out_lines.append(line)
    return "\n".join(out_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-example",
        action="store_true",
        help="Перезаписать .env.example с дефолтами",
    )
    parser.add_argument(
        "--merge-env",
        action="store_true",
        help="Обновить каталог и commented-дефолты в .env",
    )
    args = parser.parse_args()

    all_keys = scan_env_keys()
    defaults = parse_config_defaults()

    in_example = parse_active_env_keys(ENV_EXAMPLE)
    active_env = parse_active_env_keys(ENV_FILE)
    miss_example = sorted(all_keys - in_example)
    miss_env = sorted(all_keys - active_env)

    print(f"catalog keys: {len(all_keys)}")
    print(f"defaults parsed: {len(defaults)}")
    if ENV_EXAMPLE.is_file():
        print(
            f".env.example active keys: {len(in_example)} | missing: {len(miss_example)}"
        )
    print(
        f".env active keys: {len(active_env)} | not active (catalog): {len(miss_env)}"
    )

    if args.write_example:
        ENV_EXAMPLE.write_text(build_example_text(all_keys, defaults), encoding="utf-8")
        print(f"Wrote {ENV_EXAMPLE}")

    if args.merge_env:
        raw = ENV_FILE.read_text(encoding="utf-8", errors="ignore")
        base = strip_catalog_block(raw)
        base = enrich_commented_defaults_in_body(base, defaults)
        ENV_FILE.write_text(
            base + build_catalog_block(miss_env, defaults),
            encoding="utf-8",
        )
        print(f"Updated {ENV_FILE} (catalog {len(miss_env)} keys)")

    if not args.write_example and not args.merge_env:
        if miss_example[:5]:
            print("sample missing in example:", miss_example[:5])
        if miss_env[:5]:
            print("sample missing in .env active:", miss_env[:5])
        print("Use --write-example and/or --merge-env")

    return 0


if __name__ == "__main__":
    sys.exit(main())
