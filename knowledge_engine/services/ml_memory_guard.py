"""Общий guard для локальных ML-моделей RAG-пайплайна (BGE-M3 bi-encoder,
RAG Cross-Encoder): async-прогрев в порядке вызова, request-scoped 5-минутный
cooldown и OS-agnostic экстренная выгрузка по превышению RAM.

Предыстория (аудит "MEMORY OPTIMIZATION: DIAGNOSE & FIX 8.5GB RAM"): `ps`'s
RSS на macOS систематически занижает реальный footprint процесса —
Metal/MPS GPU-shared буферы учитываются как `IOAccelerator (graphics)` в
Activity Monitor, а НЕ в `ps RSS` (и часто не в `torch.mps.*` счётчиках под
нагрузкой — на живом прогоне `driver_allocated_memory()` ни разу не превысил
3.5GB, пока `footprint -s <pid>` показывал пик 6.9-7.1GB). `empty_cache()` не
освобождает вес уже загруженной модели — единственный способ реально
опуститься ниже порога — выгрузить модель целиком.

Эта версия (REFACTOR: ASYNC PIPELINE WARMUP & GLOBAL 5-MIN COOLDOWN) вводит
ДВА независимых механизма выгрузки вместо одного:

1. **Request-scoped cooldown** (``RAG_MPS_REQUEST_COOLDOWN_SEC``, 300s по
   умолчанию) — идёт от момента ПОЛНОГО завершения всего RAG-запроса
   (``rag_request_finished()``, когда счётчик активных запросов дошёл до
   нуля), не от последнего вызова конкретной модели. Каждый новый запрос
   (``rag_request_started()``) отменяет и сбрасывает таймер. По истечении —
   выгружаются ВСЕ зарегистрированные модели безусловно, не только те, что
   превышают порог RAM: раз простой подтверждён на уровне всего пайплайна,
   держать веса незачем.
2. **Emergency threshold override** (``RAG_MPS_MEMORY_THRESHOLD_GB``) —
   реактивная проверка после каждого embed/rerank (``guard_after_use``):
   если footprint ПРЯМО СЕЙЧАС выше порога (даже посреди активного запроса),
   немедленно выгружает модели, не занятые этим самым вызовом, не дожидаясь
   cooldown. Не трогает модели, которые были только прогреты
   (``register_model``), но ещё ни разу не использовались реально
   (``guard_after_use`` для них не звался) — прогретую-но-не-использованную
   модель эта проверка не выгружает, чтобы не сводить на нет пользу
   прогрева; такую модель уберёт только cooldown-сборка (1), если запрос
   так и не подтвердит её нужность.

Плюс async warmup (``warmup_pipeline_async``) — параллельный остальным
этапам RAG-запроса прогрев моделей В ТОМ ЖЕ ПОРЯДКЕ, в котором они реально
вызываются по пайплайну (Embedding → Rerank/Cross-Encoder), чтобы к моменту
реального обращения модель уже была прогрета. Загрузка — последовательная
(не ``asyncio.gather``), чтобы не создавать двойной пиковый всплеск RAM от
одновременной инициализации обеих моделей.

OS-agnostic footprint (``current_phys_footprint_mb``): на Darwin — сначала
`footprint -s <pid>` (единственный источник, подтверждённый эмпирически как
совпадающий с Activity Monitor; psutil RSS/USS систематически занижает
Metal/MPS буферы на macOS). На остальных ОС (и как fallback, если бинарник
`footprint` недоступен) — ``psutil.Process().memory_full_info().uss``, с
graceful fallback на ``memory_info().rss``, если USS недоступен на данной
платформе/правах.
"""

from __future__ import annotations

import asyncio
import gc
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable

from knowledge_engine.config import (
    RAG_MPS_MEMORY_THRESHOLD_GB,
    RAG_MPS_REQUEST_COOLDOWN_SEC,
)
from knowledge_engine.ui.run_log import trace

_lock = threading.RLock()  # реентрантный: _cooldown_fired выгружает модели,
# держа этот лок — обычный Lock самозаблокировался бы, если бы unload-колбэк
# когда-нибудь вызвал register_model/rag_request_started из того же потока.
_last_use: dict[str, float] = {}
_unloaders: dict[str, Callable[[], None]] = {}
_last_guard_check_monotonic = 0.0
_MIN_CHECK_INTERVAL_SEC = (
    5.0  # не чаще раза в 5с — не добавлять overhead на каждый вызов
)

_active_requests = 0
_cooldown_timer: threading.Timer | None = None


def register_model(name: str, unload_fn: Callable[[], None]) -> None:
    """Зарегистрировать модель и её unload-callback (вызывать один раз при
    первой загрузке модели — идемпотентно перезаписывает)."""
    with _lock:
        _unloaders[name] = unload_fn


def release_mps_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def current_mps_driver_mb() -> float:
    """torch's own счётчик — оставлен для диагностики в трейсах; НЕ
    используется для порогового решения (под реальной нагрузкой расходится
    с настоящим footprint в разы, см. модуль-докстринг)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def _footprint_darwin_mb() -> float | None:
    try:
        out = subprocess.run(
            ["footprint", "-s", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("phys_footprint:"):
                parts = s.split()
                val = float(parts[1])
                if len(parts) > 2 and parts[2].upper() == "GB":
                    val *= 1024
                return val
    except Exception:
        pass
    return None


def _psutil_mb() -> float | None:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        try:
            return proc.memory_full_info().uss / 1024 / 1024
        except Exception:
            return proc.memory_info().rss / 1024 / 1024
    except Exception:
        return None


def current_phys_footprint_mb() -> float:
    """Реальный физический footprint процесса, OS-agnostic. На Darwin —
    сначала `footprint -s <pid>` (см. модуль-докстринг: psutil RSS/USS
    занижает Metal/MPS GPU-shared буферы, которых `footprint` не пропускает);
    при недоступности бинарника — psutil. На остальных ОС — сразу psutil
    (USS, иначе RSS)."""
    if platform.system() == "Darwin":
        mb = _footprint_darwin_mb()
        if mb is not None:
            return mb
    mb = _psutil_mb()
    if mb is not None:
        return mb
    return 0.0


def _check_and_evict(*, exempt: str | None, reason: str) -> None:
    """Экстренная (threshold-based) выгрузка: выгружает наименее недавно
    использованную ИЗ РЕАЛЬНО ИСПОЛЬЗОВАННЫХ моделей (``_last_use``), не
    трогая ``exempt`` (модель, занятую прямо сейчас) и не трогая модели,
    которые были только прогреты, но ещё ни разу не использованы (их в
    ``_last_use`` нет — прогрев не в счёт, см. модуль-докстринг)."""
    release_mps_cache()
    threshold_mb = RAG_MPS_MEMORY_THRESHOLD_GB * 1024
    mb = current_phys_footprint_mb()
    if mb <= threshold_mb:
        return

    trace(
        f"ML_MEMORY_GUARD ⚠ phys_footprint={mb:.0f}MB > "
        f"порог {threshold_mb:.0f}MB ({reason}, mps_driver="
        f"{current_mps_driver_mb():.0f}MB) — экстренный поиск свободной модели"
    )
    with _lock:
        candidates = sorted(
            (
                (n, ts)
                for n, ts in _last_use.items()
                if n != exempt and n in _unloaders
            ),
            key=lambda kv: kv[1],
        )
    for n, _ts in candidates:
        _unloaders[n]()
        release_mps_cache()
        mb = current_phys_footprint_mb()
        trace(
            f"ML_MEMORY_GUARD ✓ evicted {n} (over RAM threshold) | phys_footprint={mb:.0f}MB"
        )
        if mb <= threshold_mb:
            return
    if mb > threshold_mb:
        trace(
            f"ML_MEMORY_GUARD ⊘ всё ещё {mb:.0f}MB > порога — все зарегистрированные "
            "модели активно используются (постоянная нагрузка), выгружать нечего"
        )


def guard_after_use(name: str) -> None:
    """Вызывать сразу после тяжёлого RAG-вызова (embed / rerank) с именем
    только что использованной модели. Троттлится ``_MIN_CHECK_INTERVAL_SEC``
    (не чаще раза в 5с); при срабатывании чистит MPS allocator cache и, если
    footprint всё ещё выше порога, выгружает другую (не ``name``)
    зарегистрированную модель — экстренная мера, независимая от
    request-scoped cooldown (см. ``rag_request_finished``).

    Пока идёт хотя бы один активный запрос (``_active_requests > 0``) —
    полностью no-op: транзиентные пики RAM внутри запроса — норма, эта
    проверка не должна выгружать модели прямо посреди его выполнения
    (см. модуль CRITICAL BUGFIX: TOTAL EVICTION LOCK)."""
    global _last_guard_check_monotonic
    now = time.monotonic()
    with _lock:
        _last_use[name] = now
        if _active_requests > 0:
            return
        due = now - _last_guard_check_monotonic >= _MIN_CHECK_INTERVAL_SEC
        if due:
            _last_guard_check_monotonic = now
    if not due:
        return
    _check_and_evict(exempt=name, reason=f"после {name}")


# ---------------------------------------------------------------------------
# Request-scoped 5-минутный cooldown (требование 2 REFACTOR-задачи)
# ---------------------------------------------------------------------------


def rag_request_started() -> None:
    """Вызывать в самом начале обработки RAG-запроса. Увеличивает счётчик
    активных запросов и отменяет отложенную cooldown-выгрузку — новый запрос
    отменяет и пересчитывает таймер заново после своего завершения."""
    global _active_requests, _cooldown_timer
    with _lock:
        _active_requests += 1
        active = _active_requests
        if _cooldown_timer is not None:
            _cooldown_timer.cancel()
            _cooldown_timer = None
    trace(f"ML_MEMORY_GUARD ▶ RAG-запрос начат (active={active}) — cooldown отменён")


def rag_request_finished() -> None:
    """Вызывать по ПОЛНОМУ завершению обработки RAG-запроса (успех, ошибка
    или отмена — из ``finally``). Когда счётчик активных запросов доходит до
    нуля, взводит cooldown-таймер на ``RAG_MPS_REQUEST_COOLDOWN_SEC``."""
    global _active_requests
    with _lock:
        _active_requests = max(0, _active_requests - 1)
        should_schedule = _active_requests == 0
    if should_schedule:
        _schedule_cooldown()


def _schedule_cooldown() -> None:
    global _cooldown_timer
    with _lock:
        if _cooldown_timer is not None:
            _cooldown_timer.cancel()
        _cooldown_timer = threading.Timer(
            RAG_MPS_REQUEST_COOLDOWN_SEC, _cooldown_fired
        )
        _cooldown_timer.daemon = True
        _cooldown_timer.start()
    trace(
        f"ML_MEMORY_GUARD ⏱ RAG-запрос завершён (active=0) — cooldown "
        f"{RAG_MPS_REQUEST_COOLDOWN_SEC:.0f}s до безусловной idle-выгрузки"
    )


def _cooldown_fired() -> None:
    """Проверка ``_active_requests`` и сама выгрузка — ОДНА атомарная секция
    под ``_lock``, тем же локом, что и ``rag_request_started``. Раньше лок
    отпускался сразу после проверки, а выгрузка шла отдельно: если новый
    запрос стартовал ровно в этом окне (после проверки, до unload()), таймер
    успевал безусловно выгрузить модели прямо во время его выполнения — это
    и был баг из отчёта ("cooldown истёк ПРЯМО ВО ВРЕМЯ ВЫПОЛНЕНИЯ ЗАПРОСА").
    Держать лок на всё время unload() безопасно: ``rag_request_started`` либо
    успевает увеличить счётчик до этой проверки (тогда мы её здесь увидим и
    выйдем без выгрузки), либо блокируется на лок и увидит выгрузку уже
    свершившейся — оба исхода корректны, интерливинга между ними больше нет."""
    with _lock:
        if _active_requests > 0:
            return  # новый запрос успел стартовать и уже отменил этот таймер
        names = list(_unloaders.keys())
        if not names:
            return
        trace(
            f"ML_MEMORY_GUARD ⏱ cooldown {RAG_MPS_REQUEST_COOLDOWN_SEC:.0f}s истёк, "
            f"простой подтверждён — безусловно выгружаю все модели: {','.join(names)}"
        )
        for n in names:
            unload = _unloaders.get(n)
            if unload is not None:
                unload()
    release_mps_cache()
    mb = current_phys_footprint_mb()
    trace(f"ML_MEMORY_GUARD ✓ idle cooldown sweep завершён | phys_footprint={mb:.0f}MB")


class rag_request_scope:
    """Async context manager: ``async with rag_request_scope(): ...`` вокруг
    всего тела обработчика RAG-запроса — гарантирует парный
    started/finished даже при исключении/отмене."""

    async def __aenter__(self) -> "rag_request_scope":
        rag_request_started()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        rag_request_finished()


# ---------------------------------------------------------------------------
# Async warmup (требование 1 REFACTOR-задачи)
# ---------------------------------------------------------------------------

_DEFAULT_WARMUP_STAGES: tuple[str, ...] = ("bge_m3", "cross_encoder")
_background_warmup_tasks: set[asyncio.Task] = set()


def spawn_warmup_task(
    stages: tuple[str, ...] = _DEFAULT_WARMUP_STAGES,
) -> asyncio.Task:
    """``asyncio.create_task(warmup_pipeline_async())`` с защитой от
    сборки мусора до завершения задачи (частая asyncio-ловушка при
    fire-and-forget task'ах без сохранённой ссылки)."""
    task = asyncio.create_task(warmup_pipeline_async(stages))
    _background_warmup_tasks.add(task)
    task.add_done_callback(_background_warmup_tasks.discard)
    return task


async def warmup_pipeline_async(
    stages: tuple[str, ...] = _DEFAULT_WARMUP_STAGES,
) -> None:
    """Прогреть модели RAG-пайплайна асинхронно, ПОСЛЕДОВАТЕЛЬНО, в том же
    порядке, в котором они реально вызываются далее по пайплайну (Embedding
    → Rerank/Cross-Encoder) — чтобы JIT/интерпретаторский кэш прогревался в
    реальном порядке использования. Запускать через ``asyncio.create_task``
    в самом начале обработки RAG-запроса, параллельно остальным этапам
    (vector_search и т.д.), а не только Exa-стадии.

    Последовательная (не ``asyncio.gather``) загрузка — намеренно: две
    ~3GB+ MPS-модели, инициализируемые ОДНОВРЕМЕННО, создают недопустимый
    пиковый всплеск RAM (см. модуль-докстринг живого прогона: пик
    6.9-7.3GB при параллельной активной загрузке обеих моделей).
    Блокирующая инициализация каждой модели уходит в ``asyncio.to_thread``,
    поэтому event loop не блокируется, пока модели грузятся один за другим.
    """
    t0 = time.monotonic()
    trace(f"ML_MEMORY_GUARD ▶ async warmup старт | stages={','.join(stages)}")
    for stage in stages:
        try:
            if stage == "bge_m3":
                from knowledge_engine.services.search.bge_m3_embed import (
                    ensure_bge_m3_loaded,
                )

                await asyncio.to_thread(ensure_bge_m3_loaded)
            elif stage == "cross_encoder":
                from knowledge_engine.src.rag_gateway.cross_encoder import (
                    ensure_cross_encoder_loaded,
                )

                await asyncio.to_thread(ensure_cross_encoder_loaded)
            else:
                trace(f"ML_MEMORY_GUARD ⚠ warmup: неизвестный stage={stage}")
        except Exception as exc:
            trace(f"ML_MEMORY_GUARD ⚠ warmup[{stage}] failed | {exc}")
    trace(
        f"ML_MEMORY_GUARD ✓ async warmup завершён | {(time.monotonic() - t0) * 1000:.0f}ms"
    )
