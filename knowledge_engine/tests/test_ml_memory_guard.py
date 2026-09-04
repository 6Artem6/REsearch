"""ML_MEMORY_GUARD: request-scoped cooldown, emergency threshold override,
OS-agnostic footprint fallback chain, async pipeline warmup order/batching."""

from __future__ import annotations

import asyncio
import threading

import pytest

from knowledge_engine.services import ml_memory_guard as guard


@pytest.fixture(autouse=True)
def _reset_guard_state(monkeypatch):
    """Каждый тест — с чистого листа: своё состояние, свой threading.Timer
    не наследуется от соседних тестов и не переживает тест."""
    monkeypatch.setattr(guard, "_last_use", {})
    monkeypatch.setattr(guard, "_unloaders", {})
    monkeypatch.setattr(guard, "_last_guard_check_monotonic", 0.0)
    monkeypatch.setattr(guard, "_active_requests", 0)
    monkeypatch.setattr(guard, "_cooldown_timer", None)
    yield
    with guard._lock:
        if guard._cooldown_timer is not None:
            guard._cooldown_timer.cancel()
            guard._cooldown_timer = None


# ---------------------------------------------------------------------------
# Emergency threshold override (требование 4)
# ---------------------------------------------------------------------------


def test_guard_after_use_evicts_other_idle_model_over_threshold(monkeypatch):
    monkeypatch.setattr(guard, "RAG_MPS_MEMORY_THRESHOLD_GB", 1.0)
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 2048.0)
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)

    evicted = []
    guard.register_model("cross_encoder", lambda: evicted.append("cross_encoder"))
    guard._last_use["cross_encoder"] = 0.0  # уже реально использовалась

    guard.guard_after_use("bge_m3")

    assert evicted == ["cross_encoder"]


def test_guard_after_use_never_evicts_the_model_just_used(monkeypatch):
    monkeypatch.setattr(guard, "RAG_MPS_MEMORY_THRESHOLD_GB", 1.0)
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 2048.0)
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)

    evicted = []
    guard.register_model("bge_m3", lambda: evicted.append("bge_m3"))

    guard.guard_after_use("bge_m3")

    assert evicted == []


def test_guard_after_use_does_not_evict_warmed_but_never_used_model(monkeypatch):
    """Прогретая (register_model), но ещё ни разу не использованная модель
    (guard_after_use для неё не вызывался) не должна выгружаться экстренной
    проверкой — иначе прогрев теряет смысл."""
    monkeypatch.setattr(guard, "RAG_MPS_MEMORY_THRESHOLD_GB", 1.0)
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 2048.0)
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)

    evicted = []
    guard.register_model("cross_encoder", lambda: evicted.append("cross_encoder"))
    # ВАЖНО: никакого guard._last_use["cross_encoder"] — модель только прогрета.

    guard.guard_after_use("bge_m3")

    assert evicted == []


def test_guard_after_use_noop_when_under_threshold(monkeypatch):
    monkeypatch.setattr(guard, "RAG_MPS_MEMORY_THRESHOLD_GB", 10.0)
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 500.0)
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)

    evicted = []
    guard.register_model("cross_encoder", lambda: evicted.append("cross_encoder"))
    guard._last_use["cross_encoder"] = 0.0

    guard.guard_after_use("bge_m3")

    assert evicted == []


def test_guard_after_use_ignored_while_request_active(monkeypatch):
    """CRITICAL BUGFIX: пока активен хотя бы один RAG-запрос, emergency
    threshold override должен быть полным no-op — транзиентные пики RAM
    внутри запроса это норма, а не повод выгружать другую модель."""
    monkeypatch.setattr(guard, "RAG_MPS_MEMORY_THRESHOLD_GB", 1.0)
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 2048.0)
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)

    evicted = []
    guard.register_model("cross_encoder", lambda: evicted.append("cross_encoder"))
    guard._last_use["cross_encoder"] = 0.0

    guard.rag_request_started()
    guard.guard_after_use("bge_m3")

    assert evicted == []


def test_no_eviction_during_8gb_peak_inside_request_scope(monkeypatch):
    """CRITICAL BUGFIX: ABSOLUTE EVICTION LOCK — буквальный сценарий из
    отчёта: транзиентный пик 8 ГБ ВНУТРИ rag_request_scope(). Ни bge_m3, ни
    cross_encoder не должны выгружаться ни emergency threshold override'ом,
    ни (если бы вдруг сработал) cooldown'ом, пока scope не закрыт."""
    monkeypatch.setattr(guard, "RAG_MPS_MEMORY_THRESHOLD_GB", 6.0)
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 8192.0)
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)

    evicted: list[str] = []
    guard.register_model("bge_m3", lambda: evicted.append("bge_m3"))
    guard.register_model("cross_encoder", lambda: evicted.append("cross_encoder"))
    guard._last_use["bge_m3"] = 0.0
    guard._last_use["cross_encoder"] = 0.0

    async def _run():
        async with guard.rag_request_scope():
            guard.guard_after_use("bge_m3")
            guard.guard_after_use("cross_encoder")

    asyncio.run(_run())

    assert evicted == []


def test_guard_after_use_throttled_within_min_interval(monkeypatch):
    """Не чаще раза в _MIN_CHECK_INTERVAL_SEC — второй вызов сразу после
    первого не должен повторно запускать проверку footprint/выгрузки."""
    calls = []
    monkeypatch.setattr(
        guard, "_check_and_evict", lambda **kw: calls.append(kw)
    )

    guard.guard_after_use("bge_m3")
    guard.guard_after_use("bge_m3")

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Request-scoped 5-минутный cooldown (требование 2)
# ---------------------------------------------------------------------------


def test_cooldown_timer_not_scheduled_while_request_active():
    guard.rag_request_started()
    assert guard._active_requests == 1
    assert guard._cooldown_timer is None


def test_cooldown_scheduled_only_after_last_active_request_finishes(monkeypatch):
    scheduled: list[float] = []
    real_timer = threading.Timer

    def fake_timer(delay, fn):
        scheduled.append(delay)
        t = real_timer(delay, fn)
        t.daemon = True
        return t

    monkeypatch.setattr(guard.threading, "Timer", fake_timer)
    monkeypatch.setattr(guard, "RAG_MPS_REQUEST_COOLDOWN_SEC", 300.0)

    guard.rag_request_started()
    guard.rag_request_started()  # два конкурентных запроса
    guard.rag_request_finished()
    assert scheduled == []  # ещё один активный запрос — таймер не взводится

    guard.rag_request_finished()
    assert scheduled == [300.0]


def test_new_request_cancels_pending_cooldown_timer(monkeypatch):
    monkeypatch.setattr(guard, "RAG_MPS_REQUEST_COOLDOWN_SEC", 300.0)
    guard.rag_request_started()
    guard.rag_request_finished()
    timer_after_finish = guard._cooldown_timer
    assert timer_after_finish is not None
    assert timer_after_finish.is_alive()

    guard.rag_request_started()
    assert guard._cooldown_timer is None
    timer_after_finish.join(timeout=1.0)
    assert not timer_after_finish.is_alive()


def test_cooldown_fired_evicts_all_registered_models_unconditionally(monkeypatch):
    """Cooldown-выгрузка безусловна — не завязана на RAG_MPS_MEMORY_THRESHOLD_GB."""
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 50.0)
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)
    evicted = []
    guard.register_model("bge_m3", lambda: evicted.append("bge_m3"))
    guard.register_model("cross_encoder", lambda: evicted.append("cross_encoder"))

    guard._cooldown_fired()

    assert sorted(evicted) == ["bge_m3", "cross_encoder"]


def test_cooldown_fired_skips_eviction_if_new_request_started_meanwhile():
    evicted = []
    guard.register_model("bge_m3", lambda: evicted.append("bge_m3"))
    guard.rag_request_started()  # race: новый запрос стартовал между fire и cancel

    guard._cooldown_fired()

    assert evicted == []


def test_schedule_cooldown_uses_configured_delay_after_scope_exit(monkeypatch):
    """Выгрузка взводится СТРОГО через RAG_MPS_REQUEST_COOLDOWN_SEC после
    выхода из rag_request_scope() — не раньше, не с другой задержкой."""
    scheduled: list[float] = []
    real_timer = threading.Timer

    def fake_timer(delay, fn):
        scheduled.append(delay)
        t = real_timer(delay, fn)
        t.daemon = True
        return t

    monkeypatch.setattr(guard.threading, "Timer", fake_timer)
    monkeypatch.setattr(guard, "RAG_MPS_REQUEST_COOLDOWN_SEC", 300.0)

    async def _run():
        async with guard.rag_request_scope():
            assert scheduled == []  # пока scope открыт — таймер не взведён

    asyncio.run(_run())
    assert scheduled == [300.0]


def test_cooldown_delay_is_fixed_regardless_of_request_duration(monkeypatch):
    """CRITICAL BUGFIX: COOLDOWN TIMER MUST START AFTER REQUEST EXIT — симуляция
    "долгого" запроса (здесь — реальная короткая пауза вместо буквальных 10с,
    сам принцип от длительности не зависит): пока scope открыт, таймер не
    взведён НИ РАЗУ, ни на 0-й секунде, ни посередине; ровно на выходе он
    взводится с ПОЛНОЙ задержкой RAG_MPS_REQUEST_COOLDOWN_SEC, отсчитанной от
    момента выхода — не уменьшенной на время, проведённое внутри запроса."""
    scheduled: list[float] = []
    real_timer = threading.Timer

    def fake_timer(delay, fn):
        scheduled.append(delay)
        t = real_timer(delay, fn)
        t.daemon = True
        return t

    monkeypatch.setattr(guard.threading, "Timer", fake_timer)
    monkeypatch.setattr(guard, "RAG_MPS_REQUEST_COOLDOWN_SEC", 300.0)

    async def _run():
        async with guard.rag_request_scope():
            assert scheduled == []  # на входе (0-й секунде) — ничего не взведено
            await asyncio.sleep(0.05)  # симуляция длительной обработки запроса
            assert scheduled == []  # и посередине запроса — тоже ничего

    asyncio.run(_run())
    assert scheduled == [300.0]  # взведён СТРОГО на выходе, с полной задержкой


def test_cooldown_fired_and_new_request_are_mutually_exclusive(monkeypatch):
    """Regression на реальную гонку из отчёта: раньше проверка
    ``_active_requests == 0`` и сама выгрузка были ДВУМЯ отдельными
    секциями под локом — новый запрос мог стартовать в промежутке между
    ними и получить модели выгруженными прямо во время своего выполнения
    ("cooldown истёк ПРЯМО ВО ВРЕМЯ ВЫПОЛНЕНИЯ ЗАПРОСА"). Проверяем на
    настоящих потоках, что теперь это одна атомарная секция: пока
    ``_cooldown_fired`` физически внутри критической секции (выполняет
    unload), ``rag_request_started`` из другого потока обязан
    заблокироваться на том же локе, а не проскочить вперёд."""
    monkeypatch.setattr(guard, "release_mps_cache", lambda: None)
    monkeypatch.setattr(guard, "current_phys_footprint_mb", lambda: 0.0)

    started_unload = threading.Event()
    release_unload = threading.Event()

    def slow_unload():
        started_unload.set()
        assert release_unload.wait(timeout=2.0)

    guard.register_model("bge_m3", slow_unload)

    cooldown_thread = threading.Thread(target=guard._cooldown_fired)
    cooldown_thread.start()
    assert started_unload.wait(timeout=2.0)

    request_started = threading.Event()

    def start_request():
        guard.rag_request_started()
        request_started.set()

    request_thread = threading.Thread(target=start_request)
    request_thread.start()
    # unload ещё держит лок (ждёт release_unload) — новый запрос обязан висеть.
    assert not request_started.wait(timeout=0.2)

    release_unload.set()
    cooldown_thread.join(timeout=2.0)
    request_thread.join(timeout=2.0)

    assert request_started.is_set()
    assert guard._active_requests == 1


def test_rag_request_scope_is_paired_even_on_exception():
    async def _boom():
        async with guard.rag_request_scope():
            assert guard._active_requests == 1
            raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(_boom())
    assert guard._active_requests == 0


# ---------------------------------------------------------------------------
# OS-agnostic footprint fallback chain (требование 3)
# ---------------------------------------------------------------------------


def test_footprint_prefers_darwin_binary_when_available(monkeypatch):
    monkeypatch.setattr(guard.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(guard, "_footprint_darwin_mb", lambda: 4242.0)
    called_psutil = []
    monkeypatch.setattr(
        guard, "_psutil_mb", lambda: called_psutil.append(1) or 999.0
    )

    assert guard.current_phys_footprint_mb() == 4242.0
    assert called_psutil == []


def test_footprint_falls_back_to_psutil_when_darwin_binary_missing(monkeypatch):
    monkeypatch.setattr(guard.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(guard, "_footprint_darwin_mb", lambda: None)
    monkeypatch.setattr(guard, "_psutil_mb", lambda: 777.0)

    assert guard.current_phys_footprint_mb() == 777.0


def test_footprint_uses_psutil_directly_on_non_darwin(monkeypatch):
    monkeypatch.setattr(guard.platform, "system", lambda: "Linux")
    called_darwin = []
    monkeypatch.setattr(
        guard, "_footprint_darwin_mb", lambda: called_darwin.append(1) or 111.0
    )
    monkeypatch.setattr(guard, "_psutil_mb", lambda: 333.0)

    assert guard.current_phys_footprint_mb() == 333.0
    assert called_darwin == []


def test_psutil_mb_falls_back_to_rss_when_uss_unavailable(monkeypatch):
    class _FakeProc:
        def memory_full_info(self):
            raise Exception("uss unsupported on this platform")

        def memory_info(self):
            class _Mem:
                rss = 512 * 1024 * 1024

            return _Mem()

    class _FakePsutil:
        @staticmethod
        def Process(_pid):
            return _FakeProc()

    import sys

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil())
    assert guard._psutil_mb() == 512.0


def test_current_phys_footprint_mb_returns_zero_if_everything_fails(monkeypatch):
    monkeypatch.setattr(guard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(guard, "_psutil_mb", lambda: None)
    assert guard.current_phys_footprint_mb() == 0.0


# ---------------------------------------------------------------------------
# Async warmup: sequential order matching pipeline call order (требование 1)
# ---------------------------------------------------------------------------


def test_warmup_pipeline_loads_stages_sequentially_in_pipeline_order(monkeypatch):
    order: list[str] = []

    def _load_bge_m3():
        order.append("bge_m3")

    def _load_cross_encoder():
        order.append("cross_encoder")

    import knowledge_engine.services.search.bge_m3_embed as bge_mod
    import knowledge_engine.src.rag_gateway.cross_encoder as ce_mod

    monkeypatch.setattr(bge_mod, "ensure_bge_m3_loaded", _load_bge_m3)
    monkeypatch.setattr(ce_mod, "ensure_cross_encoder_loaded", _load_cross_encoder)

    asyncio.run(guard.warmup_pipeline_async())

    assert order == ["bge_m3", "cross_encoder"]


def test_warmup_pipeline_continues_after_stage_failure(monkeypatch):
    order: list[str] = []

    def _load_bge_m3():
        raise RuntimeError("no HF cache offline")

    def _load_cross_encoder():
        order.append("cross_encoder")

    import knowledge_engine.services.search.bge_m3_embed as bge_mod
    import knowledge_engine.src.rag_gateway.cross_encoder as ce_mod

    monkeypatch.setattr(bge_mod, "ensure_bge_m3_loaded", _load_bge_m3)
    monkeypatch.setattr(ce_mod, "ensure_cross_encoder_loaded", _load_cross_encoder)

    asyncio.run(guard.warmup_pipeline_async())  # не должно упасть целиком

    assert order == ["cross_encoder"]


def test_spawn_warmup_task_keeps_strong_reference_until_done():
    async def _run():
        task = guard.spawn_warmup_task(stages=())
        assert task in guard._background_warmup_tasks
        await task
        assert task not in guard._background_warmup_tasks

    asyncio.run(_run())
