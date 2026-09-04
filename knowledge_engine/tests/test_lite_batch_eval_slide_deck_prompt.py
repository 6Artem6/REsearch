"""Flash Lite Content Quality Gate (_BATCH_SYSTEM, lite_search_pipeline.py):
промпт должен явно браковать слайд-деки/фрагментированный текст ПО
СОДЕРЖАНИЮ, а не по формату/расширению (PDF — легитимный формат, наравне с
text/md/html/github) — реальный кейс: PDF с университетских слайдов
(wisc.edu) прошёл Flash Lite как валидный источник для лекции, хотя это
отрывочные буллеты без связного текста. Промпт — на английском (правило
проекта), эта проверка — на английские ключевые слова."""

from __future__ import annotations

from knowledge_engine.src.curriculum.lite_search_pipeline import _BATCH_SYSTEM


def test_batch_system_rejects_slide_decks_by_content_not_format() -> None:
    text = _BATCH_SYSTEM.lower()
    assert "slide deck" in text
    assert "bullet" in text
    assert "domain authority" in text  # авторитетность домена не оправдывает формат
    # PDF/ppt/расширения не должны фигурировать как критерий — только текст.
    assert "pdf" not in text
    assert ".ppt" not in text
