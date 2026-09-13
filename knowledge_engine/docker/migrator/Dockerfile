# Лёгкий образ ТОЛЬКО для alembic-миграций (Phase 3, см. prompt.txt) —
# явный список COPY, без PyTorch/CUDA/sentence-transformers/exa-py и прочих
# ML-зависимостей основного Dockerfile. Деплой baked-at-build-time (не
# runtime volume + entrypoint.sh, как knowledge-api) — иммутабельный
# init-контейнер, поднимается за секунды.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY knowledge_engine/requirements-migrator.txt ./knowledge_engine/requirements-migrator.txt
RUN pip install --no-cache-dir -r knowledge_engine/requirements-migrator.txt

# Явный список — только то, что реально нужно alembic upgrade head +
# integrity-check (scripts/db/run_migrations_and_verify.py). НЕ COPY всего
# knowledge_engine/ целиком, чтобы случайно не потянуть что-то с тяжёлым
# импортом из services/src.
COPY alembic.ini ./alembic.ini
COPY knowledge_engine/__init__.py ./knowledge_engine/__init__.py
COPY knowledge_engine/src/__init__.py ./knowledge_engine/src/__init__.py
COPY knowledge_engine/src/config/__init__.py ./knowledge_engine/src/config/__init__.py
COPY knowledge_engine/src/config/dotenv_loader.py ./knowledge_engine/src/config/dotenv_loader.py
COPY knowledge_engine/src/config/settings.py ./knowledge_engine/src/config/settings.py
COPY knowledge_engine/src/config/postgres_settings.py ./knowledge_engine/src/config/postgres_settings.py
COPY knowledge_engine/alembic ./knowledge_engine/alembic
COPY knowledge_engine/scripts/db/run_migrations_and_verify.py ./knowledge_engine/scripts/db/run_migrations_and_verify.py

CMD ["python", "-m", "knowledge_engine.scripts.db.run_migrations_and_verify"]
