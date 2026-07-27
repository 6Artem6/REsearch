FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY knowledge_engine/docker/knowledge-engine/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Код и pip-пакеты — с хоста: volume ./ → /app, общий .venv
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "knowledge_engine.main"]
