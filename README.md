# REsearch — Knowledge Engine

Монорепозиторий локального **Knowledge Engine** (v0.8): FastAPI, Consensus/Playwright, Light RAG, web UI `/app`.

## Быстрый старт

```bash
cp .env.example .env          # ключи только локально, в git не коммитятся
make setup                    # SearXNG + Ollama + Python venv
make dev                      # API на http://127.0.0.1:8765
```

Подробности: [knowledge_engine/README.md](knowledge_engine/README.md) и [knowledge_engine/docs/DEV_RUNBOOK.md](knowledge_engine/docs/DEV_RUNBOOK.md).

## Качество кода

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
pip install -r knowledge_engine/requirements-dev.txt
make format    # black + isort
make lint      # flake8
make check     # format --check + lint
```

## GitHub (приватный репозиторий)

```bash
gh auth login
gh repo create REsearch --private --source=. --remote=origin --push
```

Секреты: только `.env.example` в репозитории. Браузерные профили, LanceDB, `.runs/` — в `.gitignore`.
