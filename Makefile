# Knowledge Engine — быстрые команды (из корня REsearch)

.PHONY: infra ollama python dev api sync-venv setup smoke-v07 v08 format lint check

PYTHON ?= ./.venv/bin/python
BLACK ?= ./.venv/bin/black
ISORT ?= ./.venv/bin/isort
FLAKE8 ?= ./.venv/bin/flake8
KE_PY_DIRS = knowledge_engine
export PYTHONPATH := $(CURDIR)

infra:
	docker compose up -d searxng

ollama:
	./knowledge_engine/scripts/setup-host-ollama.sh

python:
	./knowledge_engine/scripts/setup-host-python.sh

dev:
	./knowledge_engine/scripts/dev-native.sh

api:
	docker compose --profile api up -d knowledge-api

smoke-v07:
	SKIP_V07_FETCH=1 ./knowledge_engine/scripts/smoke_v07.sh "GraphRAG LanceDB cache invalidation"

v07:
	./knowledge_engine/scripts/run-v07-analysis.sh "GraphRAG LanceDB cache invalidation on Mac M4"

# v0.8 web smoke (API должен быть запущен, GRAPH_VERSION=0.8)
v08:
	@echo "Open http://127.0.0.1:$${KE_API_PORT:-8765}/app"
	@curl -sf "http://127.0.0.1:$${KE_API_PORT:-8765}/api/v1/health" | jq .

sync-venv:
	./knowledge_engine/scripts/sync-venv.sh

setup:
	./knowledge_engine/scripts/setup.sh

format:
	$(BLACK) $(KE_PY_DIRS)
	$(ISORT) $(KE_PY_DIRS)

lint:
	$(FLAKE8) $(KE_PY_DIRS)

check:
	$(BLACK) --check $(KE_PY_DIRS)
	$(ISORT) --check-only $(KE_PY_DIRS)
	$(FLAKE8) $(KE_PY_DIRS)

dev-deps:
	./.venv/bin/pip install -q -r knowledge_engine/requirements-dev.txt
