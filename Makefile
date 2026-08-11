# Knowledge Engine — быстрые команды (из корня REsearch)

.PHONY: infra ollama python dev api worker sync-venv setup smoke-v07 v08 format lint check pre-commit-install pre-commit-run skill-tree-ui ship check-gemini-grounding

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

worker:
	$(PYTHON) -m knowledge_engine.worker

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

PRE_COMMIT ?= ./.venv/bin/pre-commit

pre-commit-install:
	$(PRE_COMMIT) install --install-hooks

pre-commit-run:
	$(PRE_COMMIT) run --all-files

dev-deps:
	./.venv/bin/pip install -q -r knowledge_engine/requirements-dev.txt

skill-tree-ui:
	./knowledge_engine/scripts/build-skill-tree-ui.sh

check-gemini-grounding:
	$(PYTHON) -m knowledge_engine.scripts.check_gemini_grounding --save

check-gemini-grounding-all:
	$(PYTHON) -m knowledge_engine.scripts.check_gemini_grounding --all-candidates --compare-plain --metadata --save

# git add . + commit + push origin main (на commit — pre-commit: autoflake/isort/black/flake8)
# Пример: make ship MSG="fix: redis worker pubsub"
ship:
	@test -n "$(MSG)" || (echo 'Usage: make ship MSG="your commit message"'; exit 1)
	git add .
	git commit -m "$(MSG)"
	git push origin main
