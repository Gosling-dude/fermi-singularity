# Fermi Podcast Companion
.DEFAULT_GOAL := help
SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export PYTHONPATH := src:.

# Prefer a modern interpreter; the project needs >= 3.11.
PYTHON_BIN ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3)

.PHONY: help setup check-ffmpeg ingest reingest chat web search episodes doctor \
        test eval eval-retrieval eval-baseline eval-improved eval-compare \
        fixtures demo clean clean-data

help: ## Show available commands
	@echo "Fermi Podcast Companion"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:  make setup && make ingest && make chat"

$(VENV)/bin/python:
	@echo "→ creating virtualenv with $(PYTHON_BIN)"
	@$(PYTHON_BIN) -m venv $(VENV)
	@$(PIP) install -q --upgrade pip

check-ffmpeg:
	@command -v ffmpeg >/dev/null 2>&1 || { \
	  echo "ERROR: ffmpeg is not installed."; \
	  echo "  → macOS:  brew install ffmpeg"; \
	  echo "  → Ubuntu: sudo apt install ffmpeg"; exit 1; }

setup: $(VENV)/bin/python check-ffmpeg ## Install dependencies (one time)
	@echo "→ installing dependencies"
	@$(PIP) install -q -e ".[dev]"
	@test -f .env || { cp .env.example .env; echo "→ created .env from .env.example"; }
	@echo ""
	@echo "Setup complete."
	@echo "  1. Put the podcast .mp3 files in audio/   (or: make fixtures)"
	@echo "  2. Add your API key to .env               (chat + eval judge)"
	@echo "  3. make ingest && make chat"

fixtures: check-ffmpeg ## Generate synthetic dev audio (macOS only, for testing)
	@$(PY) scripts/make_fixture_audio.py

ingest: check-ffmpeg ## Transcribe audio and build the search indexes
	@$(PY) -m companion.interface.cli ingest

reingest: check-ffmpeg ## Re-transcribe everything from scratch
	@$(PY) -m companion.interface.cli ingest --force

chat: ## Start the conversation (main interface)
	@$(PY) -m companion.interface.cli chat

web: ## Start the web UI at http://127.0.0.1:8000
	@$(PY) -m companion.interface.web

episodes: ## List the ingested episodes
	@$(PY) -m companion.interface.cli episodes

doctor: ## Check the environment and report what is missing
	@$(PY) -m companion.interface.cli doctor

search: ## Retrieval only, no API key needed.  make search Q="your question"
	@$(PY) -m companion.interface.cli search $(Q)

test: ## Run the test suite
	@$(PY) -m pytest -q

eval: ## Full evaluation: pipeline + LLM judge  (needs an API key)
	@$(PY) -m eval.runner

eval-retrieval: ## Offline evaluation of retrieval + refusal (no API key, free)
	@$(PY) -m eval.runner --retrieval-only

eval-baseline: ## Reproduce the baseline run (dense-only, no reranking)
	@RETRIEVAL_MODE=dense ENABLE_RERANK=false \
	  $(PY) -m eval.runner --retrieval-only --label run_baseline

eval-improved: ## Reproduce the improved run (hybrid + calibrated relevance gate)
	@$(PY) -m eval.runner --retrieval-only --label run_improved

eval-compare: ## Compare baseline against improved
	@$(PY) -m eval.report run_baseline run_improved

demo: ## Print the demo script
	@cat docs/DEMO.md

clean: ## Remove caches and build artefacts
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
	@echo "cleaned"

clean-data: ## Delete all generated data (forces a full re-ingest)
	@rm -rf data/transcripts data/chunks data/index data/normalized data/manifest.json
	@echo "data/ cleared — run `make ingest` to rebuild"
