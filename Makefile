.PHONY: help venv install up down logs ingest ingest-local eval eval-retrieval eval-llm api ui lint fmt test clean

SHELL := /bin/bash
COMPOSE := docker compose

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create a local venv with pinned deps (uv)
	uv venv --python 3.11 .venv
	uv pip install --python .venv/bin/python -r requirements.txt
	uv pip install --python .venv/bin/python -r requirements-dev.txt

up: ## Start the whole stack (postgres, qdrant, api, ui, grafana)
	test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build postgres qdrant api ui grafana

down: ## Stop the stack
	$(COMPOSE) down

nuke: ## Stop the stack and delete all volumes
	$(COMPOSE) down -v

logs: ## Tail logs
	$(COMPOSE) logs -f --tail=100

ingest: ## Run the full ingestion pipeline inside docker
	$(COMPOSE) run --rm ingest

ingest-local: ## Run the full ingestion pipeline with the local venv
	.venv/bin/python -m ingestion.pipeline all

eval-retrieval: ## Evaluate all retrieval approaches
	.venv/bin/python -m eval_suite.eval_retrieval

eval-llm: ## Evaluate all prompt variants
	.venv/bin/python -m eval_suite.eval_llm

ground-truth: ## Generate the ground-truth question set
	.venv/bin/python -m eval_suite.generate_ground_truth

eval: ground-truth eval-retrieval eval-llm ## Full evaluation suite

api: ## Run the API locally
	.venv/bin/uvicorn app.api:app --reload --port 8000

ui: ## Run the Streamlit UI locally
	API_URL=http://localhost:8000 .venv/bin/streamlit run ui/streamlit_app.py

lint: ## Ruff check
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt: ## Ruff format
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

test: ## Run tests
	.venv/bin/pytest -q
