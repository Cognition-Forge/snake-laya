# Targets mirror README Setup/Run/Tests. Extra CLI flags: make run ARGS="--mode max --seed 1"
# Requires uv; ruff runs via uvx (not a project dependency). Compatible with GNU Make 3.81 (macOS).

UV    ?= uv
RUFF  ?= uvx ruff
SRC   := src tests
ARGS  ?=
N     ?= 400

.DEFAULT_GOAL := help
.PHONY: help sync sync-mlx run run-mlx run-heuristic bench test test-slow lint fmt check clean

help: ## list targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo "  vars: ARGS=\"<snake-laya/pytest flags>\"  N=<bench decisions, default $(N)>"

sync: ## install laya (PyTorch) + dev deps; drops the mlx extra if present
	$(UV) sync

sync-mlx: ## also install laya-mlx (Apple silicon only)
	$(UV) sync --extra mlx

run: ## play: human vs Laya (PyTorch)
	$(UV) run snake-laya $(ARGS)

run-mlx: ## play with the MLX runtime (needs sync-mlx)
	$(UV) run snake-laya --brain laya-mlx $(ARGS)

run-heuristic: ## play against the heuristic baseline, no model
	$(UV) run snake-laya --brain heuristic $(ARGS)

bench: ## headless: N decisions, print stats
	$(UV) run snake-laya --bench $(N) $(ARGS)

test: ## fast suite; real-model tests skipped
	$(UV) run pytest $(ARGS)

test-slow: ## real-model smoke tests on every installed backend (downloads checkpoints once)
	$(UV) run pytest --run-slow tests/test_laya_smoke.py $(ARGS)

lint: ## ruff lint + format check (no writes)
	$(RUFF) check $(SRC)
	$(RUFF) format --check $(SRC)

fmt: ## apply ruff fixes and formatting
	$(RUFF) check --fix $(SRC)
	$(RUFF) format $(SRC)

check: lint test ## lint then fast tests (pre-commit gate)

clean: ## remove caches and build output (keeps .venv and HF cache)
	rm -rf .pytest_cache .ruff_cache build dist
	find $(SRC) -type d -name __pycache__ -prune -exec rm -rf {} +
