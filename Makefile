# Reproduction pipeline. Every target is idempotent: re-running one that has
# already done its work costs nothing.
#
#   make setup          create the venv and install the package
#   make data           fetch the annotated CBERS-4A dataset
#   make weights        fetch the released fine-tuned checkpoint
#   make eval           score the released fine-tuned model  <- the short path
#   make sweep          re-run the full learning-rate sweep (GPU, hours)
#
# The short path -- setup, data, weights, eval -- reproduces the headline
# numbers without training anything.

PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup data weights eval sweep sweep-half deepglobe deepglobe-tiles \
        base results clean-runs

help:
	@sed -n 's/^#   //p' Makefile

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

setup: $(BIN)/python
	$(BIN)/pip install -e .

data: setup
	$(BIN)/python scripts/fetch/pampa_dataset.py

weights: setup
	$(BIN)/python scripts/fetch/checkpoints.py

eval: data weights
	$(BIN)/python -m pitcic.eval.cli

# The full sweep. One process per run; never run two at once on one card.
sweep: data weights
	$(BIN)/python scripts/run_sweep.py full_budget

sweep-half: data weights
	$(BIN)/python scripts/run_sweep.py half_budget

results:
	$(BIN)/python scripts/collect_results.py

# Stage 1. Needed to reproduce the fine-tuning arms: the base checkpoint is not
# redistributable, so it is built here rather than downloaded.
deepglobe: setup
	$(BIN)/pip install -e '.[kaggle]'
	$(BIN)/python scripts/fetch/deepglobe.py

deepglobe-tiles: deepglobe
	$(BIN)/python -m pitcic.data.deepglobe

base: deepglobe-tiles
	$(BIN)/python -m pitcic.train.deepglobe

clean-runs:
	rm -rf runs
