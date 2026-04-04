# ============================================================
# CnakeCharmer Makefile
# ============================================================

# --- Configuration ---
PROFILE ?= gpt_oss_120b
TRAINING_CONFIG ?= sft_base
UV_RUN := uv run --no-sync
# Helper: read a dotted key from a model profile
PROFILE_GET = $(UV_RUN) python -c "from cnake_charmer.config import load_model_profile; c=load_model_profile('$(PROFILE)'); print(c.$(1))"

# --- Data Collection ---
.PHONY: traces traces-best consolidate build-sft

traces:  ## Collect traces using a model profile
	$(UV_RUN) python scripts/collect_traces.py \
		--model $$($(call PROFILE_GET,model.id)) \
		-o $$($(call PROFILE_GET,collection.output)) \
		--all --shuffle

traces-best:  ## Collect best-of-N traces (5 attempts, keep best)
	$(UV_RUN) python scripts/collect_traces.py \
		--model $$($(call PROFILE_GET,model.id)) \
		-o $$($(call PROFILE_GET,collection.output)) \
		--all --shuffle --attempts 5

consolidate:  ## Consolidate trace files into master JSONL
	$(UV_RUN) python scripts/consolidate_traces.py

build-sft:  ## Build SFT dataset from consolidated traces
	$(UV_RUN) python scripts/build_sft.py --min-score 0.8 --top-k 1 --require-finish

# --- Training ---
.PHONY: train-sft train-grpo

train-sft:  ## Run SFT training
	$(UV_RUN) python scripts/train_sft.py

train-grpo:  ## Run GRPO training
	$(UV_RUN) python scripts/train_grpo.py

# --- Evaluation ---
.PHONY: test-model benchmark sample

test-model:  ## Test trained model on unseen problems
	$(UV_RUN) python scripts/test_sft_model.py

benchmark:  ## Run benchmarks (hash-cached, only changed files)
	$(UV_RUN) python run_benchmarks.py

sample:  ## Sample 3 random problems with a model
	$(UV_RUN) python scripts/collect_traces.py \
		--model $$($(call PROFILE_GET,model.id)) \
		--n-random 3 --attempts 1

# --- Development ---
.PHONY: test test-data test-tooling compile lint

test:  ## Run all tests
	$(UV_RUN) pytest tests/ -x -q

test-data:  ## Run dataset equivalence tests only
	$(UV_RUN) pytest tests/data/ -x -q

test-tooling:  ## Run tooling tests only
	$(UV_RUN) pytest tests/tooling/ -x -q

compile:  ## Compile Cython extensions
	uv run python setup.py build_ext --inplace

lint:  ## Lint Python and Cython code
	$(UV_RUN) ruff check cnake_charmer/ cnake_data/ scripts/
	$(UV_RUN) cython-lint --no-pycodestyle cnake_data/cy/

# --- Prompt Optimization ---
.PHONY: optimize-prompt

optimize-prompt:  ## Optimize prompt for a model profile
	$(UV_RUN) python scripts/optimize_prompt.py \
		--model $$($(call PROFILE_GET,model.id))

# --- Utilities ---
.PHONY: list-problems

list-problems:  ## List all problem pairs in the dataset
	$(UV_RUN) python -c "from cnake_data.loader import discover_pairs; pairs=discover_pairs(); print(f'{len(pairs)} problems'); [print(f'  {p.problem_id} ({p.category})') for p in sorted(pairs, key=lambda x: x.problem_id)[:20]]; print('  ...' if len(pairs)>20 else '')"

# --- Help ---
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
