# CnakeCharmer Refactor: North Star Architecture

## Context

CnakeCharmer has grown organically through multiple pivots — from a pure Cython dataset, to an MCP-assisted coding tool, to a full training pipeline (SFT + GRPO). The result is a repo with duplicated evaluation logic across 3 callers, ~30 ad-hoc shell scripts, dead code from abandoned experiments, no configuration management, and no clear boundary between the "living dataset" and the "training tooling." This refactor establishes clean architecture, eliminates redundancy, and makes the codebase sustainable for continued development.

**Goals:**
- Two-package separation: dataset vs tooling
- Single DRY evaluation submodule (the backbone shared by MCP, training, trace collection)
- YAML configuration with OmegaConf (model profiles + training configs)
- Makefile-driven workflows replacing shell scripts
- Archive dead code, delete superseded artifacts
- Pydantic-modeled trace format with migration path

---

## Phase 0: Pre-Refactor Safety

Before any code changes:

1. **Tag current state**: `git tag pre-refactor-v1`
2. **Back up critical data**: Copy `data/traces/master_thinking.jsonl` and `master_nothink.jsonl` to external storage
3. **Compress old traces**: `zip -r data/traces_old.zip data/traces_old/ && rm -rf data/traces_old/`
4. **Verify tests pass**: `uv run --no-sync pytest tests/ -x -q`

---

## Phase 1: Archive & Delete

### Archive to `archive/` branch (create branch, move files, then remove from main)

| Item | Current Location | Reason |
|------|-----------------|--------|
| SIMD kernels | `cnake_charmer/cy_simd/` | Moving to separate repo eventually |
| Engine kernels | `cnake_charmer/engine/` | Future inference engine, not in scope |
| Pure-python Cython | `cnake_charmer/pp/` | Third tier superseded by py→cy focus |
| Source loaders | `cnake_charmer/sources/` | One-time dataset expansion, not active |
| Code generation | `cnake_charmer/generate/` | Not actively used |
| Credit assignment | `cnake_charmer/training/credit.py` | MURPHY tree rollouts not in use |
| Web dashboard | `builder/` | Separate service, not in pipeline |
| Prompt reference | `prompt/` | Manual reference only, not loaded by code |
| Test output | `output/` | Stale test checkpoint |
| XNNPACK comparison | `scripts/compare_xnnpack.py` | Research script, results in benchmarks.md |
| Old trace converter | `scripts/reconvert_old_traces.py` | Legacy migration, purpose fulfilled |
| Marimo notebook | `utils/train.py` | Demo/reference, not core pipeline |

### Delete (superseded, no archive needed)

| Item | Reason |
|------|--------|
| `data/tool_schemas.json` | Superseded by `tools.json` (3-param version) |
| `data/sft_dataset.jsonl` (v1) | Superseded by `sft_dataset_v2.jsonl` |
| `scripts/test_training.py` | Redundant with `test_grpo_memory.py` |
| All 29 `start_*.sh` scripts | Replaced by YAML profiles + Makefile |

---

## Phase 2: Two-Package Split

### Target structure

```
CnakeCharmer/
├── cnake_data/                    # PACKAGE 1: Living dataset
│   ├── __init__.py
│   ├── py/                        # 685 Python implementations
│   │   ├── algorithms/
│   │   ├── compression/
│   │   └── ... (19 categories)
│   ├── cy/                        # 665 Cython implementations
│   │   ├── algorithms/
│   │   └── ... (19 categories)
│   ├── benchmarks/                # Decorator registry (tags data)
│   │   ├── __init__.py
│   │   └── registry.py
│   └── loader.py                  # Problem pair discovery
│
├── cnake_charmer/                 # PACKAGE 2: Tooling
│   ├── __init__.py
│   ├── eval/                      # Unified evaluation (THE core submodule)
│   │   ├── __init__.py
│   │   ├── compiler.py            # Cython compilation
│   │   ├── annotations.py         # HTML annotation parsing
│   │   ├── correctness.py         # Test execution
│   │   ├── benchmark.py           # Performance timing
│   │   ├── memory_safety.py       # ASan integration
│   │   ├── lint.py                # cython-lint
│   │   ├── pipeline.py            # Orchestrate all steps, composite scoring
│   │   ├── sandbox.py             # Subprocess isolation + safety checks
│   │   └── models.py              # Pydantic result types
│   ├── training/                  # SFT + GRPO
│   │   ├── __init__.py
│   │   ├── environment.py         # CythonToolEnvironment (uses eval/)
│   │   ├── grpo.py                # GRPO reward, dataset, curriculum
│   │   ├── sft.py                 # SFT scoring + validation (merged)
│   │   ├── dspy_agent.py          # DSPy ReAct agent (uses eval/)
│   │   └── prompts.py             # Prompt loading + formatting
│   ├── traces/                    # Trace collection + dataset building
│   │   ├── __init__.py
│   │   ├── collector.py           # Merged collect + generate traces
│   │   ├── builder.py             # SFT dataset builder (from build_sft.py)
│   │   ├── consolidate.py         # Trace consolidation
│   │   ├── optimize.py            # GEPA prompt optimization
│   │   ├── models.py              # Pydantic trace/trajectory models
│   │   └── lm.py                  # LM configuration utilities (DRY)
│   ├── config.py                  # OmegaConf config loading
│   └── mcp_server.py              # MCP tools (uses eval/)
│
├── tests/
│   ├── data/                      # Dataset equivalence tests
│   │   ├── algorithms/
│   │   │   ├── test_binary_search.py
│   │   │   └── ...
│   │   └── ... (19 categories)
│   └── tooling/                   # Tooling unit tests
│       ├── test_eval_pipeline.py
│       ├── test_sft_builder.py
│       ├── test_grpo_training.py
│       └── ...
│
├── configs/
│   ├── models/                    # OmegaConf model profiles
│   │   ├── _base.yaml             # Defaults
│   │   ├── gpt_oss_120b.yaml
│   │   ├── deepseek_v3.yaml
│   │   ├── qwen_3_5.yaml
│   │   ├── glm5.yaml
│   │   └── ...
│   └── training/                  # Training configs
│       ├── sft_base.yaml
│       ├── grpo_base.yaml
│       └── ...
│
├── scripts/                       # Thin CLI entry points
│   ├── collect_traces.py          # Merged collect + generate
│   ├── train_sft.py
│   ├── train_grpo.py
│   ├── build_sft.py
│   ├── optimize_prompt.py
│   ├── test_model.py              # Merged test_sft_model
│   ├── test_memory.py             # GPU memory smoke test
│   └── sample.py                  # Manual sampling / inspection
│
├── data/
│   ├── system_prompt.txt
│   ├── tools.json                 # Single canonical tool schema
│   ├── sft_dataset_v2.jsonl       # Current SFT dataset
│   ├── grpo_problems/             # 59 standalone GRPO problems
│   ├── traces/
│   │   ├── master_thinking.jsonl
│   │   └── master_nothink.jsonl
│   ├── traces_old.zip             # Compressed archive
│   └── optimized_prompts/
│       ├── seed_prompt.txt
│       └── {model_slug}/program.json
│
├── Makefile                       # Primary workflow interface
├── pyproject.toml                 # Dual-package config
├── setup.py                       # Cython build (cnake_data/cy/)
├── run_benchmarks.py
├── benchmarks.md
└── .benchmark_cache.json
```

### pyproject.toml changes

```toml
[project]
name = "cnake-charmer"
# ...existing deps...
dependencies = [
    # Add omegaconf
    "omegaconf>=2.3",
    "pydantic>=2.0",
    # ...existing...
]

[tool.setuptools.packages.find]
include = ["cnake_charmer*", "cnake_data*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "data: dataset equivalence tests",
    "tooling: tooling unit tests",
]
```

### Import migration (1,350+ files)

All `cnake_data/py/*.py` files:
```python
# OLD: from cnake_charmer.benchmarks import python_benchmark
# NEW: from cnake_data.benchmarks import python_benchmark
```

All `cnake_data/cy/*.pyx` files:
```python
# OLD: from cnake_charmer.benchmarks import cython_benchmark
# NEW: from cnake_data.benchmarks import cython_benchmark
```

All `tests/data/**/*.py` files:
```python
# OLD: from cnake_charmer.py.algorithms.binary_search import ...
#      from cnake_charmer.cy.algorithms.binary_search import ...
# NEW: from cnake_data.py.algorithms.binary_search import ...
#      from cnake_data.cy.algorithms.binary_search import ...
```

**Strategy:** Scripted sed/replace. Verify with `uv run --no-sync pytest tests/data/ -x` after.

### setup.py changes

Update glob paths from `cnake_charmer/cy` to `cnake_data/cy`. Remove `pp`, `cy_simd`, `engine` paths.

### loader.py migration

Move `cnake_charmer/dataset/loader.py` → `cnake_data/loader.py`. Update path constants:
```python
PACKAGE_ROOT = Path(__file__).parent      # → cnake_data/
PY_DIR = PACKAGE_ROOT / "py"
CY_DIR = PACKAGE_ROOT / "cy"
TESTS_DIR = PACKAGE_ROOT.parent / "tests" / "data"
```

Move `difficulty.py` into `cnake_data/` as well — it classifies dataset problems.

### Benchmark registry changes

Move `cnake_charmer/benchmarks/` → `cnake_data/benchmarks/`. Update `registry.py` module path extraction:
```python
# OLD: extracts category from cnake_charmer.{py|cy}.{category}.{name}
# NEW: extracts category from cnake_data.{py|cy}.{category}.{name}
```

Update `run_benchmarks.py`:
```python
# OLD: import_all_submodules("cnake_charmer")
# NEW: import_all_submodules("cnake_data")
```

---

## Phase 3: Unified Evaluation Submodule (`cnake_charmer/eval/`)

This is the most critical refactor. Three separate implementations become one.

### Current state (3 implementations)

| Caller | Location | Features |
|--------|----------|----------|
| DSPy agent | `training/dspy_agent.py` | Subprocess isolation, safety checks, combined output |
| Training env | `training/environment.py` | Inline execution, step scores, DEFAULT_WEIGHTS |
| MCP server | `mcp_server.py` + `validate/*` | Direct function calls, granular tools |

### Target state (1 implementation)

```python
# cnake_charmer/eval/__init__.py
from .pipeline import evaluate_cython, compile_file, annotate_file, check_memory
from .pipeline import score_problem, composite_score
from .models import EvaluationResult, CompilationResult, AnnotationResult
from .sandbox import safe_evaluate
```

### Key design: `cnake_charmer/eval/sandbox.py`

```python
"""Subprocess-isolated evaluation with safety checks."""
import multiprocessing as mp
from .safety import check_code_safety
from .pipeline import _evaluate_impl

def safe_evaluate(
    code: str,
    python_code: str,
    test_code: str,
    *,
    timeout: int = 120,
) -> EvaluationResult:
    """Run full evaluation in subprocess. All callers use this."""
    check_code_safety(code)
    ctx = mp.get_context("spawn")
    # ... run _evaluate_impl in subprocess with timeout ...
    return result
```

### Key design: `cnake_charmer/eval/pipeline.py`

```python
"""Core evaluation orchestration. Merges validate/ + rewards/ logic."""

def evaluate_cython(code, python_code, test_code, *, 
                     timeout=120) -> EvaluationResult:
    """Full evaluation: compile → lint → annotate → test → benchmark → ASan."""
    # This is the canonical implementation.
    # Called by sandbox.safe_evaluate() in subprocess.

def composite_score(result: EvaluationResult, 
                    weights: dict | None = None) -> float:
    """Weighted composite reward. Used by GRPO, SFT scoring, MCP."""
    # Merges rewards/composite.py logic here.
    # DEFAULT_WEIGHTS defined here.

def compile_file(pyx_path: str, **kwargs) -> CompilationResult:
    """Single-file compilation (for MCP tool)."""

def annotate_file(pyx_path: str) -> AnnotationResult:
    """Compile + parse annotations (for MCP tool)."""

def check_memory(pyx_path: str, func_name: str, test_args: str) -> MemoryResult:
    """ASan memory safety check (for MCP tool)."""
```

### Key design: `cnake_charmer/eval/models.py`

```python
"""Pydantic models for all evaluation results."""
from pydantic import BaseModel

class CompilationResult(BaseModel):
    success: bool
    errors: list[str]
    warnings: list[str]
    so_path: str | None = None
    html_path: str | None = None

class AnnotationResult(BaseModel):
    score: float          # 0.0 (all Python) → 1.0 (all C)
    total_lines: int
    typed_lines: int
    hints: list[str]

class CorrectnessResult(BaseModel):
    score: float          # tests_passed / tests_total
    passed: int
    total: int
    failures: list[str]

class BenchmarkResult(BaseModel):
    speedup: float
    py_mean: float
    cy_mean: float
    py_std: float
    cy_std: float

class LintResult(BaseModel):
    score: float
    violations: int
    details: list[str]

class MemoryResult(BaseModel):
    score: float
    error_count: int
    leak_bytes: int
    error_types: list[str]

class EvaluationResult(BaseModel):
    compilation: CompilationResult
    annotations: AnnotationResult | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkResult | None = None
    lint: LintResult | None = None
    memory_safety: MemoryResult | None = None
    composite_score: float = 0.0
```

### Migration: Who calls what after refactor

| Caller | Before | After |
|--------|--------|-------|
| MCP `score_problem` | `rewards.composite_reward()` | `eval.safe_evaluate()` + `eval.composite_score()` |
| MCP `compile_file` | `validate.compiler.compile_cython()` | `eval.compile_file()` |
| MCP `annotate_file` | `validate.compiler` + `validate.annotations` | `eval.annotate_file()` |
| MCP `check_memory` | `validate.memory_safety` | `eval.check_memory()` |
| MCP `evaluate_cython` | `validate.pipeline.validate()` | `eval.safe_evaluate()` |
| Training environment | `environment.evaluate_cython()` (inline) | `eval.safe_evaluate()` |
| DSPy agent | `dspy_agent.evaluate_cython()` (subprocess) | `eval.safe_evaluate()` |
| SFT scoring | `sft_scoring.score_trace()` | `eval.composite_score()` |
| GRPO reward | `grpo.cython_reward()` | `eval.composite_score()` with curriculum weights |

### Directories to delete after migration

- `cnake_charmer/validate/` → logic absorbed into `eval/`
- `cnake_charmer/rewards/` → logic absorbed into `eval/pipeline.py`

---

## Phase 4: Configuration Management (OmegaConf)

### Model profiles: `configs/models/`

```yaml
# configs/models/_base.yaml
model:
  temperature: 1.0
  max_tokens: 8192
  cache: false
collection:
  attempts: 2
  max_iters: 5
  shuffle: true
  output: data/traces/master_thinking.jsonl
```

```yaml
# configs/models/deepseek_v3.yaml
defaults:
  - _base

model:
  id: openrouter/deepseek/deepseek-v3.2
  api_key_env: OPENROUTER_API_KEY
prompt:
  program: data/optimized_prompts/openrouter_deepseek_deepseek-v3.2/program.json
  id: gepa_deepseek
```

```yaml
# configs/models/gpt_oss_120b.yaml
defaults:
  - _base

model:
  id: openai/gpt-oss-120b
  base_url: http://localhost:8000/v1
  api_key_env: APIKEY
prompt:
  program: data/optimized_prompts/openai_gpt-oss-120b/program.json
  id: gepa_gptoss
```

### Training configs: `configs/training/`

```yaml
# configs/training/sft_base.yaml
model:
  base: openai/gpt-oss-20b
  output: models/gpt-oss-20b-cython-sft
dataset:
  path: data/sft_dataset_v2.jsonl
  max_length: 16384
lora:
  rank: 64
  alpha: 128
  dropout: 0.05
training:
  epochs: 1
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  warmup_ratio: 0.05
  logging_steps: 1
  max_steps: -1
merge:
  enabled: true
  mxfp4: true
```

```yaml
# configs/training/grpo_base.yaml
model:
  base: models/gpt-oss-20b-cython-sft-merged
  output: models/gpt-oss-20b-cython-grpo
data:
  n_problems: -1  # all
  eval_holdout: 50
  seed: 42
grpo:
  num_generations: 8
  max_completion_length: 8192
  max_tool_iters: 5
  loss_type: cispo
  beta: 0.001
lora:
  rank: 32
  alpha: 64
  dropout: 0.05
training:
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 5e-7
  warmup_ratio: 0.05
  logging_steps: 1
  max_steps: -1
```

### Config loading: `cnake_charmer/config.py`

```python
"""OmegaConf-based configuration management."""
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

def load_model_profile(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a model profile by name (e.g., 'deepseek_v3')."""
    base = OmegaConf.load(CONFIGS_DIR / "models" / "_base.yaml")
    profile = OmegaConf.load(CONFIGS_DIR / "models" / f"{name}.yaml")
    cfg = OmegaConf.merge(base, profile)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg

def load_training_config(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a training config by name (e.g., 'sft_base')."""
    cfg = OmegaConf.load(CONFIGS_DIR / "training" / f"{name}.yaml")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
```

---

## Phase 5: Trace Format Standardization (Pydantic)

### Current format (flat dict, ad-hoc keys)

```json
{
  "problem_id": "algorithms/a_star_grid",
  "model": "openai/gpt-oss-120b",
  "tool_name_0": "evaluate_cython",
  "tool_args_0": {"code": "..."},
  "observation_0": "...",
  "reasoning_0": "...",
  "tool_name_1": "evaluate_cython",
  "...": "...",
  "num_iterations": 3,
  "cython_code": "...",
  "reward": 0.87
}
```

### Target format (Pydantic models)

```python
# cnake_charmer/traces/models.py
from pydantic import BaseModel
from datetime import datetime

class ToolStep(BaseModel):
    tool_name: str
    tool_args: dict
    observation: str
    reasoning: str | None = None

class Trace(BaseModel):
    """Single agent trace: one problem, one attempt."""
    problem_id: str
    model: str
    prompt_id: str
    attempt: int = 0
    timestamp: datetime
    steps: list[ToolStep]          # Replaces flat tool_name_0, etc.
    final_code: str | None = None
    reward: float = 0.0
    metrics: dict = {}             # speedup, correctness, etc.
    thinking: bool = False
    version: str = "2.0"

class TraceCollection(BaseModel):
    """A file of traces with metadata."""
    traces: list[Trace]
    metadata: dict = {}
```

### Migration plan

1. **Back up**: `cp data/traces/master_thinking.jsonl data/traces/master_thinking_v1_backup.jsonl`
2. **Write converter**: Script that reads flat-key format, outputs Pydantic `Trace` objects
3. **Validate**: Load converted traces, verify count matches, spot-check 10 random traces
4. **Update build_sft.py**: Read new format, convert ToolSteps to Harmony messages
5. **Update collect_traces.py**: Write new format directly
6. **Keep v1 backup** until SFT dataset is rebuilt and validated from v2 traces

---

## Phase 6: Script Consolidation

### Merged: `collect_traces.py` + `generate_traces.py` → `scripts/collect_traces.py`

The merged script handles both workflows:
- **Collect mode** (default): Collect traces from a model, save all
- **Best-of-N mode** (`--best-of N`): Generate N candidates per problem, keep best

### Extracted shared utilities → `cnake_charmer/traces/lm.py`

```python
"""DRY utilities shared across trace collection scripts."""

def configure_dspy_lm(model_id: str, base_url: str = None, 
                       api_key: str = None, **kwargs) -> dspy.LM:
    """Auto-detect remote vs local, build LM."""

def load_prompt(model_id: str, program_path: str = None) -> CythonOptimization:
    """Load GEPA-optimized or seed prompt for a model."""

def apply_optimized_signatures(react_module, optimized_program):
    """Apply GEPA signatures to ReAct module."""

def model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug."""
```

**Current duplication eliminated:** ~130 lines of copy-paste across 4 scripts.

### Merged: `sft_scoring.py` + `sft_validation.py` → `cnake_charmer/training/sft.py`

Both are small (<250 lines each) and tightly related. One module handles SFT format concerns.

### Script inventory after refactor

| Script | Purpose | Config Source |
|--------|---------|---------------|
| `scripts/collect_traces.py` | Trace collection (merged) | `configs/models/{profile}.yaml` |
| `scripts/build_sft.py` | Build SFT dataset from traces | CLI args (data pipeline) |
| `scripts/train_sft.py` | SFT training | `configs/training/sft_*.yaml` |
| `scripts/train_grpo.py` | GRPO training | `configs/training/grpo_*.yaml` |
| `scripts/optimize_prompt.py` | GEPA prompt optimization | `configs/models/{profile}.yaml` |
| `scripts/test_model.py` | Model evaluation | `configs/models/{profile}.yaml` |
| `scripts/test_memory.py` | GPU memory smoke test | CLI args |
| `scripts/sample.py` | Manual sampling/inspection | `configs/models/{profile}.yaml` |
| `scripts/consolidate_traces.py` | Trace consolidation | CLI args |

---

## Phase 7: Makefile

```makefile
# ============================================================
# CnakeCharmer Makefile
# ============================================================

# --- Configuration ---
PROFILE ?= gpt_oss_120b
TRAINING_CONFIG ?= sft_base
UV_RUN := uv run --no-sync

# --- Data Collection ---
.PHONY: traces consolidate build-sft

traces:  ## Collect traces using a model profile
	$(UV_RUN) python scripts/collect_traces.py --profile $(PROFILE) --all --shuffle

traces-best:  ## Collect best-of-N traces
	$(UV_RUN) python scripts/collect_traces.py --profile $(PROFILE) --all --shuffle --best-of 5

consolidate:  ## Consolidate trace files into master JSONL
	$(UV_RUN) python scripts/consolidate_traces.py

build-sft:  ## Build SFT dataset from consolidated traces
	$(UV_RUN) python scripts/build_sft.py --min-score 0.8 --top-k 1 --require-finish

# --- Training ---
.PHONY: train-sft train-grpo

train-sft:  ## Run SFT training
	$(UV_RUN) python scripts/train_sft.py --config $(TRAINING_CONFIG)

train-grpo:  ## Run GRPO training
	$(UV_RUN) python scripts/train_grpo.py --config $(TRAINING_CONFIG)

# --- Evaluation ---
.PHONY: test-model benchmark sample

test-model:  ## Test trained model on unseen problems
	$(UV_RUN) python scripts/test_model.py --profile $(PROFILE)

benchmark:  ## Run benchmarks (hash-cached, only changed files)
	$(UV_RUN) python run_benchmarks.py

sample:  ## Sample problems manually for inspection
	$(UV_RUN) python scripts/sample.py --profile $(PROFILE)

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

lint:  ## Lint Cython files
	$(UV_RUN) cython-lint cnake_data/cy/

# --- Prompt Optimization ---
.PHONY: optimize-prompt

optimize-prompt:  ## Optimize prompt for a model profile
	$(UV_RUN) python scripts/optimize_prompt.py --profile $(PROFILE)

# --- Help ---
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
```

**Usage examples:**
```bash
make traces PROFILE=deepseek_v3        # Collect traces from DeepSeek
make traces PROFILE=glm5               # Collect traces from GLM-5
make train-sft TRAINING_CONFIG=sft_v3  # Train SFT with custom config
make benchmark                         # Run benchmarks
make test-data                         # Validate dataset only
```

---

## Phase 8: Execution Order

### Step 1: Safety (Phase 0)
- Tag, back up, compress old traces, verify tests pass

### Step 2: Archive & delete (Phase 1)
- Create archive branch, remove dead code, delete superseded files
- Verify: `pytest tests/ -x` still passes

### Step 3: Unified eval submodule (Phase 3)
- Create `cnake_charmer/eval/` with Pydantic models
- Migrate validate/ logic into eval/
- Migrate rewards/ weighting into eval/pipeline.py
- Add sandbox.py with subprocess isolation + safety checks
- Update MCP server, training environment, DSPy agent to use eval/
- Delete `cnake_charmer/validate/` and `cnake_charmer/rewards/`
- Verify: `pytest tests/ -x`, test MCP tools manually

### Step 4: Package split (Phase 2)
- Create `cnake_data/` with py/, cy/, benchmarks/, loader.py, difficulty.py
- Scripted import migration (1,350+ files)
- Move tests to `tests/data/` and `tests/tooling/`
- Update setup.py, run_benchmarks.py, pyproject.toml
- Verify: `pytest tests/ -x`, `make benchmark`, `make compile`

### Step 5: Trace format migration (Phase 5)
- Define Pydantic models in `cnake_charmer/traces/models.py`
- Write + run converter script on backed-up data
- Update collector and builder to use new format
- Verify: Rebuild SFT dataset, diff against previous version

### Step 6: Config management (Phase 4)
- Add OmegaConf dependency
- Create configs/ directory with model profiles + training configs
- Add `cnake_charmer/config.py` loader
- Update scripts to accept `--profile` / `--config` args
- Delete shell scripts
- Verify: `make traces PROFILE=gpt_oss_120b` works

### Step 7: Script consolidation (Phase 6)
- Merge collect + generate traces
- Extract shared LM utilities
- Merge SFT scoring + validation
- Verify: full pipeline end-to-end

### Step 8: Makefile (Phase 7)
- Create Makefile
- Verify all targets work

---

## Critical Data Handling Details

### SFT Dataset Format (target output of build_sft.py)

The SFT dataset is Harmony-rendered text. Each JSONL line contains:
```json
{
  "text": "<|start|>system<|message|>...<|end|>...",
  "model": "openai/gpt-oss-120b",
  "problem_id": "algorithms/a_star_grid",
  "func_name": "a_star_grid",
  "category": "algorithms",
  "difficulty": "hard",
  "num_iterations": 3,
  "sft_score": 0.87,
  "speedup": 40.73,
  "thinking": true,
  "reasoning_effort": "high"
}
```

The `"text"` field contains the complete Harmony conversation:
```
PREAMBLE (3 messages):
  system → auto-generated with "Reasoning: {effort}"
  developer → system_prompt + tool definitions
  user → python_code + func_name + description + test_cases

BODY (1+ turn groups):
  [optional] assistant analysis → reasoning/thinking
  assistant to=functions.evaluate_cython → JSON tool args
  functions.evaluate_cython to=assistant → JSON result

TERMINAL: Must end with a tool response (not standalone assistant)
```

### Composite Scoring Formula

```
IF compilation fails → score = 0.0

ELSE:
  correctness  = tests_passed / tests_total           (weight: 0.30)
  performance  = min(log2(speedup) / log2(100), 1.0)  (weight: 0.25)
  annotations  = typed_lines / total_lines             (weight: 0.20)
  lint         = max(0, 1 - 0.1 * violations)          (weight: 0.10)
  memory_safety = ASan score                            (weight: 0.15)
  
  score = weighted sum of above
```

GRPO uses curriculum-shifted weights (more correctness early, more performance late).

### Tool Schema (canonical)

```json
{
  "name": "evaluate_cython",
  "description": "Evaluate Cython code against Python reference",
  "parameters": {
    "code": "The Cython (.pyx) code to evaluate",
    "python_code": "The reference Python implementation",
    "test_code": "Python test code asserting equivalence"
  }
}
```

Single tool. No separate compile/annotate/test tools for the agent — the unified tool prevents reward hacking by always running the full pipeline.

### Safety Checks (from DSPy agent, now in eval/sandbox.py)

Blocked imports: `os`, `subprocess`, `shutil`, `pickle`, `socket`, `http`, `urllib`, `ctypes`, `importlib`, `__import__`

These prevent the model from generating malicious Cython code during evaluation.

---

## Verification Plan

After each phase, verify:

1. **Phase 0**: `pytest tests/ -x` passes
2. **Phase 1 (archive)**: `pytest tests/ -x` passes, archived items accessible on branch
3. **Phase 3 (eval)**: MCP tools work (`mcp__cnake-charmer__score_problem`, `evaluate_cython`, etc.), `pytest tests/ -x`
4. **Phase 2 (split)**: `make compile`, `make test-data`, `make test-tooling`, `make benchmark`
5. **Phase 5 (traces)**: Rebuild SFT dataset from converted traces, compare to existing `sft_dataset_v2.jsonl`
6. **Phase 4 (config)**: `make traces PROFILE=gpt_oss_120b` runs successfully
7. **Phase 6 (scripts)**: Full end-to-end: collect → consolidate → build-sft → train-sft
8. **Phase 7 (Makefile)**: All Makefile targets execute without errors

---

## Files Modified/Created Per Phase

| Phase | New Files | Modified Files | Deleted Files |
|-------|-----------|----------------|---------------|
| 0 | 0 | 0 | 0 (just tag + backup) |
| 1 | 0 | 0 | ~50 (archived/deleted) |
| 3 | ~10 (eval/) | ~5 (mcp_server, environment, dspy_agent) | ~13 (validate/ + rewards/) |
| 2 | ~3 (cnake_data init) | ~1,360 (imports) | ~5 (old locations) |
| 5 | ~2 (trace models, converter) | ~3 (collector, builder) | 0 |
| 4 | ~15 (configs/) + 1 (config.py) | ~8 (scripts) | ~29 (shell scripts) |
| 6 | ~2 (lm.py, merged sft.py) | ~6 (scripts) | ~2 (generate_traces, old sft files) |
| 7 | 1 (Makefile) | 0 | 0 |
