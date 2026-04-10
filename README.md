# CnakeCharmer

A living dataset of parallel Python/Cython implementations for training AI models to translate Python → optimized Cython.

## Quickstart

```bash
# Install dependencies
uv sync

# Build Cython extensions
make compile

# Run tests
make test-tooling

# Run benchmarks (parallel, hash-cached)
make benchmark

# See all available commands
make help
```

> `uv sync` only installs Python dependencies — it does **not** compile Cython.
> Run `make compile` when you add or change `.pyx` files.

## Project Goals

LLMs can write decent Python but struggle with efficient Cython. This is a training data gap.

This repo is both the **dataset** and the **training infrastructure**:

- **Dataset**: 723 matched Python/Cython pairs across 19 categories, version-controlled and CI-testable
- **Training**: Multi-turn GRPO with TRL GRPOTrainer — the model iteratively compiles, reviews HTML annotations, and optimizes its Cython output
- **Tools**: MCP server for AI-assisted development (compile, annotate, benchmark, score, memory safety via ASan)

## Project Structure

```
CnakeCharmer/
├── cnake_data/                    # Living dataset
│   ├── py/{category}/{name}.py    # Pure Python (training prompts)
│   ├── cy/{category}/{name}.pyx   # Cython implementations (ground truth)
│   ├── unpaired/*.py              # GRPO unpaired Python problems
│   ├── benchmarks/                # Benchmark decorator registry
│   ├── loader.py                  # Problem pair discovery
│   └── difficulty.py              # Difficulty classification
│
├── cnake_charmer/                 # Tooling
│   ├── eval/                      # Unified evaluation pipeline
│   │   ├── compiler.py            # Cython compilation
│   │   ├── annotations.py         # HTML annotation parsing
│   │   ├── correctness.py         # Test execution
│   │   ├── benchmark.py           # Performance timing
│   │   ├── memory_safety.py       # ASan integration
│   │   ├── lint.py                # cython-lint
│   │   └── pipeline.py            # Orchestration + composite scoring
│   ├── training/                  # SFT + GRPO
│   │   ├── environment.py         # CythonToolEnvironment
│   │   ├── grpo.py                # GRPO reward + dataset
│   │   └── dspy_agent.py          # DSPy ReAct agent
│   ├── traces/                    # Trace collection + models
│   │   ├── models.py              # Pydantic trace format (v2)
│   │   ├── io.py                  # Load/save with auto-detect
│   │   └── lm.py                  # Shared LM configuration
│   ├── sources/                   # Data source adapters/loaders
│   ├── config.py                  # OmegaConf config loader
│   └── mcp_server.py              # MCP server for Claude Code
│
├── configs/
│   ├── models/                    # Model profiles (YAML)
│   └── training/                  # Training hyperparameters (YAML)
│
├── tests/
│   ├── data/{category}/           # Dataset equivalence tests
│   └── tooling/                   # Tooling unit tests
│
├── scripts/                       # CLI entry points + utilities
│   ├── collect_traces.py          # DSPy trace collection
│   ├── build_sft.py               # SFT dataset builder
│   ├── export_parallel_pairs.py   # Parallel Python/Cython export
│   ├── run_benchmarks.py          # Benchmark runner
│   └── utils/stack_data/          # Stack data tooling/artifacts
│
├── docs/
│   ├── BENCHMARKS.md
│   ├── FEATURE_COVERAGE.md
│   ├── CONTRIBUTING.md
│   ├── TOOL_DESIGN.md
│   └── SFT_SELECTION_CRITERIA.md
│
├── data/                          # Traces, prompts, HF dataset assets
│   ├── traces/                    # Master trace logs + error logs
│   └── hf/                        # raw/sft/grpo/parallel exports
└── Makefile                       # Primary workflow interface
```

### Categories

`algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`, `geometry`, `simulation`, `graph`, `statistics`, `cryptography`, `nn_ops`, `image_processing`, `compression`, `leetcode`, `physics`, `diff_equations`, `dsp`, `optimization`

### Cython Feature Coverage

The problem set covers a broad range of Cython features beyond basic typed functions:

| Feature | Examples |
|---------|---------|
| `cdef class` (extension types) | `__cinit__`/`__dealloc__`, typed C attributes, `cdef`/`cpdef` methods |
| Special methods | `__getitem__`, `__setitem__`, `__len__`, `__contains__`, `__iter__`/`__next__`, `__richcmp__`, `__call__`, `__hash__` |
| Inheritance | `cdef class Child(Parent)` with `cpdef` method dispatch |
| Typed memoryviews | 1D/2D, C-contiguous `[::1]`, Fortran `[::1, :]`, `const`, `.T`, `.copy()` |
| `cdef struct` / `cdef union` | Nested structs, packed structs, tagged unions, struct return |
| Fused types | `ctypedef fused` with type dispatch, memoryview params |
| C++ interop | `libcpp.vector`/`map`/`set`/`unordered_map`, `cdef cppclass`, `except +` |
| `prange` / `nogil` | OpenMP parallel loops, GIL release for C computation |
| NumPy interop | `cimport numpy`, typed memoryviews from arrays, prange+NumPy |

See [FEATURE_COVERAGE.md](docs/FEATURE_COVERAGE.md) for the full checklist.

### Benchmarks

```bash
make benchmark              # 4 parallel workers, hash caching
uv run scripts/run_benchmarks.py --all   # force re-run everything
uv run scripts/run_benchmarks.py -j 8    # 8 workers
```

## Configuration

Model profiles and training configs live in `configs/`:

```bash
# Collect traces with a specific model profile
make traces PROFILE=deepseek_v3

# Override config values via CLI
uv run --no-sync python -c "
from cnake_charmer.config import load_model_profile
cfg = load_model_profile('gpt_oss_120b', overrides=['model.temperature=0.8'])
"
```

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full guide to adding new problem pairs.

## Training

The training pipeline uses TRL's GRPOTrainer with `environment_factory` for multi-turn tool-calling RL:

```python
from cnake_data.loader import discover_pairs
from cnake_charmer.training.grpo import create_trainer

trainer = create_trainer(
    model="openai/gpt-oss-20b",
    problems=discover_pairs(),
)
trainer.train()
```

The model learns to call `evaluate_cython` to iteratively compile, test, and benchmark its Cython output.

### Reward Signals

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Correctness | 30% | py/cy output equivalence across test cases |
| Performance | 25% | log-scaled speedup vs Python baseline |
| Annotations | 20% | Ratio of pure-C lines in Cython HTML annotations |
| Memory safety | 15% | AddressSanitizer (leaks, overflows, use-after-free) |
| Lint | 10% | cython-lint violations |
