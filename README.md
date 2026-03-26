# CnakeCharmer

A living dataset of parallel Python/Cython implementations for training AI models to translate Python → optimized Cython.

## Quickstart

```bash
# Install dependencies (first time only — also compiles all Cython extensions)
uv sync

# Run tests
uv run pytest tests/ -q

# Run benchmarks (parallel, with hash caching)
uv run run_benchmarks.py
```

After the initial `uv sync`, use `--no-sync` to skip recompiling everything:

```bash
# Day-to-day development — skip the full package rebuild
uv run --no-sync pytest tests/ -q
uv run --no-sync run_benchmarks.py

# Only rebuild when you change .pyx files
uv run --no-sync python setup.py build_ext --inplace
```

> **Why `--no-sync`?** `uv run` without it triggers `uv sync` which rebuilds the
> entire package including all 250+ Cython extensions. This takes several minutes.
> Use `--no-sync` for code-only changes and rebuild extensions separately when needed.

## Project Goals

LLMs can write decent Python but struggle with efficient Cython. This is a training data gap.

This repo is both the **dataset** and the **training infrastructure**:

- **Dataset**: 250+ matched Python/Cython pairs across 19 categories, version-controlled and CI-testable
- **Training**: Multi-turn GRPO with TRL GRPOTrainer where the model iteratively compiles, reviews HTML annotations, and optimizes its Cython output
- **Tools**: MCP server for AI-assisted development (compile, annotate, benchmark, score)

The Python/Cython syntax similarity makes translation tractable, and the compilation + benchmarking loop provides dense reward signal for RL training.

## Project Structure

```
cnake_charmer/
  py/{category}/{name}.py       ← Pure Python (training prompt)
  cy/{category}/{name}.pyx      ← Cython (ground truth baseline)
  cy_simd/{category}/{name}.pyx ← SIMD-optimized Cython (AVX2/FMA)
  pp/{category}/{name}.py       ← Pure Python Cython syntax (optional)
  validate/                     ← Compilation, annotation, correctness, benchmark tools
  rewards/                      ← Reward functions for GRPO training
  training/                     ← TRL GRPOTrainer integration
  dataset/                      ← Loader that discovers pairs from repo structure
  mcp_server.py                 ← MCP server for Claude Code
tests/
  {category}/test_{name}.py     ← Equivalence tests (Python == Cython output)
```

### Categories

`algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`, `geometry`, `simulation`, `graph`, `statistics`, `cryptography`, `nn_ops`, `image_processing`, `compression`, `leetcode`, `physics`, `diff_equations`, `dsp`, `optimization`

### Three Tiers

For compute-intensive operations (e.g. `nn_ops`), we provide three implementations:

| Tier | Directory | What it teaches |
|------|-----------|----------------|
| Python | `py/` | Naive baseline |
| Basic Cython | `cy/` | `cdef` types, C arrays, `libc.math` |
| SIMD Cython | `cy_simd/` | AVX2 intrinsics, cache tiling, FMA |

### Benchmarking

```bash
uv run --no-sync run_benchmarks.py         # 4 parallel workers, hash caching
uv run --no-sync run_benchmarks.py --all   # force re-run everything
uv run --no-sync run_benchmarks.py -j 8    # 8 workers
```

Benchmarks use source file hashing — only changed problems re-run. Results saved to `benchmarks.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide to adding new problem pairs.

## Training

The training pipeline uses TRL's GRPOTrainer with `environment_factory` for multi-turn tool-calling RL:

```python
from cnake_charmer.training.grpo import create_trainer
from cnake_charmer.dataset.loader import discover_pairs

trainer = create_trainer(
    model="Qwen/Qwen3-0.6B",
    problems=discover_pairs(),
    environment_factory=CythonToolEnvironment,
)
trainer.train()
```

The model learns to call `compile`, `annotate`, `test`, and `benchmark` tools to iteratively improve its Cython output.
