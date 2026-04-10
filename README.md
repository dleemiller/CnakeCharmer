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
├── cnake_data/        # Dataset: Python/Cython pairs + unpaired GRPO problems
├── cnake_charmer/     # Core tooling: eval, training, traces, MCP server
├── configs/           # Model and training YAML profiles
├── scripts/           # Collection/training/benchmark/export entry points
├── tests/             # Dataset and tooling tests
├── docs/              # Project documentation
├── data/              # Traces, prompts, and HF export artifacts
└── Makefile           # Primary workflow interface
```

### Categories

`algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`, `geometry`, `simulation`, `graph`, `statistics`, `cryptography`, `nn_ops`, `image_processing`, `compression`, `leetcode`, `physics`, `diff_equations`, `dsp`, `optimization`

### Cython Feature Coverage

See [FEATURE_COVERAGE.md](docs/FEATURE_COVERAGE.md) for the full checklist and examples.

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

## Hugging Face Dataset

Dataset: https://huggingface.co/datasets/CnakeCharmer/CnakeCharmer

```python
from datasets import load_dataset

raw = load_dataset("CnakeCharmer/CnakeCharmer", split="raw")
sft = load_dataset("CnakeCharmer/CnakeCharmer", split="sft")
grpo = load_dataset("CnakeCharmer/CnakeCharmer", split="grpo")
parallel = load_dataset("CnakeCharmer/CnakeCharmer", split="parallel")
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
