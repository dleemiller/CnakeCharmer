# evaluate_cython Tool Design

## Overview

The `evaluate_cython` tool is the core interface between the Cython optimization agent and the validation infrastructure. It accepts three parameters:

- **`code`**: Complete .pyx Cython source code
- **`python_code`**: Original Python source code (reference implementation)
- **`test_code`**: Equivalence test assertions

The tool compiles the Cython, runs the model's tests, and benchmarks against the Python reference.

## Test Code Format

Tests run in a namespace where `py` is the Python module and `cy` is the compiled Cython module. Lines containing `==` are assertions; other lines are setup.

```
py.sieve_primes(10) == cy.sieve_primes(10)
py.sieve_primes(100) == cy.sieve_primes(100)
```

Works for any exported name — functions, classes, constants:

```
obj_py = py.MyClass(5)
obj_cy = cy.MyClass(5)
obj_py.compute() == obj_cy.compute()
py.CONSTANT == cy.CONSTANT
```

Each assertion has a 5-second timeout. All results are collected (no fail-fast).

## Tool Response

```
## Compilation
Compilation successful.

## Annotation
Annotation score: 0.92 (3 Python-fallback lines / 36 total)

## Tests
Tests: 3/3 passed

## Benchmark
Speedup: 5.4x (Python: 0.000050s, Cython: 0.000009s)
```

## Training vs Production

| Aspect | Training (GRPO/SFT) | Production (MCP) |
|--------|---------------------|-------------------|
| Python reference | From dataset via `reset()` — model can't modify | From model's `python_code` param |
| Test execution | Model's `test_code` runs against ground truth Python | Model's `test_code` runs against model's Python |
| Reward signal | Based on ground truth equivalence | N/A (no reward) |

### Reward Hacking Prevention

During training, the environment stores the original Python code from the dataset at `reset()` time. The tool always uses this ground truth for equivalence checking, regardless of what `python_code` the model passes in the tool call. This prevents three attack vectors:

1. **Trivial tests** (`True == True`): The GRPO reward function runs auto-generated verifier tests using the original Python — model's tests don't affect the reward signal
2. **Weak tests** (only trivial inputs): Same — verifier uses its own inputs
3. **Modified reference** (model alters `python_code`): Tool silently uses the original from `reset()`, not the model's version

## File Locations

| File | Role |
|------|------|
| `data/tools.json` | Tool schema (3-param) |
| `data/system_prompt.txt` | Instructions for the model |
| `cnake_charmer/training/environment.py` | Tool implementation (training + production) |
| `cnake_charmer/mcp_server.py` | MCP server exposing the tool |
| `cnake_charmer/training/grpo.py` | GRPO reward function with anti-hacking verifier |
| `data/grpo_problems/` | Plain Python files for GRPO training (no tests, no Cython) |
