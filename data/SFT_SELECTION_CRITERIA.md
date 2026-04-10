# SFT Data Selection Criteria

This document reflects the **current** `scripts/build_sft.py` + `cnake_charmer/training/sft_scoring.py` behavior.

## Hard Filters (current behavior)

1. **Known problem only**: `trace.problem_id` must exist in discovered pairs.
2. **Valid tool calls**: no `None` tool names in `tools_used`.
3. **Must use `evaluate_cython`** at least once.
4. **Must produce code**: final `cython_code` is non-empty.
5. **Nonzero trace reward**: `trace.reward != 0`.
6. **Perfect correctness for SFT score**: parsed correctness must be `1.0`, otherwise score is forced to `0.0`.

Optional CLI filters applied by `build_sft.py`:

- `--require-finish`: keep only traces that called `finish`.
- `--min-iters N`: minimum number of `evaluate_cython` calls.
- `--min-score S`: minimum SFT score threshold (default `0.8` in project workflows).

## SFT Scoring (current formula)

Primary quality:

- `correctness`: 0.25
- `performance`: 0.40, where performance uses `log2(speedup) / log2(100)` (capped at 1.0)
- `annotation`: 0.15
- `lint`: 0.05
- `memory_safety`: 0.05

Tiebreakers:

- `efficiency` (fewer `evaluate_cython` calls): 0.05
- `conciseness` (shorter thought text): 0.025
- `compactness` (shorter code): 0.025

Total score is quality + tiebreakers.

## Selection Strategy

After filtering:

1. Keep the best trace per `(problem_id, model)` by SFT score.
2. For each problem, keep top-`k` models by score (`--top-k`, default `2` in project workflows).
3. Validate trace structure for rendering.
4. Render with Harmony template and token-screen using `--max-tokens`.

## Important Notes

- There is **no explicit hard minimum** for speedup or annotation in `build_sft.py`.
- `finish` is only required when `--require-finish` is set.
- Token length and render validity can still remove traces after score filtering.
