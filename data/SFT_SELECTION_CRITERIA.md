# SFT Data Selection Criteria

## Hard Filters (must pass all)

1. **Correctness**: All tests must pass (tests_passed == tests_total)
2. **Compiles**: Code must compile successfully
3. **No None tool calls**: All tool calls must be valid (`evaluate_cython` or `finish`), no parse failures
4. **Uses evaluate_cython**: Must have called `evaluate_cython` at least once (not just finish)
5. **No unsafe imports**: No `os`, `subprocess`, `shutil`, etc.
6. **Valid Cython output**: Final code must be non-empty and extractable

## Ranking Criteria (for selecting best per model per problem)

Priority order:

1. **Speedup** (higher is better) — primary differentiator since reward function caps at 10x
2. **Annotation score** (higher is better, target >0.9) — indicates well-typed C code
3. **Fewer iterations** (lower is better) — efficient problem-solving is better training signal
4. **Calls finish explicitly** — indicates the model knew it was done vs running out of iterations
5. **Shorter thought tokens** (lower is better) — concise reasoning, less training cost
6. **Shorter code** (lower is better, given same correctness+speedup) — Gemini's compact solutions preferred over verbose equivalent ones
7. **Higher reward** — tiebreaker after the above

## Selection Strategy

- **One best trace per model per problem** — diverse approaches from different models
- **Minimum speedup threshold**: >5x (exclude traces that barely beat Python)
- **Minimum annotation score**: >0.7 (exclude poorly typed code)
- **Prefer traces that finish in ≤3 iterations** over 5-iteration grinds, given comparable quality

## What NOT to include

- Traces with `None` tool calls (model failed to follow format)
- Traces from the old 4-tool format (compile_cython, annotate_cython, test_cython, benchmark_cython)
- Traces where reward = 0 (failed completely)
- Traces that fail any test case
- Traces with extremely high iteration count but low improvement (grinding without learning)

## SFT Training Weighting (optional)

Consider weighting examples by quality during training:
- High speedup (>50x) traces could have higher weight
- Low iteration (≤2 calls) traces could have higher weight
- These weights push the student toward efficient, high-performance solutions

## Diversity Requirements

- Include traces from multiple models per problem where available
- Different models teach different optimization strategies (typing vs algorithmic)
- At least one trace per problem in the final dataset (even if below ideal thresholds)
