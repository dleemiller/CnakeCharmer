"""
Retry problems that failed or scored below threshold in a trace file.

Reads an existing JSONL trace file, identifies problems below threshold,
and re-runs just those problems with additional attempts.

Usage:
    # Retry problems below 0.8 reward from gemini run
    uv run python scripts/retry_failures.py \
        --input data/traces/gemini_pro_glm5prompt.jsonl \
        --threshold 0.8 --attempts 3

    # Retry problems with zero reward (failed completely)
    uv run python scripts/retry_failures.py \
        --input data/traces/gemini_pro_glm5prompt.jsonl \
        --threshold 0.01 --attempts 5

    # Just list failures without running
    uv run python scripts/retry_failures.py \
        --input data/traces/gemini_pro_glm5prompt.jsonl \
        --threshold 0.8 --dry-run
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_failures(input_path: str, threshold: float) -> list[str]:
    """Find problems where best SFT score is below threshold."""
    from cnake_charmer.training.sft_scoring import score_trace

    by_problem = defaultdict(list)
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r.get("problem_id", r.get("func_name", ""))
            sft = score_trace(r)
            by_problem[pid].append((sft, r.get("reward", 0) or 0))

    failures = []
    for pid, scores in sorted(by_problem.items()):
        best_sft = max(s[0] for s in scores)
        best_reward = max(s[1] for s in scores)
        if best_sft < threshold:
            failures.append(pid)
            logger.info(
                f"  {pid}: best_sft={best_sft:.3f} reward={best_reward:.3f} from {len(scores)} attempts"
            )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Retry problems below reward threshold")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL trace file to analyze")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Min reward threshold (default 0.8)"
    )
    parser.add_argument(
        "--attempts", type=int, default=3, help="Additional attempts per failed problem"
    )
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Just list failures, don't run")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Scanning {input_path} for problems below {args.threshold} reward...")
    failures = find_failures(str(input_path), args.threshold)

    if not failures:
        logger.info("No failures found — all problems above threshold!")
        return

    logger.info(f"\n{len(failures)} problems below threshold")

    if args.dry_run:
        return

    # Extract model and prompt_id from existing traces
    with open(input_path) as f:
        first = json.loads(f.readline())
    model = first.get("model", "")
    prompt_id = first.get("prompt_id", "seed_v1")

    logger.info(f"Retrying with model={model}, prompt_id={prompt_id}, attempts={args.attempts}")

    # Build collect_traces command
    problems_str = ",".join(failures)
    cmd = [
        sys.executable,
        "scripts/collect_traces.py",
        "--model",
        model,
        "--problems",
        problems_str,
        "--attempts",
        str(args.attempts),
        "--max-iters",
        str(args.max_iters),
        "--prompt-id",
        prompt_id,
        "-o",
        str(input_path),  # append to same file
    ]

    # Handle API key for remote models
    if model.startswith("openrouter/"):
        api_key = os.environ.get("APIKEY", os.environ.get("OPENROUTER_API_KEY", ""))
        if api_key:
            cmd.extend(["--api-key", api_key])

    import subprocess

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
