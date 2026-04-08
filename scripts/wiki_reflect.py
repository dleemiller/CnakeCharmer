#!/usr/bin/env python3
"""
Reflect on agent traces and extract patterns for wiki curation.

Usage:
    # Regex-only: all traces (fast, no LLM)
    python scripts/wiki_reflect.py --traces data/traces/master_traces.jsonl

    # Filtered by category
    python scripts/wiki_reflect.py --traces data/traces/master_traces.jsonl --category numerical

    # Recent traces only
    python scripts/wiki_reflect.py --traces data/traces/master_traces.jsonl --since 2026-03-01

    # Output to file
    python scripts/wiki_reflect.py --traces data/traces/master_traces.jsonl -o wiki/reflections.json

    # LLM-powered reflection (dry run)
    python scripts/wiki_reflect.py --traces data/traces/master_traces.jsonl --llm --dry-run

    # LLM-powered reflection (apply changes)
    python scripts/wiki_reflect.py --traces data/traces/master_traces.jsonl --llm --apply

    # LLM reflection on specific pages
    python scripts/wiki_reflect.py --traces data/traces/master_traces.jsonl --llm --apply --pages pitfalls,memoryviews
"""

import argparse
import json
import logging
import sys

from cnake_charmer.wiki.reflect import reflect_on_traces


def main():
    parser = argparse.ArgumentParser(description="Reflect on traces for wiki curation")
    parser.add_argument("--traces", required=True, help="Path to traces JSONL file")
    parser.add_argument("--category", help="Filter by problem category")
    parser.add_argument("--problem-id", help="Filter by specific problem ID")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Minimum reward threshold")
    parser.add_argument("--max-traces", type=int, default=2000, help="Max traces to analyze")
    parser.add_argument("--since", help="Only include traces after this date (YYYY-MM-DD)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # LLM reflection mode
    parser.add_argument(
        "--llm", action="store_true", help="Enable LLM-powered reflection (default: regex-only)"
    )
    parser.add_argument("--model", default=None, help="Model for LLM reflection")
    parser.add_argument("--base-url", default=None, help="API base URL for LLM")
    parser.add_argument(
        "--pages", default=None, help="Comma-separated pages to update (default: all with findings)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without writing"
    )
    parser.add_argument("--apply", action="store_true", help="Actually write wiki updates")
    parser.add_argument(
        "--max-iters", type=int, default=8, help="Max ReAct iterations per page (LLM mode)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.llm:
        _run_llm_reflection(args)
    else:
        _run_regex_reflection(args)


def _run_regex_reflection(args):
    """Original regex-only reflection."""
    findings = reflect_on_traces(
        traces_path=args.traces,
        category=args.category,
        problem_id=args.problem_id,
        min_reward=args.min_reward,
        max_traces=args.max_traces,
        since=args.since,
    )

    output = json.dumps(findings, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Wrote findings to {args.output}", file=sys.stderr)
    else:
        # Print summary to stderr, full JSON to stdout
        summary = findings.get("summary", {})
        print(f"Analyzed {summary.get('total_traces', 0)} traces", file=sys.stderr)
        print(
            f"  Traces with errors: {summary.get('traces_with_errors', 0)}",
            file=sys.stderr,
        )
        print(f"  Avg steps: {summary.get('avg_steps')}", file=sys.stderr)
        print(f"  Avg speedup: {summary.get('avg_speedup')}x", file=sys.stderr)
        print(f"  Avg annotation: {summary.get('avg_annotation')}", file=sys.stderr)

        n_pages = len(findings.get("findings", {}))
        print(f"  Findings across {n_pages} wiki pages", file=sys.stderr)
        print(file=sys.stderr)

        print(output)


def _run_llm_reflection(args):
    """LLM-powered reflection using DSPy ReAct agent."""
    if not args.apply and not args.dry_run:
        print("LLM mode requires --apply or --dry-run", file=sys.stderr)
        sys.exit(1)

    import dspy

    from cnake_charmer.wiki.llm_reflect import run_reflection

    # Configure LM
    if args.model:
        lm_kwargs = {}
        if args.base_url:
            lm_kwargs["api_base"] = args.base_url
        lm = dspy.LM(args.model, **lm_kwargs)
        dspy.configure(lm=lm)

    pages = args.pages.split(",") if args.pages else None

    results = run_reflection(
        traces_path=args.traces,
        model=args.model,
        category=args.category,
        since=args.since,
        max_traces=args.max_traces,
        pages=pages,
        dry_run=args.dry_run,
        max_iters=args.max_iters,
    )

    print(json.dumps(results, indent=2))

    updated = sum(1 for r in results.values() if r.get("updated"))
    print(f"\nUpdated {updated}/{len(results)} pages", file=sys.stderr)
    if args.dry_run:
        print("(dry run — no files were written)", file=sys.stderr)


if __name__ == "__main__":
    main()
