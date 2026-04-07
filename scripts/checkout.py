"""CLI for managing problem checkouts from bsmith925/cnake-charmer-stack-v2.

Usage examples:
    uv run python scripts/checkout.py checkout --worker bsmith925 --n 10
    uv run python scripts/checkout.py status
    uv run python scripts/checkout.py list --worker bsmith925
    uv run python scripts/checkout.py complete --worker bsmith925 --blob-id abc123 --problem-id algorithms/my_func
    uv run python scripts/checkout.py fail --worker bsmith925 --blob-id abc123 --notes "too complex"
    uv run python scripts/checkout.py abandon --worker bsmith925 --blob-id abc123
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.checkout import CheckoutManager


def _get_token(args) -> str | None:
    if args.token:
        return args.token
    return os.environ.get("HF_TOKEN")


def _manager(args) -> CheckoutManager:
    return CheckoutManager(token=_get_token(args))


def cmd_checkout(args):
    manager = _manager(args)
    print(f"Checking out {args.n} problems for worker '{args.worker}'...")
    rows = manager.checkout(worker_id=args.worker, n=args.n)

    output = Path(args.output or f"data/checkouts/{args.worker}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output.exists() else "w"
    with open(output, mode) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Checked out {len(rows)} problems → {output}")
    for row in rows:
        print(f"  {row['blob_id'][:12]}  {row.get('filename', '')}")


def cmd_complete(args):
    manager = _manager(args)
    manager.complete(
        blob_id=args.blob_id,
        worker_id=args.worker,
        problem_id=args.problem_id,
        verify_exists=not args.no_verify,
    )
    print(f"Marked {args.blob_id[:12]} as completed → {args.problem_id}")


def cmd_fail(args):
    manager = _manager(args)
    manager.fail(blob_id=args.blob_id, worker_id=args.worker, notes=args.notes or "")
    print(f"Marked {args.blob_id[:12]} as failed")


def cmd_abandon(args):
    manager = _manager(args)
    manager.abandon(blob_id=args.blob_id, worker_id=args.worker)
    print(f"Returned {args.blob_id[:12]} to pool")


def cmd_status(args):
    manager = _manager(args)
    s = manager.status(worker_id=args.worker)

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Checkout Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        table.add_row("Total candidates", str(s["total_candidates"]))
        table.add_row("Available", str(s["available"]))
        table.add_row("Checked out", str(s["checked_out"]))
        table.add_row("Completed", str(s["completed"]))
        table.add_row("Failed", str(s["failed"]))
        console.print(table)

        if s["by_worker"]:
            wtable = Table(title="By Worker")
            wtable.add_column("Worker", style="cyan")
            wtable.add_column("Checked out", justify="right")
            wtable.add_column("Completed", justify="right")
            wtable.add_column("Failed", justify="right")
            for worker, counts in sorted(s["by_worker"].items()):
                wtable.add_row(
                    worker,
                    str(counts.get("checked_out", 0)),
                    str(counts.get("completed", 0)),
                    str(counts.get("failed", 0)),
                )
            console.print(wtable)
    except ImportError:
        print(f"Total candidates : {s['total_candidates']}")
        print(f"Available        : {s['available']}")
        print(f"Checked out      : {s['checked_out']}")
        print(f"Completed        : {s['completed']}")
        print(f"Failed           : {s['failed']}")
        if s["by_worker"]:
            print("\nBy worker:")
            for worker, counts in sorted(s["by_worker"].items()):
                print(
                    f"  {worker}: checked_out={counts.get('checked_out', 0)} "
                    f"completed={counts.get('completed', 0)} "
                    f"failed={counts.get('failed', 0)}"
                )


def cmd_list(args):
    manager = _manager(args)
    records = manager.list_checked_out(worker_id=args.worker)

    if not records:
        print("No active checkouts.")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Active Checkouts{' for ' + args.worker if args.worker else ''}")
        table.add_column("blob_id", style="dim")
        table.add_column("worker")
        table.add_column("filename")
        table.add_column("claimed_at")
        table.add_column("expires_at")
        for r in records:
            table.add_row(
                r["blob_id"][:12],
                r["worker_id"],
                r.get("filename", ""),
                r["claimed_at"][:19],
                r["expires_at"][:19],
            )
        console.print(table)
    except ImportError:
        header = f"{'blob_id':<14} {'worker':<16} {'filename':<30} {'claimed_at':<20} expires_at"
        print(header)
        print("-" * len(header))
        for r in records:
            print(
                f"{r['blob_id'][:12]:<14} {r['worker_id']:<16} "
                f"{r.get('filename', ''):<30} {r['claimed_at'][:19]:<20} "
                f"{r['expires_at'][:19]}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Manage problem checkouts from the Stack v2 candidate pool."
    )
    parser.add_argument(
        "--token", help="HuggingFace token (default: HF_TOKEN env or huggingface-cli login)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # checkout
    p_co = sub.add_parser("checkout", help="Claim N problems for a worker")
    p_co.add_argument("--worker", required=True, help="Worker ID (e.g. bsmith925, agent-gemma4)")
    p_co.add_argument("--n", type=int, default=10, help="Number of problems to claim (default: 10)")
    p_co.add_argument(
        "--expiry-hours", type=int, default=48, help="Lease expiry in hours (default: 48)"
    )
    p_co.add_argument("--output", help="Output JSONL path (default: data/checkouts/{worker}.jsonl)")
    p_co.set_defaults(func=cmd_checkout)

    # complete
    p_done = sub.add_parser("complete", help="Mark a checkout as completed")
    p_done.add_argument("--worker", required=True)
    p_done.add_argument("--blob-id", required=True)
    p_done.add_argument("--problem-id", required=True, help="e.g. algorithms/my_func")
    p_done.add_argument("--no-verify", action="store_true", help="Skip verify_exists check")
    p_done.set_defaults(func=cmd_complete)

    # fail
    p_fail = sub.add_parser("fail", help="Mark a checkout as failed")
    p_fail.add_argument("--worker", required=True)
    p_fail.add_argument("--blob-id", required=True)
    p_fail.add_argument("--notes", help="Optional reason")
    p_fail.set_defaults(func=cmd_fail)

    # abandon
    p_ab = sub.add_parser("abandon", help="Return a problem to the pool")
    p_ab.add_argument("--worker", required=True)
    p_ab.add_argument("--blob-id", required=True)
    p_ab.set_defaults(func=cmd_abandon)

    # status
    p_st = sub.add_parser("status", help="Show checkout status summary")
    p_st.add_argument("--worker", help="Filter to a specific worker")
    p_st.set_defaults(func=cmd_status)

    # list
    p_ls = sub.add_parser("list", help="List active checkouts")
    p_ls.add_argument("--worker", help="Filter to a specific worker")
    p_ls.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
