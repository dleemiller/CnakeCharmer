"""
Collect traces from any model/prompt combo into a standardized JSONL format.

Output files can be concatenated, then filtered/combined using dspy-data tools.
Every trace includes model, prompt_id, problem metadata for full traceability.

Usage:
    # 10 random problems with seed prompt on local gpt-oss
    uv run python scripts/collect_traces.py \
        --model openai/gpt-oss-120b \
        --n-random 10

    # Specific problems with a saved program
    uv run python scripts/collect_traces.py \
        --model openrouter/z-ai/glm-5 \
        --problems algorithms/primes,nn_ops/gemm \
        --program data/optimized_prompts/openai_gpt-oss-120b/program.json

    # All problems, 5 attempts each
    uv run python scripts/collect_traces.py \
        --model openai/gpt-oss-120b \
        --all --attempts 5 \
        --output data/traces/gptoss_seed_all.jsonl

    # Failed problems from another run
    uv run python scripts/collect_traces.py \
        --model openrouter/z-ai/glm-5 \
        --problems-from-file failed_problems.txt \
        --attempts 10
"""

import argparse
import json
import logging
import os
import random
import sys
import traceback
from collections import Counter, deque
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))

from cnake_charmer.logging_config import setup_logging
from cnake_charmer.traces.io import append_trace
from cnake_charmer.traces.lm import (
    apply_optimized_signatures,
    configure_dspy_lm,
    get_seed_text,
    load_optimized_prompt,
    model_slug,
)
from cnake_charmer.traces.models import ToolStep, Trace
from cnake_charmer.training.dspy_agent import CythonOptimization, make_tools
from cnake_charmer.training.rollout import extract_code_from_content
from cnake_data.loader import discover_pairs

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


def _parse_tool_args_raw(args) -> dict:
    """Normalize tool args from DSPy trajectory to dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"raw": args}
    return {}


def _tool_call_counts(trace: Trace) -> str:
    """Format per-tool call counts for logging.

    For evaluate_cython, distinguish executed calls from blocked attempts
    (recorded decision with empty observation).
    """
    counts = Counter()
    eval_blocked = 0
    for s in trace.steps:
        if s.tool_name == "evaluate_cython" and not (s.observation or "").strip():
            eval_blocked += 1
            continue
        counts[s.tool_name] += 1

    parts = [f"{name}={n}" for name, n in counts.most_common()]
    if eval_blocked:
        parts.append(f"evaluate_cython_blocked={eval_blocked}")
    return " ".join(parts)


def make_trace(
    problem,
    result,
    model: str,
    prompt_id: str,
    attempt: int,
    lm_history: list | None = None,
) -> Trace:
    """Build a v2 Trace object from a DSPy ReAct result."""
    traj = result.trajectory if result.trajectory else {}
    cython_code = extract_code_from_content(getattr(result, "cython_code", "") or "")

    # Fallback: extract code from the last tool call if output field is empty.
    # Some models (e.g. Gemma 4) don't produce a clean final output with the
    # cython_code field, but the code is in the tool call trajectory.
    if not cython_code:
        for i in range(20, -1, -1):
            args = traj.get(f"tool_args_{i}")
            if args and isinstance(args, dict) and "code" in args:
                cython_code = args["code"]
                break
            elif args and isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if "code" in parsed:
                        cython_code = parsed["code"]
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

    # Build reasoning map from LM history (thinking models)
    reasoning_map = {}  # idx -> reasoning_content
    if lm_history:
        for idx, interaction in enumerate(lm_history):
            response = interaction.get("response")
            if response is None:
                continue
            for choice in getattr(response, "choices", []):
                rc = getattr(getattr(choice, "message", None), "reasoning_content", None)
                if rc:
                    reasoning_map[idx] = rc

    # Parse tool steps
    steps = []
    i = 0
    while f"tool_name_{i}" in traj:
        tool_name = traj[f"tool_name_{i}"]
        if tool_name is not None:
            tool_name_s = str(tool_name)
            steps.append(
                ToolStep(
                    tool_name=tool_name_s,
                    tool_args=_parse_tool_args_raw(traj.get(f"tool_args_{i}", {})),
                    observation=str(traj.get(f"observation_{i}", "")),
                    thought=traj.get(f"thought_{i}"),
                    reasoning=reasoning_map.get(i),
                )
            )
        i += 1

    # Collect trailing reasoning entries (beyond tool steps)
    n_steps = len(steps)
    trailing = []
    j = n_steps
    while j in reasoning_map:
        trailing.append(reasoning_map[j])
        j += 1

    thinking = bool(reasoning_map)

    # If final output field is empty, prefer last evaluate_cython code.
    if not cython_code:
        for step in reversed(steps):
            if step.tool_name == "evaluate_cython":
                code = step.tool_args.get("code")
                if isinstance(code, str) and code.strip():
                    cython_code = code
                    break

    return Trace(
        problem_id=problem.problem_id,
        model=model,
        prompt_id=prompt_id,
        attempt=attempt,
        timestamp=datetime.now(UTC),
        steps=steps,
        trailing_reasoning=trailing,
        final_code=cython_code,
        reward=0.0,
        thinking=thinking,
        func_name=problem.func_name,
        category=problem.category,
        difficulty=problem.difficulty,
    )


def score_trace(trace: Trace, problem) -> float:
    """Score the generated cython code using composite reward.

    Also populates trace.metrics with detailed scoring breakdown.
    """
    from cnake_charmer.training.dspy_agent import _safe_composite_reward

    code = trace.final_code or ""
    if not code:
        return 0.0

    scores = _safe_composite_reward(
        cython_code=code,
        python_code=problem.python_code,
        func_name=problem.func_name,
        test_cases=problem.test_cases,
        benchmark_args=problem.benchmark_args,
    )
    trace.metrics = {
        "compiled": scores.get("compiled", False),
        "correctness": round(scores.get("correctness", 0.0), 3),
        "speedup": round(scores.get("speedup", 0.0), 2),
        "annotations": round(scores.get("annotations", 0.0), 3),
    }
    return scores["total"]


def _format_trace_log(trace: Trace) -> str:
    """Format a single trace result for logging."""
    m = trace.metrics
    parts = [f"reward={trace.reward:.3f}"]

    if m.get("compiled"):
        parts.append(f"correct={m.get('correctness', 0):.0%}")
        parts.append(f"speedup={m.get('speedup', 0):.1f}x")
        parts.append(f"ann={m.get('annotations', 0):.3f}")
    else:
        parts.append("COMPILE_FAIL")

    parts.append(f"iters={trace.num_iterations}")
    parts.append(f"tools=[{_tool_call_counts(trace)}]")
    return " ".join(parts)


class RollingErrorLog:
    """Append timestamped error events to a rotating log file."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        max_mb: int = 10,
        backups: int = 5,
    ):
        self.path = Path(path) if path else None
        self._logger = None
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"{__name__}.errors.{os.getpid()}")
        self._logger.setLevel(logging.ERROR)
        self._logger.propagate = False
        self._logger.handlers.clear()
        self._logger.addHandler(
            RotatingFileHandler(
                self.path,
                maxBytes=max_mb * 1024 * 1024,
                backupCount=backups,
                encoding="utf-8",
            )
        )

    def record(
        self,
        *,
        model: str,
        prompt_id: str,
        problem_id: str,
        attempt_number: int,
        error: str,
        traceback_text: str | None = None,
    ):
        if self._logger is None:
            return
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "model": model,
            "prompt_id": prompt_id,
            "problem_id": problem_id,
            "attempt": attempt_number,
            "error": str(error),
        }
        if traceback_text:
            payload["traceback"] = traceback_text
        self._logger.error(json.dumps(payload, ensure_ascii=False))


class TraceDisplay:
    """Render attempt results as plain logs, row output, or a live dashboard."""

    def __init__(self, mode: str, total: int):
        self.mode = mode
        self.total = total
        self.processed = 0
        self.saved = 0
        self.compiled = 0
        self.failed = 0
        self.reward_sum = 0.0
        self.speedup_sum = 0.0
        self.speedup_count = 0
        self.recent = deque(maxlen=16)

        self.console = None
        self._live = None
        self._progress = None
        self._task_id = None
        self._rich_table = None
        self._rich_panel = None
        self._rich_group = None

        if self.mode in {"table", "live"}:
            try:
                from rich.console import Console, Group
                from rich.live import Live
                from rich.panel import Panel
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )
                from rich.table import Table

                self.console = Console(stderr=True)
                self._rich_table = Table
                self._rich_panel = Panel
                self._rich_group = Group
                if self.mode == "live":
                    self._progress = Progress(
                        TextColumn("[bold cyan]Trace Collection[/bold cyan]"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                        console=self.console,
                    )
                    self._task_id = self._progress.add_task("traces", total=max(1, total))
                    self._live = Live(
                        self._render_live(),
                        console=self.console,
                        refresh_per_second=4,
                        transient=False,
                    )
                    self._live.start()
                else:
                    self.console.print(
                        "[bold]idx[/bold]  [bold]problem[/bold]  [bold]try[/bold]  "
                        "[bold]reward[/bold]  [bold]ok[/bold]  [bold]speedup[/bold]  "
                        "[bold]ann[/bold]  [bold]iters[/bold]  [bold]tools[/bold]"
                    )
            except ImportError:
                logger.warning(
                    "Rich display requested but Rich is unavailable; falling back to log."
                )
                self.mode = "log"

    @staticmethod
    def _reward_style(reward: float) -> str:
        if reward >= 0.95:
            return "green"
        if reward >= 0.8:
            return "yellow"
        return "red"

    @staticmethod
    def _clip(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        return text[: width - 1] + "…"

    @staticmethod
    def _sanitize(text: str) -> str:
        """Collapse multiline/noisy error text into a stable single-line string."""
        return " ".join(str(text).split())

    def _build_row(
        self,
        problem_id: str,
        attempt_number: int,
        trace: Trace | None = None,
        error: str | None = None,
    ) -> dict[str, str]:
        if trace is None:
            return {
                "idx": f"{self.processed}/{self.total}",
                "problem": problem_id,
                "attempt": str(attempt_number),
                "reward": "-",
                "ok": "ERR",
                "speedup": "-",
                "ann": "-",
                "iters": "-",
                "tools": self._clip(self._sanitize(error or "error"), 48),
                "style": "bold red",
            }

        m = trace.metrics
        compiled = bool(m.get("compiled"))
        reward = trace.reward
        return {
            "idx": f"{self.processed}/{self.total}",
            "problem": problem_id,
            "attempt": str(attempt_number),
            "reward": f"{reward:.3f}",
            "ok": f"{m.get('correctness', 0):.0%}" if compiled else "FAIL",
            "speedup": f"{m.get('speedup', 0):.1f}x" if compiled else "-",
            "ann": f"{m.get('annotations', 0):.3f}" if compiled else "-",
            "iters": str(trace.num_iterations),
            "tools": self._clip(_tool_call_counts(trace).replace("=", "×"), 48),
            "style": self._reward_style(reward) if compiled else "bold red",
        }

    def _summary_line(self) -> str:
        avg_reward = self.reward_sum / self.saved if self.saved else 0.0
        avg_speedup = self.speedup_sum / self.speedup_count if self.speedup_count else 0.0
        return (
            f"saved={self.saved}  processed={self.processed}/{self.total}  "
            f"compiled={self.compiled}  errors={self.failed}  "
            f"avg_reward={avg_reward:.3f}  avg_speedup={avg_speedup:.1f}x"
        )

    def _render_recent_table(self):
        table = self._rich_table(title="Recent Attempts", expand=True)
        table.add_column("idx", style="dim", justify="right", no_wrap=True)
        table.add_column("problem", style="cyan")
        table.add_column("try", justify="right", no_wrap=True)
        table.add_column("reward", justify="right", no_wrap=True)
        table.add_column("ok", justify="right", no_wrap=True)
        table.add_column("speedup", justify="right", no_wrap=True)
        table.add_column("ann", justify="right", no_wrap=True)
        table.add_column("iters", justify="right", no_wrap=True)
        table.add_column("tools", style="dim")
        for row in self.recent:
            table.add_row(
                row["idx"],
                row["problem"],
                row["attempt"],
                f"[{row['style']}]{row['reward']}[/]",
                f"[{row['style']}]{row['ok']}[/]",
                row["speedup"],
                row["ann"],
                row["iters"],
                row["tools"],
            )
        return table

    def _render_live(self):
        return self._rich_group(
            self._rich_panel(self._summary_line(), title="Run Stats"),
            self._progress,
            self._render_recent_table(),
        )

    def _advance(self):
        if self.mode == "live" and self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id, 1)
            self._live.update(self._render_live())

    def on_success(self, problem_id: str, attempt_number: int, trace: Trace):
        self.processed += 1
        self.saved += 1
        self.reward_sum += trace.reward

        compiled = bool(trace.metrics.get("compiled"))
        if compiled:
            self.compiled += 1
            self.speedup_sum += trace.metrics.get("speedup", 0.0)
            self.speedup_count += 1

        row = self._build_row(problem_id, attempt_number, trace=trace)
        self.recent.append(row)
        self._advance()

        if self.mode == "log":
            logger.info(
                f"[{self.processed}/{self.total}] {problem_id} attempt {attempt_number}: "
                f"{_format_trace_log(trace)}"
            )
            return

        if self.mode == "table" and self.console is not None:
            self.console.print(
                f"[dim]{row['idx']:>9}[/]  [cyan]{self._clip(row['problem'], 44)}[/]  "
                f"[white]{row['attempt']:>3}[/]  [{row['style']}]{row['reward']:>6}[/]  "
                f"[{row['style']}]{row['ok']:>5}[/]  [white]{row['speedup']:>8}[/]  "
                f"[white]{row['ann']:>5}[/]  [white]{row['iters']:>5}[/]  [dim]{row['tools']}[/]"
            )

    def on_error(self, problem_id: str, attempt_number: int, error: Exception | str):
        self.processed += 1
        self.failed += 1
        row = self._build_row(problem_id, attempt_number, error=self._sanitize(str(error)))
        self.recent.append(row)
        self._advance()

        if self.mode == "log":
            logger.error(
                f"[{self.processed}/{self.total}] {problem_id} attempt {attempt_number}: "
                f"{self._sanitize(str(error))}"
            )
            return

        if self.mode == "table" and self.console is not None:
            self.console.print(
                f"[dim]{row['idx']:>9}[/]  [cyan]{self._clip(row['problem'], 44)}[/]  "
                f"[white]{row['attempt']:>3}[/]  [bold red]{row['reward']:>6}[/]  "
                f"[bold red]{row['ok']:>5}[/]  [white]{row['speedup']:>8}[/]  "
                f"[white]{row['ann']:>5}[/]  [white]{row['iters']:>5}[/]  [red]{row['tools']}[/]"
            )

    def finish(self):
        if self.mode == "live" and self._live is not None:
            self._live.update(self._render_live())
            self._live.stop()
            if self.console is not None:
                self.console.print(self._rich_panel(self._summary_line(), title="Final Stats"))
        elif self.mode == "table" and self.console is not None:
            self.console.print(f"\n[bold]{self._summary_line()}[/bold]")


def run_problem(
    problem,
    model_id,
    max_iters,
    optimized_program,
    seed_text,
    use_thinking=False,
    include_wiki=False,
    workflow: str | None = None,
):
    """Run a single problem and return (result, lm_history)."""
    from copy import deepcopy

    tools, _env = make_tools(
        problem.python_code,
        problem.func_name,
        problem.test_cases,
        problem.benchmark_args,
        include_wiki=include_wiki,
    )
    workflow_mode = "LC" if workflow in {"LC", "low_certainty"} else "HC"
    # Budget max_iters as evaluate_cython calls; reserve slots for wiki + finish.
    react_max_iters = max_iters + (2 if include_wiki else 0) + 1
    if use_thinking:
        from cnake_charmer.traces.thinking_react import ThinkingReAct

        react = ThinkingReAct(
            CythonOptimization,
            tools=tools,
            max_iters=react_max_iters,
            max_evaluations=max_iters,
        )
    else:
        from cnake_charmer.traces.budgeted_react import BudgetedReAct

        react = BudgetedReAct(
            CythonOptimization,
            tools=tools,
            max_iters=react_max_iters,
            max_evaluations=max_iters,
        )
    apply_optimized_signatures(react, optimized_program, seed_text)
    thread_local_lm = deepcopy(dspy.settings.lm)
    thread_local_lm.history = []
    with dspy.context(lm=thread_local_lm):
        result = react(
            python_code=problem.python_code,
            func_name=problem.func_name,
            description=problem.description or "",
            workflow_mode=workflow_mode,
        )
    return result, thread_local_lm.history


def main():
    parser = argparse.ArgumentParser(description="Collect traces into standardized JSONL format")

    # Model
    parser.add_argument(
        "--model", default="openai/gpt-oss-120b", help="Model ID (default: local gpt-oss-120b)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (auto-detected: localhost for local, omit for OpenRouter)",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)

    # Prompt
    parser.add_argument(
        "--program", default=None, help="Path to GEPA program.json (default: seed prompt)"
    )
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Override prompt ID label (default: auto from program path)",
    )

    # Problems
    parser.add_argument("--n-random", type=int, default=None, help="Run N random problems")
    parser.add_argument("--problems", default=None, help="Comma-separated problem IDs")
    parser.add_argument(
        "--problems-from-file", default=None, help="File with one problem ID per line"
    )
    parser.add_argument("--all", action="store_true", help="Run all problems")
    parser.add_argument("--shuffle", action="store_true", help="Randomize problem order")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high"],
        help="Set reasoning_effort for thinking models (e.g. Mistral Small 4)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty",
    )

    # Execution
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per problem (default: 1)")
    parser.add_argument(
        "--max-iters", type=int, default=4, help="Max evaluate_cython calls per attempt"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Run N attempts concurrently via dspy.Parallel (use with vLLM prefix caching)",
    )

    parser.add_argument(
        "--thinking-react",
        action="store_true",
        default=False,
        help="Use ThinkingReAct (native LM thinking) instead of standard ReAct",
    )
    parser.add_argument(
        "--enable-wiki",
        action="store_true",
        default=False,
        help="Add wiki_read tool (capped at 2 calls per problem)",
    )
    parser.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help='JSON extra_body for LM (e.g. \'{"chat_template_kwargs": {"enable_thinking": true}}\')',
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSONL path (default: data/traces/{model}_{prompt}.jsonl)",
    )
    parser.add_argument(
        "--display",
        choices=["log", "table", "live"],
        default="log",
        help="Attempt output mode: log (default), table rows, or live dashboard",
    )
    parser.add_argument(
        "--sandbox-events",
        action="store_true",
        default=False,
        help="Show verbose sandbox lifecycle logs (sandbox.start/sandbox.complete)",
    )
    parser.add_argument(
        "--error-log",
        default="data/traces/trace_errors.log",
        help="Path for rotating JSONL error log (set to 'none' to disable)",
    )
    parser.add_argument(
        "--error-log-max-mb",
        type=int,
        default=10,
        help="Max size in MB before rotating error log (default: 10)",
    )
    parser.add_argument(
        "--error-log-backups",
        type=int,
        default=5,
        help="Number of rotated error log files to keep (default: 5)",
    )

    args = parser.parse_args()
    os.environ["CNAKE_SANDBOX_LOG_EVENTS"] = "1" if args.sandbox_events else "0"
    if not args.sandbox_events:
        logging.getLogger("cnake_charmer.eval.sandbox").setLevel(logging.WARNING)

    # Configure LM (shared utility handles remote vs local detection)
    lm_extra = {}
    if args.extra_body:
        lm_extra["extra_body"] = args.extra_body
    configure_dspy_lm(
        args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
        **lm_extra,
    )
    logger.info(f"Model: {args.model}")

    # Load prompt
    optimized_program, prompt_id = load_optimized_prompt(
        model_id=args.model, program_path=args.program
    )
    if args.prompt_id:
        prompt_id = args.prompt_id
    seed_text = get_seed_text() if not optimized_program else None

    # Resolve problems
    all_problems = {p.problem_id: p for p in discover_pairs()}
    if args.difficulty:
        all_problems = {k: v for k, v in all_problems.items() if v.difficulty == args.difficulty}

    if args.problems:
        problem_ids = [p.strip() for p in args.problems.split(",")]
        problems = [all_problems[pid] for pid in problem_ids if pid in all_problems]
    elif args.problems_from_file:
        with open(args.problems_from_file) as f:
            problem_ids = [line.strip() for line in f if line.strip()]
        problems = [all_problems[pid] for pid in problem_ids if pid in all_problems]
    elif args.all:
        problems = list(all_problems.values())
    elif args.n_random:
        problems = random.sample(list(all_problems.values()), min(args.n_random, len(all_problems)))
    else:
        problems = random.sample(list(all_problems.values()), min(10, len(all_problems)))

    logger.info(f"Problems: {len(problems)}, Attempts: {args.attempts}, Prompt: {prompt_id}")
    if args.enable_wiki:
        logger.info("Wiki tool enabled: wiki_read (2 calls max per attempt)")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        slug = model_slug(args.model)
        output_path = Path(f"data/traces/{slug}_{prompt_id}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path = (
        None if str(args.error_log).lower() in {"none", "off", "false"} else args.error_log
    )
    error_log = RollingErrorLog(
        error_log_path,
        max_mb=max(1, args.error_log_max_mb),
        backups=max(1, args.error_log_backups),
    )

    # Resume: count existing traces per problem for this model
    # Normalize model names: strip :free suffix and -preview so variants match
    def normalize_model(m: str) -> str:
        return m.removesuffix(":free").removesuffix("-preview")

    existing_counts = Counter()
    model_norm = normalize_model(args.model)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if normalize_model(r.get("model", "")) == model_norm:
                            existing_counts[r.get("problem_id", "")] += 1
                    except json.JSONDecodeError:
                        pass
        if existing_counts:
            logger.info(
                f"Resuming: {sum(existing_counts.values())} existing traces for {args.model} across {len(existing_counts)} problems"
            )

    # Build work list, skipping completed problems
    work = []
    skipped = 0
    for problem in problems:
        existing = existing_counts.get(problem.problem_id, 0)
        remaining = args.attempts - existing
        if remaining <= 0:
            skipped += 1
            continue
        for attempt in range(existing, existing + remaining):
            work.append((problem, attempt))

    if skipped:
        logger.info(f"Skipping {skipped} complete problems, {len(work)} traces remaining")

    if args.shuffle:
        random.shuffle(work)

    # Run
    total = len(work)
    done = 0
    wiki_used = 0
    display = TraceDisplay(args.display, total)
    workflow_by_attempt: dict[tuple[str, int], str] = {}
    if args.parallel and total:
        # Alternate workflow modes within each problem's attempts so
        # a 2-attempt problem gets one LC and one HC when both are present.
        attempts_by_problem: dict[str, list[int]] = {}
        for problem, attempt in work:
            attempts_by_problem.setdefault(problem.problem_id, []).append(attempt)
        for problem_id, attempts in attempts_by_problem.items():
            for idx, attempt in enumerate(sorted(attempts)):
                workflow_by_attempt[(problem_id, attempt)] = "LC" if idx % 2 == 0 else "HC"
        low_n = sum(1 for w in workflow_by_attempt.values() if w == "LC")
        high_n = total - low_n
        logger.info(f"Parallel workflow split: low_certainty={low_n}, high_confidence={high_n}")

    if args.parallel:
        # Parallel execution using dspy.Parallel — ideal for vLLM prefix caching.
        # Groups attempts by problem so same-prefix requests hit the KV cache.
        logger.info(f"Running {total} traces with {args.parallel} parallel threads")

        # Group work by problem — shuffle already controls problem order
        work_by_pid = {}
        pid_order = []
        for problem, attempt in work:
            pid = problem.problem_id
            if pid not in work_by_pid:
                work_by_pid[pid] = []
                pid_order.append(pid)
            work_by_pid[pid].append((problem, attempt))

        for pid in pid_order:
            group_items = work_by_pid[pid]
            problem = group_items[0][0]

            # Build exec pairs: each attempt gets its own module + example
            exec_pairs = []
            for prob, attempt in group_items:
                module = dspy.Predict("question -> answer")  # placeholder, run_problem handles it
                workflow = workflow_by_attempt.get((prob.problem_id, attempt), "HC")
                example = dspy.Example(
                    problem=prob,
                    attempt=attempt,
                    model_id=args.model,
                    max_iters=args.max_iters,
                    optimized_program=optimized_program,
                    seed_text=seed_text,
                    use_thinking=args.thinking_react,
                    include_wiki=args.enable_wiki,
                    workflow=workflow,
                ).with_inputs(
                    "problem",
                    "attempt",
                    "model_id",
                    "max_iters",
                    "optimized_program",
                    "seed_text",
                    "use_thinking",
                    "include_wiki",
                    "workflow",
                )
                exec_pairs.append((module, example))

            # Define a wrapper that dspy.Parallel can call
            class TraceRunner:
                def __call__(
                    self,
                    problem,
                    attempt,
                    model_id,
                    max_iters,
                    optimized_program,
                    seed_text,
                    use_thinking=False,
                    include_wiki=False,
                    workflow="HC",
                    **kwargs,
                ):
                    try:
                        result, lm_history = run_problem(
                            problem,
                            model_id,
                            max_iters,
                            optimized_program,
                            seed_text,
                            use_thinking=use_thinking,
                            include_wiki=include_wiki,
                            workflow=workflow,
                        )
                        return dspy.Example(
                            result=result,
                            lm_history=lm_history,
                            problem=problem,
                            attempt=attempt,
                            workflow=workflow,
                            error=None,
                        )
                    except Exception as e:
                        return dspy.Example(
                            result=None,
                            lm_history=None,
                            problem=problem,
                            attempt=attempt,
                            workflow=workflow,
                            error=f"{type(e).__name__}: {e}",
                        )

            runner = TraceRunner()
            exec_pairs = [(runner, ex) for _, ex in exec_pairs]

            parallel = dspy.Parallel(
                num_threads=min(args.parallel, len(exec_pairs)),
                max_errors=len(exec_pairs),  # don't stop on errors
                disable_progress_bar=args.display != "log",
                timeout=300,
            )
            try:
                results = parallel(exec_pairs=exec_pairs)
            except Exception as e:
                logger.error(f"  Parallel failed for {pid}: {e}")
                results = []

            seen_attempts: set[int] = set()
            for res in results:
                if res is None:
                    continue
                seen_attempts.add(int(getattr(res, "attempt", -1)))
                wf = getattr(res, "workflow", "")
                wf_tag = "LC" if wf == "LC" else ("HC" if wf == "HC" else "")
                display_pid = (
                    f"{res.problem.problem_id} [{wf_tag}]" if wf_tag else res.problem.problem_id
                )
                if getattr(res, "error", None):
                    display.on_error(display_pid, res.attempt + 1, res.error)
                    error_log.record(
                        model=args.model,
                        prompt_id=prompt_id,
                        problem_id=res.problem.problem_id,
                        attempt_number=res.attempt + 1,
                        error=str(res.error),
                    )
                    continue
                try:
                    trace = make_trace(
                        res.problem,
                        res.result,
                        args.model,
                        prompt_id,
                        res.attempt,
                        res.lm_history,
                    )
                    trace.reward = score_trace(trace, res.problem)
                    append_trace(trace, output_path)
                    done += 1
                    if any(s.tool_name == "wiki_read" for s in trace.steps):
                        wiki_used += 1
                    display.on_success(display_pid, res.attempt + 1, trace)
                except Exception as e:
                    display.on_error(display_pid, res.attempt + 1, e)
                    error_log.record(
                        model=args.model,
                        prompt_id=prompt_id,
                        problem_id=res.problem.problem_id,
                        attempt_number=res.attempt + 1,
                        error=f"{type(e).__name__}: {e}",
                        traceback_text=traceback.format_exc(),
                    )

            # dspy.Parallel can yield missing/None items for failed workers.
            # Emit explicit errors so every scheduled attempt is accounted for.
            for prob, attempt in group_items:
                if attempt in seen_attempts:
                    continue
                msg = "Missing result from parallel worker (None/omitted response)"
                wf = workflow_by_attempt.get((prob.problem_id, attempt), "")
                wf_tag = "LC" if wf == "LC" else ("HC" if wf == "HC" else "")
                display_pid = f"{prob.problem_id} [{wf_tag}]" if wf_tag else prob.problem_id
                display.on_error(display_pid, attempt + 1, msg)
                error_log.record(
                    model=args.model,
                    prompt_id=prompt_id,
                    problem_id=prob.problem_id,
                    attempt_number=attempt + 1,
                    error=msg,
                )
    else:
        # Sequential execution
        for problem, attempt in work:
            try:
                result, lm_history = run_problem(
                    problem,
                    args.model,
                    args.max_iters,
                    optimized_program,
                    seed_text,
                    use_thinking=args.thinking_react,
                    include_wiki=args.enable_wiki,
                )
                trace = make_trace(
                    problem,
                    result,
                    args.model,
                    prompt_id,
                    attempt,
                    lm_history,
                )
                trace.reward = score_trace(trace, problem)
                append_trace(trace, output_path)
                done += 1
                if any(s.tool_name == "wiki_read" for s in trace.steps):
                    wiki_used += 1
                display.on_success(problem.problem_id, attempt + 1, trace)
            except Exception as e:
                display.on_error(problem.problem_id, attempt + 1, e)
                error_log.record(
                    model=args.model,
                    prompt_id=prompt_id,
                    problem_id=problem.problem_id,
                    attempt_number=attempt + 1,
                    error=f"{type(e).__name__}: {e}",
                    traceback_text=traceback.format_exc(),
                )

    # Summary
    display.finish()
    logger.info(f"\nSaved {done} traces to {output_path}")
    if error_log_path:
        logger.info(
            f"Error log: {error_log_path} (rolling, {args.error_log_max_mb}MB x {args.error_log_backups})"
        )
    if args.enable_wiki and done:
        logger.info(f"Wiki usage: {wiki_used}/{done} attempts ({100.0 * wiki_used / done:.1f}%)")


if __name__ == "__main__":
    main()
