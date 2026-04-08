"""
LLM-powered wiki reflection agent.

Uses a DSPy ReAct agent to analyze traces, read docs and reference code,
and update wiki pages with real findings (code diffs, patterns, fixes).

This replaces the regex-only approach with an agent that can:
- Read current wiki page content
- Read reference Cython implementations from cy/
- Read official Cython documentation
- Verify code snippets compile via evaluate_cython
- Write updated wiki pages with flock safety
"""

import json
import logging
from pathlib import Path

import dspy

from cnake_charmer.wiki.merge import atomic_wiki_write
from cnake_charmer.wiki.reflect import reflect_on_traces
from cnake_charmer.wiki.search import wiki_read as _wiki_read
from cnake_charmer.wiki.trace_analysis import (
    format_trace_for_llm,
    load_cython_docs,
    load_reference_code,
    select_example_traces,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WIKI_DIR = _PROJECT_ROOT / "wiki"
_WIKI_PAGES = _WIKI_DIR / "pages"


# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------


class WikiCurator(dspy.Signature):
    """Analyze Cython optimization traces and update a wiki page with new patterns and fixes.

    You have access to trace data showing how models attempted Cython optimizations,
    including what errors they hit and how they fixed them. Your job is to improve
    the wiki page with concrete, verified patterns from these traces.

    Read the current page first, then decide what to add or update based on the
    trace evidence. Verify any code examples compile before adding them.
    """

    page_name: str = dspy.InputField(desc="Wiki page to update (e.g. 'pitfalls', 'memoryviews')")
    regex_findings: str = dspy.InputField(
        desc="Statistical summary from regex analysis: error counts, fix rates, top errors"
    )
    example_traces: str = dspy.InputField(
        desc="Selected traces with code diffs showing error->fix patterns"
    )
    updated: bool = dspy.OutputField(desc="Whether the page was updated with new content")


# ---------------------------------------------------------------------------
# Tool functions for the reflection agent
# ---------------------------------------------------------------------------


def _make_reflection_tools(dry_run: bool = False):
    """Create tool functions for the wiki reflection agent."""

    def wiki_read(page: str) -> str:
        """Read a wiki page in full. Returns the complete markdown content.

        Args:
            page: Page name (e.g. 'memoryviews', 'pitfalls').
        """
        return _wiki_read(page)

    def wiki_update(page: str, content: str) -> str:
        """Write updated content to a wiki page. Uses atomic file locking.

        Args:
            page: Page name (e.g. 'pitfalls').
            content: Full markdown content for the page.
        """
        path = _WIKI_PAGES / f"{page.removesuffix('.md')}.md"
        if dry_run:
            preview = content[:500] + "..." if len(content) > 500 else content
            return json.dumps(
                {
                    "dry_run": True,
                    "page": page,
                    "content_length": len(content),
                    "preview": preview,
                }
            )

        atomic_wiki_write(path, content)

        # Append to log
        from datetime import datetime

        log_path = _WIKI_DIR / "log.md"
        if log_path.exists():
            ts = datetime.now().strftime("%Y-%m-%d")
            entry = f"| {ts} | update | {page} | llm_reflect |\n"
            with open(log_path, "a") as f:
                f.write(entry)

        return json.dumps({"success": True, "page": page, "content_length": len(content)})

    def read_reference(category: str) -> str:
        """Load reference Cython implementations from cy/ for a category.

        Args:
            category: Problem category (e.g. 'numerical', 'algorithms', 'graph').
                      Or a wiki page name — will auto-map to relevant categories.
        """
        return load_reference_code(category)

    def read_cython_docs(topic: str) -> str:
        """Search official Cython documentation for a topic.

        Requires `make wiki-setup` to have been run to clone docs.

        Args:
            topic: Search terms (e.g. 'memoryview', 'nogil prange', 'cdef class').
        """
        return load_cython_docs(topic)

    def evaluate_cython(code: str, python_code: str, test_code: str) -> str:
        """Compile and test a Cython code snippet to verify it works.

        Use this to verify code examples before adding them to wiki pages.

        Args:
            code: Complete .pyx Cython source code.
            python_code: Original Python source code (reference).
            test_code: Equivalence test assertions.
        """
        from cnake_charmer.training.environment import CythonToolEnvironment

        env = CythonToolEnvironment()
        env.reset()
        return env.evaluate_cython(code=code, python_code=python_code, test_code=test_code)

    return [wiki_read, wiki_update, read_reference, read_cython_docs, evaluate_cython]


# ---------------------------------------------------------------------------
# Reflection agent
# ---------------------------------------------------------------------------


class WikiReflectionAgent(dspy.Module):
    """Agentic wiki curator that reads traces, docs, and reference code to improve wiki pages."""

    def __init__(self, max_iters: int = 8, dry_run: bool = False):
        super().__init__()
        tools = _make_reflection_tools(dry_run=dry_run)
        self.react = dspy.ReAct(WikiCurator, tools=tools, max_iters=max_iters)

    def forward(self, page_name: str, regex_findings: str, example_traces: str):
        return self.react(
            page_name=page_name,
            regex_findings=regex_findings,
            example_traces=example_traces,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_reflection(
    traces_path: Path,
    model: str | None = None,
    category: str | None = None,
    since: str | None = None,
    max_traces: int = 2000,
    pages: list[str] | None = None,
    dry_run: bool = False,
    max_iters: int = 8,
) -> dict:
    """Run LLM-powered reflection on traces to update wiki pages.

    1. Load traces, run regex reflect_on_traces() for statistics
    2. For each wiki page with findings (or specified pages):
       a. Select example traces with code diffs
       b. Format regex findings + trace examples
       c. Run WikiReflectionAgent
    3. Return {page: {updated, reasoning}}

    Args:
        traces_path: Path to master_traces.jsonl.
        model: LM model ID (if None, uses current dspy.settings.lm).
        category: Filter traces by category.
        since: ISO date filter for traces.
        max_traces: Max traces to analyze.
        pages: Specific pages to update (default: all with findings).
        dry_run: If True, show proposed changes without writing.
        max_iters: Max ReAct iterations per page.

    Returns:
        Dict mapping page names to {updated, error} results.
    """
    from cnake_charmer.traces.io import load_traces

    # Step 1: Run regex analysis for statistics
    logger.info("Running regex analysis...")
    regex_results = reflect_on_traces(
        traces_path=traces_path,
        category=category,
        min_reward=0.0,
        max_traces=max_traces,
        since=since,
    )

    findings = regex_results.get("findings", {})
    if not findings:
        logger.info("No findings from regex analysis.")
        return {}

    # Determine which pages to process
    target_pages = pages if pages else list(findings.keys())
    logger.info(f"Processing {len(target_pages)} pages: {target_pages}")

    # Step 2: Load traces for example selection
    traces = load_traces([traces_path])
    if since:
        from datetime import datetime

        since_dt = datetime.fromisoformat(since)
        traces = [t for t in traces if t.timestamp and t.timestamp >= since_dt]
    if category:
        traces = [t for t in traces if t.category == category]
    traces = traces[:max_traces]

    # Step 3: Create agent
    agent = WikiReflectionAgent(max_iters=max_iters, dry_run=dry_run)

    results = {}
    for page in target_pages:
        logger.info(f"Reflecting on page: {page}")

        # Select and format example traces
        examples = select_example_traces(traces, page, max_examples=5)
        if not examples:
            logger.info(f"  No example traces for {page}, skipping")
            results[page] = {"updated": False, "reason": "no_examples"}
            continue

        formatted_traces = "\n\n---\n\n".join(format_trace_for_llm(t) for t in examples)

        # Format regex findings for this page
        page_findings = findings.get(page, {})
        regex_summary = json.dumps(page_findings, indent=2)

        try:
            result = agent(
                page_name=page,
                regex_findings=regex_summary,
                example_traces=formatted_traces,
            )
            results[page] = {"updated": result.updated}
            logger.info(f"  {page}: updated={result.updated}")
        except Exception as e:
            logger.error(f"  {page}: error - {e}")
            results[page] = {"updated": False, "error": str(e)}

    return results
