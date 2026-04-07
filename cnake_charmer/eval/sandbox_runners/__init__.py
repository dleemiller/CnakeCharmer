"""Sandbox runner scripts — executed inside the bwrap sandbox.

These are real Python files (not embedded strings) so they get linted,
have IDE support, and can be tested independently. They are ro-bound
into the sandbox at runtime and invoked via ``run_runner_sandboxed()``.

See each runner's module docstring for the JSON config it expects and
the JSON output it produces.
"""

from pathlib import Path

RUNNERS_DIR = Path(__file__).parent
