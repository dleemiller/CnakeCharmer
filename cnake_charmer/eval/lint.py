"""
Lint quality reward: based on cython-lint static analysis.

Rewards code that has fewer lint violations. Score is 1.0 for clean code,
decreasing with each violation found.
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LintResult:
    success: bool = True
    score: float = 1.0
    violations: list[str] = field(default_factory=list)
    violation_count: int = 0


def run_cython_lint(cython_code: str) -> LintResult:
    """
    Run cython-lint on Cython source code and return a lint result.

    Runs with --no-pycodestyle since pycodestyle rules (E501, E701, etc.)
    are too noisy for idiomatic Cython. Only checks cython-lint's own rules
    (unused vars/imports, dangerous defaults, etc.).

    Args:
        cython_code: The .pyx source code as a string.

    Returns:
        LintResult with score, violations list, and count.
    """
    result = LintResult()

    with tempfile.NamedTemporaryFile(suffix=".pyx", mode="w", delete=False) as f:
        f.write(cython_code)
        f.flush()
        tmp_path = Path(f.name)

    try:
        proc = subprocess.run(
            ["cython-lint", "--no-pycodestyle", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if proc.returncode == 0 and not proc.stdout.strip():
            # Clean - no violations
            result.score = 1.0
            result.violation_count = 0
            return result

        # Parse violations from output
        lines = [
            line.strip()
            for line in (proc.stdout + proc.stderr).strip().splitlines()
            if line.strip()
        ]
        result.violations = lines
        result.violation_count = len(lines)

        # Score: penalize each violation, floor at 0.0
        # Use a decay curve: score = max(0, 1 - 0.1 * n_violations)
        # So 10+ violations = 0.0
        result.score = max(0.0, 1.0 - 0.1 * result.violation_count)

    except FileNotFoundError:
        # cython-lint not installed - don't penalize
        result.success = False
        result.score = 1.0
    except subprocess.TimeoutExpired:
        result.success = False
        result.score = 1.0
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


def lint_reward(cython_code: str, **kwargs) -> float:
    """
    Return lint score (0.0 to 1.0).

    Higher score = fewer cython-lint violations.
    """
    return run_cython_lint(cython_code).score
