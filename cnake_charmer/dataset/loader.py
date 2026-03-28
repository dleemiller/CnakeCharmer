"""
Dataset loader that discovers Python/Cython pairs from the repo structure.

The repo IS the dataset:
- py/{category}/{name}.py — Python implementations (training prompts)
- cy/{category}/{name}.pyx — Cython implementations (ground truth)
- tests/{category}/test_{name}.py — Equivalence tests

This module walks the py/ directory to discover all problems and builds
ProblemSpecs suitable for training.
"""

import ast
import logging
import re
from pathlib import Path

from cnake_charmer.dataset.difficulty import classify_difficulty
from cnake_charmer.sources.base import ProblemSpec

logger = logging.getLogger(__name__)

# Root paths
PACKAGE_ROOT = Path(__file__).parent.parent
PY_DIR = PACKAGE_ROOT / "py"
CY_DIR = PACKAGE_ROOT / "cy"
TESTS_DIR = PACKAGE_ROOT.parent / "tests"


def discover_pairs() -> list[ProblemSpec]:
    """Walk py/ to discover all Python/Cython problem pairs.

    Returns:
        List of ProblemSpecs with python_code, cython_code (if exists),
        func_name, category, and test_cases extracted from the repo.
    """
    problems = []

    for py_path in sorted(PY_DIR.rglob("*.py")):
        if py_path.name.startswith("__"):
            continue

        # Derive paths
        rel = py_path.relative_to(PY_DIR)
        category = str(rel.parent) if rel.parent != Path(".") else "general"
        stem = py_path.stem
        cy_path = CY_DIR / rel.with_suffix(".pyx")
        test_path = TESTS_DIR / rel.parent / f"test_{stem}.py"

        # Read Python source
        python_code = py_path.read_text()

        # Read Cython source if it exists
        cython_code = ""
        if cy_path.exists():
            cython_code = cy_path.read_text()

        # Extract the main function name and benchmark args
        func_name, benchmark_args = _extract_func_info(python_code)
        if not func_name:
            logger.warning(f"Could not find function in {py_path}, skipping")
            continue

        # Strip the benchmark decorator to get clean Python for the prompt
        clean_python = _strip_benchmark_decorator(python_code)

        # Extract test cases from the test file
        test_cases = _extract_test_cases(test_path, func_name)

        # Also strip decorator from cython for clean ground truth
        clean_cython = _strip_benchmark_decorator(cython_code) if cython_code else ""

        problems.append(
            ProblemSpec(
                problem_id=f"{category}/{stem}",
                description=_extract_docstring(python_code),
                python_code=clean_python,
                cython_code=clean_cython,
                func_name=func_name,
                test_cases=test_cases,
                benchmark_args=benchmark_args,
                category=category,
                difficulty=classify_difficulty(clean_cython, category),
                source="repo",
                metadata={
                    "py_path": str(py_path),
                    "cy_path": str(cy_path) if cy_path.exists() else "",
                    "test_path": str(test_path) if test_path.exists() else "",
                },
            )
        )

    logger.info(f"Discovered {len(problems)} problem pairs")
    return problems


def _extract_func_info(source: str) -> tuple[str, tuple | None]:
    """Extract the main function name and benchmark args from Python source."""
    # Look for @python_benchmark(args=(...)) or @cython_benchmark(...)
    args_match = re.search(r"@(?:python|cython)_benchmark\(.*?args=\(([^)]*)\)", source, re.DOTALL)
    benchmark_args = None
    if args_match:
        try:
            args_str = args_match.group(1).strip().rstrip(",")
            benchmark_args = ast.literal_eval(f"({args_str},)")
        except (ValueError, SyntaxError):
            pass

    # Find the first public def
    for match in re.finditer(r"^def (\w+)\(", source, re.MULTILINE):
        name = match.group(1)
        if not name.startswith("_"):
            return name, benchmark_args

    return "", None


def _strip_benchmark_decorator(source: str) -> str:
    """Remove benchmark decorator and its import, leaving clean code."""
    lines = source.split("\n")
    cleaned = []
    for line in lines:
        # Skip benchmark imports
        if re.match(r"\s*from cnake_charmer\.benchmarks import", line):
            continue
        # Skip benchmark decorators
        if re.match(r"\s*@(python|cython)_benchmark", line):
            continue
        # Skip `import cython` (only needed in .pyx context)
        if re.match(r"\s*import cython\s*$", line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def _extract_docstring(source: str) -> str:
    """Extract the module-level docstring."""
    try:
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        return docstring or ""
    except SyntaxError:
        # .pyx files won't parse as Python
        match = re.search(r'"""(.*?)"""', source, re.DOTALL)
        return match.group(1).strip() if match else ""


def _extract_test_cases(test_path: Path, func_name: str) -> list:
    """Extract parametrized test values from a pytest test file.

    Looks for @pytest.mark.parametrize decorators to find test inputs.
    Returns list of ((arg,),) tuples suitable for correctness checking.
    """
    if not test_path.exists():
        return []

    source = test_path.read_text()
    test_cases = []

    # Match @pytest.mark.parametrize("param", [val1, val2, ...])
    for match in re.finditer(
        r'@pytest\.mark\.parametrize\(\s*["\'](\w+)["\']\s*,\s*\[([^\]]+)\]',
        source,
    ):
        values_str = match.group(2)
        try:
            values = ast.literal_eval(f"[{values_str}]")
            for val in values:
                if isinstance(val, (list, tuple)):
                    test_cases.append((tuple(val),))
                else:
                    test_cases.append(((val,),))
        except (ValueError, SyntaxError):
            pass

    return test_cases
