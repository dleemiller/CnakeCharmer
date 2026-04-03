"""
Validate all GRPO problem files in data/grpo_problems/.

Each file must:
- Import without errors
- Define at least one public function
- Run with small inputs without crashing
- Be deterministic (same input → same output)
"""

import importlib.util
import inspect
from pathlib import Path

import pytest

PROBLEMS_DIR = Path("data/grpo_problems")

pytestmark = pytest.mark.skipif(
    not PROBLEMS_DIR.exists(),
    reason="data/grpo_problems/ not found",
)


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_public_funcs(mod):
    return [
        (name, getattr(mod, name))
        for name in dir(mod)
        if not name.startswith("_")
        and callable(getattr(mod, name))
        and not isinstance(getattr(mod, name), type)
        and getattr(getattr(mod, name), "__module__", None) == mod.__name__
    ]


def _call_with_small_args(fn):
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    n = len(params)
    if n == 0:
        return fn()
    if n == 1:
        return fn(10)
    if n == 2:
        return fn(5, 10)
    if n == 3:
        return fn(5, 10, 100)
    return fn(*([5] * n))


def _get_problem_files():
    if not PROBLEMS_DIR.exists():
        return []
    return sorted(PROBLEMS_DIR.glob("*.py"))


@pytest.fixture(params=_get_problem_files(), ids=lambda p: p.stem)
def problem_module(request):
    return _load_module(request.param)


def test_has_public_function(problem_module):
    funcs = _get_public_funcs(problem_module)
    assert len(funcs) > 0, f"No public functions in {problem_module.__name__}"


def test_runs_without_error(problem_module):
    for _name, fn in _get_public_funcs(problem_module):
        _call_with_small_args(fn)


def test_deterministic(problem_module):
    for name, fn in _get_public_funcs(problem_module):
        r1 = _call_with_small_args(fn)
        r2 = _call_with_small_args(fn)
        assert r1 == r2, f"{name} is non-deterministic: {r1} != {r2}"
