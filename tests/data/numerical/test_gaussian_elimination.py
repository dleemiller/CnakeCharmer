"""Test gaussian_elimination equivalence."""

import pytest

from cnake_data.cy.numerical.gaussian_elimination import gaussian_elimination as cy_func
from cnake_data.py.numerical.gaussian_elimination import gaussian_elimination as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_gaussian_elimination_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={p}, cy={c}"
