"""Test tdma_solver_stats equivalence."""

import pytest

from cnake_data.cy.numerical.tdma_solver_stats import tdma_solver_stats as cy_func
from cnake_data.py.numerical.tdma_solver_stats import tdma_solver_stats as py_func


@pytest.mark.parametrize(
    "a0,b0,c0,d0,n,passes",
    [
        (0.1, 2.1, 0.12, 1.5, 60, 1),
        (0.11, 2.2, 0.13, 1.7, 120, 2),
        (0.09, 2.4, 0.1, 1.1, 180, 2),
    ],
)
def test_tdma_solver_stats_equivalence(a0, b0, c0, d0, n, passes):
    py_result = py_func(a0, b0, c0, d0, n, passes)
    cy_result = cy_func(a0, b0, c0, d0, n, passes)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-8
