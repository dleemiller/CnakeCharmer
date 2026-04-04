"""Test babylonian_sqrt_trace equivalence."""

import pytest

from cnake_charmer.cy.numerical.babylonian_sqrt_trace import babylonian_sqrt_trace as cy_func
from cnake_charmer.py.numerical.babylonian_sqrt_trace import babylonian_sqrt_trace as py_func


@pytest.mark.parametrize(
    "value,loops,tol",
    [
        (0.25, 50, 1e-10),
        (2.0, 60, 1e-12),
        (9999.0, 80, 1e-12),
        (123456.789, 120, 1e-14),
    ],
)
def test_babylonian_sqrt_trace_equivalence(value, loops, tol):
    assert py_func(value, loops, tol) == cy_func(value, loops, tol)
