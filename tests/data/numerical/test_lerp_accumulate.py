"""Test lerp_accumulate equivalence."""

import pytest

from cnake_data.cy.numerical.lerp_accumulate import lerp_accumulate as cy_func
from cnake_data.py.numerical.lerp_accumulate import lerp_accumulate as py_func


@pytest.mark.parametrize(
    "a0,b0,steps,delta",
    [(1.0, 2.0, 100, 1e-3), (0.25, 3.5, 5000, 1e-5), (10.0, -2.0, 20000, 1e-6)],
)
def test_lerp_accumulate_equivalence(a0, b0, steps, delta):
    py_result = py_func(a0, b0, steps, delta)
    cy_result = cy_func(a0, b0, steps, delta)
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-10
