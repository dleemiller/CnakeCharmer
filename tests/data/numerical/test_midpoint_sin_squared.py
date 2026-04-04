"""Test midpoint_sin_squared equivalence."""

import pytest

from cnake_data.cy.numerical.midpoint_sin_squared import midpoint_sin_squared as cy_func
from cnake_data.py.numerical.midpoint_sin_squared import midpoint_sin_squared as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_midpoint_sin_squared_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for i in range(2):
        rel = abs(py_result[i] - cy_result[i]) / max(abs(py_result[i]), 1.0)
        assert rel < 1e-4, f"Mismatch at element {i}: {py_result[i]} vs {cy_result[i]}"
