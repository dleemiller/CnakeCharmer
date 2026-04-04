"""Test running mean equivalence."""

import pytest

from cnake_data.cy.numerical.running_mean import running_mean as cy_running_mean
from cnake_data.py.numerical.running_mean import running_mean as py_running_mean


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_running_mean_equivalence(n):
    py_result = py_running_mean(n)
    cy_result = cy_running_mean(n)
    assert len(py_result) == len(cy_result)
    for py_val, cy_val in zip(py_result, cy_result, strict=False):
        assert abs(py_val - cy_val) < 1e-9, f"Mismatch: py={py_val}, cy={cy_val}"
