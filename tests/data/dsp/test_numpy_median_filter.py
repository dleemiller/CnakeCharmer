"""Test numpy_median_filter equivalence."""

import pytest

from cnake_data.cy.dsp.numpy_median_filter import (
    numpy_median_filter as cy_func,
)
from cnake_data.py.dsp.numpy_median_filter import (
    numpy_median_filter as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 5000])
def test_numpy_median_filter_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
