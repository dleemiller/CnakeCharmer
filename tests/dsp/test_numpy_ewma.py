"""Test numpy_ewma equivalence."""

import pytest

from cnake_charmer.cy.dsp.numpy_ewma import numpy_ewma as cy_func
from cnake_charmer.py.dsp.numpy_ewma import numpy_ewma as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_numpy_ewma_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
