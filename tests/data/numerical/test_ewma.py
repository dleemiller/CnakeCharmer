"""Test EWMA equivalence."""

import pytest

from cnake_data.cy.numerical.ewma import ewma as cy_ewma
from cnake_data.py.numerical.ewma import ewma as py_ewma


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_ewma_equivalence(n):
    py_result = py_ewma(n)
    cy_result = cy_ewma(n)
    assert len(py_result) == len(cy_result)
    for py_val, cy_val in zip(py_result, cy_result, strict=False):
        assert abs(py_val - cy_val) < 1e-9, f"Mismatch: py={py_val}, cy={cy_val}"
