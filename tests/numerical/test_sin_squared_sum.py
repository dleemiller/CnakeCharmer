"""Test sin_squared_sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.sin_squared_sum import sin_squared_sum as cy_func
from cnake_charmer.py.numerical.sin_squared_sum import sin_squared_sum as py_func


@pytest.mark.parametrize(
    "offset,step,samples",
    [(0.0, 0.01, 100), (0.25, 0.001, 5000), (1.0, 0.0005, 20000)],
)
def test_sin_squared_sum_equivalence(offset, step, samples):
    py_result = py_func(offset, step, samples)
    cy_result = cy_func(offset, step, samples)
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-10
