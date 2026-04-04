"""Test mann_whitney_u equivalence."""

import pytest

from cnake_data.cy.statistics.mann_whitney_u import mann_whitney_u as cy_func
from cnake_data.py.statistics.mann_whitney_u import mann_whitney_u as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_mann_whitney_u_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for a, b in zip(py_result, cy_result, strict=False):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-4
