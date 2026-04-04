"""Test gp_kernel equivalence."""

import pytest

from cnake_data.cy.statistics.gp_kernel import gp_kernel as cy_func
from cnake_data.py.statistics.gp_kernel import gp_kernel as py_func


@pytest.mark.parametrize("n", [10, 50, 100])
def test_gp_kernel_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
