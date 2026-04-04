"""Test kernel_density equivalence."""

import pytest

from cnake_data.cy.statistics.kernel_density import kernel_density as cy_func
from cnake_data.py.statistics.kernel_density import kernel_density as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_kernel_density_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
