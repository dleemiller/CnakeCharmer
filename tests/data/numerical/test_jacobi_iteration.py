"""Test jacobi_iteration equivalence."""

import pytest

from cnake_data.cy.numerical.jacobi_iteration import jacobi_iteration as cy_func
from cnake_data.py.numerical.jacobi_iteration import jacobi_iteration as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_jacobi_iteration_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
