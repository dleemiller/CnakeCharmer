"""Test gauss_legendre_pi equivalence."""

import pytest

from cnake_data.cy.numerical.gauss_legendre_pi import gauss_legendre_pi as cy_func
from cnake_data.py.numerical.gauss_legendre_pi import gauss_legendre_pi as py_func


@pytest.mark.parametrize("n", [1, 5, 10, 25, 50])
def test_gauss_legendre_pi_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-10, f"Mismatch at n={n}: {py_result} vs {cy_result}"
