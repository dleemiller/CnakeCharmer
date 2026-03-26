"""Test cholesky_decompose equivalence."""

import pytest

from cnake_charmer.cy.numerical.cholesky_decompose import cholesky_decompose as cy_func
from cnake_charmer.py.numerical.cholesky_decompose import cholesky_decompose as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_cholesky_decompose_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
