"""Test lu_decomposition equivalence."""

import pytest

from cnake_charmer.cy.numerical.lu_decomposition import lu_decomposition as cy_func
from cnake_charmer.py.numerical.lu_decomposition import lu_decomposition as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_lu_decomposition_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
