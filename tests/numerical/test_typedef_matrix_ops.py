"""Test typedef_matrix_ops equivalence."""

import pytest

from cnake_charmer.cy.numerical.typedef_matrix_ops import (
    typedef_matrix_ops as cy_typedef_matrix_ops,
)
from cnake_charmer.py.numerical.typedef_matrix_ops import (
    typedef_matrix_ops as py_typedef_matrix_ops,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_typedef_matrix_ops_equivalence(n):
    py_result = py_typedef_matrix_ops(n)
    cy_result = cy_typedef_matrix_ops(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
