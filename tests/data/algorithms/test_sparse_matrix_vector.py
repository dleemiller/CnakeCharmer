"""Test sparse_matrix_vector equivalence."""

import pytest

from cnake_data.cy.algorithms.sparse_matrix_vector import sparse_matrix_vector as cy_func
from cnake_data.py.algorithms.sparse_matrix_vector import sparse_matrix_vector as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_sparse_matrix_vector_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
