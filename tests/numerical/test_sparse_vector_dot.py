"""Test sparse_vector_dot equivalence."""

import pytest

from cnake_charmer.cy.numerical.sparse_vector_dot import sparse_vector_dot as cy_func
from cnake_charmer.py.numerical.sparse_vector_dot import sparse_vector_dot as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_sparse_vector_dot_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
