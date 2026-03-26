"""Test hash_table_ops equivalence."""

import pytest

from cnake_charmer.cy.algorithms.hash_table_ops import hash_table_ops as cy_hash_table_ops
from cnake_charmer.py.algorithms.hash_table_ops import hash_table_ops as py_hash_table_ops


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_hash_table_ops_equivalence(n):
    py_result = py_hash_table_ops(n)
    cy_result = cy_hash_table_ops(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
