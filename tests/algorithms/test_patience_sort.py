"""Test patience_sort equivalence."""

import pytest

from cnake_charmer.cy.algorithms.patience_sort import patience_sort as cy_func
from cnake_charmer.py.algorithms.patience_sort import patience_sort as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_patience_sort_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
