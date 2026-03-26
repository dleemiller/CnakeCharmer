"""Test topological_sort equivalence."""

import pytest

from cnake_charmer.cy.algorithms.topological_sort import topological_sort as cy_topological_sort
from cnake_charmer.py.algorithms.topological_sort import topological_sort as py_topological_sort


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_topological_sort_equivalence(n):
    py_result = py_topological_sort(n)
    cy_result = cy_topological_sort(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
