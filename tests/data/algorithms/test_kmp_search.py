"""Test kmp_search equivalence."""

import pytest

from cnake_data.cy.algorithms.kmp_search import kmp_search as cy_kmp_search
from cnake_data.py.algorithms.kmp_search import kmp_search as py_kmp_search


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_kmp_search_equivalence(n):
    py_result = py_kmp_search(n)
    cy_result = cy_kmp_search(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
