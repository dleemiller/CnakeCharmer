"""Test wildcard_matching equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.wildcard_matching import wildcard_matching as cy_func
from cnake_data.py.dynamic_programming.wildcard_matching import wildcard_matching as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_wildcard_matching_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
