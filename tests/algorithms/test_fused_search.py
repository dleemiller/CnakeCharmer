"""Test fused_search equivalence."""

import pytest

from cnake_charmer.cy.algorithms.fused_search import (
    fused_search as cy_func,
)
from cnake_charmer.py.algorithms.fused_search import (
    fused_search as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_fused_search_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
