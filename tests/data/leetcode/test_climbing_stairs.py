"""Test climbing_stairs equivalence."""

import pytest

from cnake_data.cy.leetcode.climbing_stairs import climbing_stairs as cy_func
from cnake_data.py.leetcode.climbing_stairs import climbing_stairs as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_climbing_stairs_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
