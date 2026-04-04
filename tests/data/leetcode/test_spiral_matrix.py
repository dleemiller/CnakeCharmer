"""Test spiral_matrix equivalence."""

import pytest

from cnake_data.cy.leetcode.spiral_matrix import spiral_matrix as cy_func
from cnake_data.py.leetcode.spiral_matrix import spiral_matrix as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_spiral_matrix_equivalence(n):
    assert py_func(n) == cy_func(n)
