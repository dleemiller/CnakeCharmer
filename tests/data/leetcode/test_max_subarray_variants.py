"""Test max_subarray_variants equivalence."""

import pytest

from cnake_data.cy.leetcode.max_subarray_variants import max_subarray_variants as cy_func
from cnake_data.py.leetcode.max_subarray_variants import max_subarray_variants as py_func


@pytest.mark.parametrize("n", [50, 100, 500, 1000])
def test_max_subarray_variants_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
