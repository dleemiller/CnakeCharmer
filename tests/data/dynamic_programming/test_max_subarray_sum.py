"""Test max_subarray_sum equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.max_subarray_sum import max_subarray_sum as cy_func
from cnake_data.py.dynamic_programming.max_subarray_sum import max_subarray_sum as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_max_subarray_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
