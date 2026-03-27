"""Test freelist_point_sum equivalence."""

import pytest

from cnake_charmer.cy.geometry.freelist_point_sum import freelist_point_sum as cy_func
from cnake_charmer.py.geometry.freelist_point_sum import freelist_point_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_freelist_point_sum_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-4
