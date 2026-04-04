"""Test introsort equivalence."""

import pytest

from cnake_data.cy.sorting.introsort import introsort as cy_func
from cnake_data.py.sorting.introsort import introsort as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_introsort_equivalence(n):
    assert py_func(n) == cy_func(n)
