"""Test stdlib_qsort equivalence."""

import pytest

from cnake_charmer.cy.sorting.stdlib_qsort import stdlib_qsort as cy_func
from cnake_charmer.py.sorting.stdlib_qsort import stdlib_qsort as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_stdlib_qsort_equivalence(n):
    assert py_func(n) == cy_func(n)
