"""Test shell_sort equivalence."""

import pytest

from cnake_charmer.cy.sorting.shell_sort import shell_sort as cy_func
from cnake_charmer.py.sorting.shell_sort import shell_sort as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_shell_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
