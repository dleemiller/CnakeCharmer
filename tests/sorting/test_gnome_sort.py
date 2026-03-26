"""Test gnome_sort equivalence."""

import pytest

from cnake_charmer.cy.sorting.gnome_sort import gnome_sort as cy_func
from cnake_charmer.py.sorting.gnome_sort import gnome_sort as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_gnome_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
