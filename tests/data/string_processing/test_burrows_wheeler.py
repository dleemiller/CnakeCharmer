"""Test burrows_wheeler equivalence."""

import pytest

from cnake_data.cy.string_processing.burrows_wheeler import burrows_wheeler as cy_func
from cnake_data.py.string_processing.burrows_wheeler import burrows_wheeler as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_burrows_wheeler_equivalence(n):
    assert py_func(n) == cy_func(n)
