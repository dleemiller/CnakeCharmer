"""Test diagonal_scan equivalence."""

import pytest

from cnake_charmer.cy.string_processing.diagonal_scan import diagonal_scan as cy_func
from cnake_charmer.py.string_processing.diagonal_scan import diagonal_scan as py_func


@pytest.mark.parametrize("n", [200, 500, 1000, 2000])
def test_diagonal_scan_equivalence(n):
    assert py_func(n) == cy_func(n)
