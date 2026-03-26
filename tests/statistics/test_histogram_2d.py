"""Test histogram_2d equivalence."""

import pytest

from cnake_charmer.cy.statistics.histogram_2d import histogram_2d as cy_func
from cnake_charmer.py.statistics.histogram_2d import histogram_2d as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_histogram_2d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
