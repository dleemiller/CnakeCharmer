"""Test median_filter equivalence."""

import pytest

from cnake_charmer.cy.image_processing.median_filter import median_filter as cy_func
from cnake_charmer.py.image_processing.median_filter import median_filter as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 150])
def test_median_filter_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
