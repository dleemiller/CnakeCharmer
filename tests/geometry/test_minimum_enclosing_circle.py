"""Test minimum_enclosing_circle equivalence."""

import pytest

from cnake_charmer.cy.geometry.minimum_enclosing_circle import minimum_enclosing_circle as cy_func
from cnake_charmer.py.geometry.minimum_enclosing_circle import minimum_enclosing_circle as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_minimum_enclosing_circle_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for a, b in zip(py_result, cy_result, strict=False):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-4
