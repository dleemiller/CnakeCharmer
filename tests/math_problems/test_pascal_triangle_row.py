"""Test Pascal's triangle row equivalence."""

import pytest

from cnake_charmer.cy.math_problems.pascal_triangle_row import (
    pascal_triangle_row as cy_pascal_triangle_row,
)
from cnake_charmer.py.math_problems.pascal_triangle_row import (
    pascal_triangle_row as py_pascal_triangle_row,
)


@pytest.mark.parametrize("n", [5, 20, 100, 500])
def test_pascal_triangle_row_equivalence(n):
    assert py_pascal_triangle_row(n) == cy_pascal_triangle_row(n)
