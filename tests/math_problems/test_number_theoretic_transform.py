"""Test number_theoretic_transform equivalence."""

import pytest

from cnake_charmer.cy.math_problems.number_theoretic_transform import (
    number_theoretic_transform as cy_func,
)
from cnake_charmer.py.math_problems.number_theoretic_transform import (
    number_theoretic_transform as py_func,
)


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_number_theoretic_transform_equivalence(n):
    assert py_func(n) == cy_func(n)
