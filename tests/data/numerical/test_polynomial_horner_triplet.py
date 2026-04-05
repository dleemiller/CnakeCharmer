"""Test polynomial_horner_triplet equivalence."""

import pytest

from cnake_data.cy.numerical.polynomial_horner_triplet import polynomial_horner_triplet as cy_func
from cnake_data.py.numerical.polynomial_horner_triplet import polynomial_horner_triplet as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_polynomial_horner_triplet_equivalence(n):
    assert py_func(n) == cy_func(n)
