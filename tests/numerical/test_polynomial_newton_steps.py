"""Test polynomial_newton_steps equivalence."""

import pytest

from cnake_charmer.cy.numerical.polynomial_newton_steps import polynomial_newton_steps as cy_func
from cnake_charmer.py.numerical.polynomial_newton_steps import polynomial_newton_steps as py_func


@pytest.mark.parametrize(
    "start,iters,scale",
    [
        (1.2, 10, 0.8),
        (1.5, 25, 1.0),
        (2.0, 40, 1.3),
        (-1.0, 35, 0.6),
    ],
)
def test_polynomial_newton_steps_equivalence(start, iters, scale):
    assert py_func(start, iters, scale) == cy_func(start, iters, scale)
