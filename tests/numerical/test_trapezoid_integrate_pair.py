"""Test trapezoid_integrate_pair equivalence."""

import pytest

from cnake_charmer.cy.numerical.trapezoid_integrate_pair import trapezoid_integrate_pair as cy_func
from cnake_charmer.py.numerical.trapezoid_integrate_pair import trapezoid_integrate_pair as py_func


@pytest.mark.parametrize(
    "a,b,steps,freq",
    [
        (0.0, 1.0, 64, 0.1),
        (0.0, 5.0, 500, 0.7),
        (-2.0, 3.0, 1024, 1.2),
        (1.0, 9.0, 2048, -0.3),
    ],
)
def test_trapezoid_integrate_pair_equivalence(a, b, steps, freq):
    assert py_func(a, b, steps, freq) == cy_func(a, b, steps, freq)
