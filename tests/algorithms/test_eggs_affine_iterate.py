"""Test eggs_affine_iterate equivalence."""

import pytest

from cnake_charmer.cy.algorithms.eggs_affine_iterate import eggs_affine_iterate as cy_func
from cnake_charmer.py.algorithms.eggs_affine_iterate import eggs_affine_iterate as py_func


@pytest.mark.parametrize(
    "a,b,steps,x0,y0",
    [(0.8, 0.2, 100, 0.2, -0.1), (0.83, 0.17, 400, -0.5, 0.7), (0.75, 0.25, 600, 0.1, 0.1)],
)
def test_eggs_affine_iterate_equivalence(a, b, steps, x0, y0):
    py_result = py_func(a, b, steps, x0, y0)
    cy_result = cy_func(a, b, steps, x0, y0)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-9
