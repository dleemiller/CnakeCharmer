"""Test mandelbrot_escape_stats equivalence."""

import pytest

from cnake_charmer.cy.numerical.mandelbrot_escape_stats import mandelbrot_escape_stats as cy_func
from cnake_charmer.py.numerical.mandelbrot_escape_stats import mandelbrot_escape_stats as py_func


@pytest.mark.parametrize(
    "w,h,iters",
    [
        (20, 12, 20),
        (64, 40, 40),
        (90, 60, 60),
        (120, 80, 75),
    ],
)
def test_mandelbrot_escape_stats_equivalence(w, h, iters):
    assert py_func(w, h, iters) == cy_func(w, h, iters)
