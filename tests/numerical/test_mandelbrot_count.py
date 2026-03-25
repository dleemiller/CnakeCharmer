"""Test mandelbrot count equivalence."""

import pytest

from cnake_charmer.cy.numerical.mandelbrot_count import mandelbrot_count as cy_mandelbrot_count
from cnake_charmer.py.numerical.mandelbrot_count import mandelbrot_count as py_mandelbrot_count


@pytest.mark.parametrize("n", [10, 25, 50, 100])
def test_mandelbrot_count_equivalence(n):
    assert py_mandelbrot_count(n) == cy_mandelbrot_count(n)
