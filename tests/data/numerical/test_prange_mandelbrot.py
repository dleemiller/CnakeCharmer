"""Test prange_mandelbrot equivalence."""

import pytest

from cnake_data.cy.numerical.prange_mandelbrot import (
    prange_mandelbrot as cy_func,
)
from cnake_data.py.numerical.prange_mandelbrot import (
    prange_mandelbrot as py_func,
)


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_prange_mandelbrot_equivalence(n):
    assert py_func(n) == cy_func(n)
