"""Test mandelbrot_tile_stats equivalence."""

import pytest

from cnake_data.cy.numerical.mandelbrot_tile_stats import mandelbrot_tile_stats as cy_func
from cnake_data.py.numerical.mandelbrot_tile_stats import mandelbrot_tile_stats as py_func


@pytest.mark.parametrize(
    "min_val,max_val,size,max_iter,threshold",
    [
        (-2.0, 1.0, 40, 40, 4.0),
        (-1.8, 0.8, 60, 60, 4.0),
        (-2.2, 1.1, 48, 55, 4.0),
    ],
)
def test_mandelbrot_tile_stats_equivalence(min_val, max_val, size, max_iter, threshold):
    py_result = py_func(min_val, max_val, size, max_iter, threshold)
    cy_result = cy_func(min_val, max_val, size, max_iter, threshold)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-9
