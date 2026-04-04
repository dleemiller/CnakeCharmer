"""Test fft_naive equivalence."""

import pytest

from cnake_data.cy.numerical.fft_naive import fft_naive as cy_fft_naive
from cnake_data.py.numerical.fft_naive import fft_naive as py_fft_naive


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_fft_naive_equivalence(n):
    py_result = py_fft_naive(n)
    cy_result = cy_fft_naive(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
