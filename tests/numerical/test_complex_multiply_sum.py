"""Test complex_multiply_sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.complex_multiply_sum import complex_multiply_sum as cy_func
from cnake_charmer.py.numerical.complex_multiply_sum import complex_multiply_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_complex_multiply_sum_equivalence(n):
    py_re, py_im = py_func(n)
    cy_re, cy_im = cy_func(n)
    tol_re = max(1e-6, abs(py_re) * 1e-9)
    tol_im = max(1e-6, abs(py_im) * 1e-9)
    assert abs(py_re - cy_re) < tol_re, f"Real mismatch: py={py_re}, cy={cy_re}"
    assert abs(py_im - cy_im) < tol_im, f"Imag mismatch: py={py_im}, cy={cy_im}"
