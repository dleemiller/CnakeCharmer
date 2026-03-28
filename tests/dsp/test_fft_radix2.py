"""Test fft_radix2 equivalence."""

import pytest

from cnake_charmer.cy.dsp.fft_radix2 import fft_radix2 as cy_func
from cnake_charmer.py.dsp.fft_radix2 import fft_radix2 as py_func


@pytest.mark.parametrize("n", [64, 256, 1024, 4096])
def test_fft_radix2_equivalence(n):
    py = py_func(n)
    cy = cy_func(n)
    for a, b in zip(py, cy):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-6
