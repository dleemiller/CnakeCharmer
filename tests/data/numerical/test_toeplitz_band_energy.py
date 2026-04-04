"""Test toeplitz_band_energy equivalence."""

import pytest

from cnake_data.cy.numerical.toeplitz_band_energy import toeplitz_band_energy as cy_func
from cnake_data.py.numerical.toeplitz_band_energy import toeplitz_band_energy as py_func


@pytest.mark.parametrize("n", [8, 64, 512, 4096])
def test_toeplitz_band_energy_equivalence(n):
    assert py_func(n) == cy_func(n)
