"""Test cyclic_autocorr_peak equivalence."""

import pytest

from cnake_data.cy.numerical.cyclic_autocorr_peak import cyclic_autocorr_peak as cy_func
from cnake_data.py.numerical.cyclic_autocorr_peak import cyclic_autocorr_peak as py_func


@pytest.mark.parametrize("n", [16, 64, 256, 1024])
def test_cyclic_autocorr_peak_equivalence(n):
    assert py_func(n) == cy_func(n)
