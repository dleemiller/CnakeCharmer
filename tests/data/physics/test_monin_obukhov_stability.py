"""Test monin_obukhov_stability equivalence."""

import pytest

from cnake_data.cy.physics.monin_obukhov_stability import monin_obukhov_stability as cy_func
from cnake_data.py.physics.monin_obukhov_stability import monin_obukhov_stability as py_func


@pytest.mark.parametrize("n", [50, 100, 200, 500])
def test_monin_obukhov_stability_equivalence(n):
    py_mean, py_first, py_last = py_func(n)
    cy_mean, cy_first, cy_last = cy_func(n)
    assert abs(py_mean - cy_mean) / max(abs(py_mean), 1.0) < 1e-12
    assert py_first == cy_first
    assert abs(py_last - cy_last) / max(abs(py_last), 1.0) < 1e-12
