"""Test snell_refraction equivalence."""

import pytest

from cnake_data.cy.physics.snell_refraction import snell_refraction as cy_func
from cnake_data.py.physics.snell_refraction import snell_refraction as py_func


@pytest.mark.parametrize(
    "n_rays,n_layers",
    [
        (1000, 8),
        (3000, 10),
        (2000, 6),
        (500, 4),
    ],
)
def test_snell_refraction_equivalence(n_rays, n_layers):
    assert py_func(n_rays, n_layers) == cy_func(n_rays, n_layers)
