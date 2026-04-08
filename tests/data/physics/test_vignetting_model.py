"""Test vignetting_model equivalence."""

import pytest

from cnake_data.cy.physics.vignetting_model import vignetting_model as cy_func
from cnake_data.py.physics.vignetting_model import vignetting_model as py_func


@pytest.mark.parametrize(
    "rows,cols",
    [
        (100, 100),
        (200, 200),
        (150, 200),
        (80, 80),
    ],
)
def test_vignetting_model_equivalence(rows, cols):
    assert py_func(rows, cols) == cy_func(rows, cols)
