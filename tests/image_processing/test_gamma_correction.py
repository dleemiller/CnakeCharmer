"""Test gamma_correction equivalence."""

import pytest

from cnake_charmer.cy.image_processing.gamma_correction import gamma_correction as cy_func
from cnake_charmer.py.image_processing.gamma_correction import gamma_correction as py_func


@pytest.mark.parametrize("n", [10, 20, 50, 100])
def test_gamma_correction_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert isinstance(py_result, tuple)
    assert isinstance(cy_result, tuple)
    assert py_result[0] == cy_result[0], (
        f"Gamma 0.5 sum mismatch at n={n}: py={py_result[0]}, cy={cy_result[0]}"
    )
    assert py_result[1] == cy_result[1], (
        f"Gamma 1.5 sum mismatch at n={n}: py={py_result[1]}, cy={cy_result[1]}"
    )
    assert py_result[2] == cy_result[2], (
        f"Gamma 2.2 sum mismatch at n={n}: py={py_result[2]}, cy={cy_result[2]}"
    )
