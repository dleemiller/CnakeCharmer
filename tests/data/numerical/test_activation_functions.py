"""Test activation_functions equivalence."""

import pytest

from cnake_data.cy.numerical.activation_functions import (
    activation_functions as cy_func,
)
from cnake_data.py.numerical.activation_functions import (
    activation_functions as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_activation_functions_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
