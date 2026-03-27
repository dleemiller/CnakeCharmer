"""Test kolmogorov_smirnov equivalence."""

import pytest

from cnake_charmer.cy.statistics.kolmogorov_smirnov import kolmogorov_smirnov as cy_func
from cnake_charmer.py.statistics.kolmogorov_smirnov import kolmogorov_smirnov as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_kolmogorov_smirnov_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for a, b in zip(py_result, cy_result, strict=False):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-4
