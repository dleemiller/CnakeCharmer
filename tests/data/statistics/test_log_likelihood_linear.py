"""Test log_likelihood_linear equivalence."""

import pytest

from cnake_data.cy.statistics.log_likelihood_linear import log_likelihood_linear as cy_func
from cnake_data.py.statistics.log_likelihood_linear import log_likelihood_linear as py_func


@pytest.mark.parametrize("n,a,b,c", [(100, 1.0, 0.5, 0.1), (1000, 2.5, 0.3, -0.1)])
def test_log_likelihood_linear_equivalence(n, a, b, c):
    py_result = py_func(n, a, b, c)
    cy_result = cy_func(n, a, b, c)
    for p, c_ in zip(py_result, cy_result, strict=False):
        assert abs(p - c_) / max(abs(p), 1.0) < 1e-6
