"""Test nelder_mead equivalence."""

import pytest

from cnake_data.cy.optimization.nelder_mead import nelder_mead as cy_func
from cnake_data.py.optimization.nelder_mead import nelder_mead as py_func


@pytest.mark.parametrize("dim,n_starts", [(2, 5), (3, 10), (4, 8)])
def test_nelder_mead_equivalence(dim, n_starts):
    py_result = py_func(dim, n_starts)
    cy_result = cy_func(dim, n_starts)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
