"""Test double_list_accumulator equivalence."""

import pytest

from cnake_charmer.cy.algorithms.double_list_accumulator import (
    double_list_accumulator as cy_func,
)
from cnake_charmer.py.algorithms.double_list_accumulator import (
    double_list_accumulator as py_func,
)


@pytest.mark.parametrize(
    "n,scale,rounds,alpha,beta",
    [(40, 0.2, 20, 0.6, 0.4), (80, 0.17, 50, 0.61, 0.39), (100, 0.12, 60, 0.7, 0.2)],
)
def test_double_list_accumulator_equivalence(n, scale, rounds, alpha, beta):
    py_result = py_func(n, scale, rounds, alpha, beta)
    cy_result = cy_func(n, scale, rounds, alpha, beta)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-9
