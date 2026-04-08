"""Test logistic_bifurcation equivalence."""

import pytest

from cnake_data.cy.simulation.logistic_bifurcation import logistic_bifurcation as cy_func
from cnake_data.py.simulation.logistic_bifurcation import logistic_bifurcation as py_func


@pytest.mark.parametrize(
    "n_params,n_iter,n_out",
    [
        (50, 200, 50),
        (100, 500, 100),
        (200, 800, 150),
        (500, 1000, 200),
    ],
)
def test_logistic_bifurcation_equivalence(n_params, n_iter, n_out):
    assert py_func(n_params, n_iter, n_out) == cy_func(n_params, n_iter, n_out)
