"""Test euclid_distance_pair equivalence."""

import pytest

from cnake_charmer.cy.numerical.euclid_distance_pair import euclid_distance_pair as cy_func
from cnake_charmer.py.numerical.euclid_distance_pair import euclid_distance_pair as py_func


@pytest.mark.parametrize(
    "seed,pair_count,scale",
    [(3, 10, 0.1), (17, 100, 0.01), (99, 1000, 0.005)],
)
def test_euclid_distance_pair_equivalence(seed, pair_count, scale):
    py_result = py_func(seed, pair_count, scale)
    cy_result = cy_func(seed, pair_count, scale)
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-10
