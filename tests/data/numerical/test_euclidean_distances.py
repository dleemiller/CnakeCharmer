"""Test euclidean distances equivalence."""

import pytest

from cnake_data.cy.numerical.euclidean_distances import (
    euclidean_distances as cy_euclidean_distances,
)
from cnake_data.py.numerical.euclidean_distances import (
    euclidean_distances as py_euclidean_distances,
)


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_euclidean_distances_equivalence(n):
    py_result = py_euclidean_distances(n)
    cy_result = cy_euclidean_distances(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
