"""Test reservoir_sampling equivalence."""

import pytest

from cnake_data.cy.algorithms.reservoir_sampling import (
    reservoir_sampling as cy_reservoir_sampling,
)
from cnake_data.py.algorithms.reservoir_sampling import (
    reservoir_sampling as py_reservoir_sampling,
)


@pytest.mark.parametrize("n", [200, 1000, 10000, 100000])
def test_reservoir_sampling_equivalence(n):
    assert py_reservoir_sampling(n) == cy_reservoir_sampling(n)
