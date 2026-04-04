"""Test entropy equivalence."""

import pytest

from cnake_data.cy.compression.entropy import entropy as cy_func
from cnake_data.py.compression.entropy import entropy as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_entropy_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
