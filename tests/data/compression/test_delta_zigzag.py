"""Test delta_zigzag equivalence."""

import pytest

from cnake_data.cy.compression.delta_zigzag import delta_zigzag as cy_func
from cnake_data.py.compression.delta_zigzag import delta_zigzag as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_delta_zigzag_equivalence(n):
    assert py_func(n) == cy_func(n)
