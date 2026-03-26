"""Test histogram equivalence."""

import pytest

from cnake_charmer.cy.numerical.histogram import histogram as cy_histogram
from cnake_charmer.py.numerical.histogram import histogram as py_histogram


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_histogram_equivalence(n):
    assert py_histogram(n) == cy_histogram(n)
