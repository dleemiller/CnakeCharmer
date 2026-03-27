"""Test histogram_bucket equivalence."""

import pytest

from cnake_charmer.cy.statistics.histogram_bucket import histogram_bucket as cy_func
from cnake_charmer.py.statistics.histogram_bucket import histogram_bucket as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_histogram_bucket_equivalence(n):
    assert py_func(n) == cy_func(n)
