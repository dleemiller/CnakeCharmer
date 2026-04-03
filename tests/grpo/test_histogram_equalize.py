"""Test histogram_equalize."""

import pytest

from cnake_charmer.py.grpo.histogram_equalize import histogram_equalize


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_histogram_equalize(n):
    assert histogram_equalize(n) == histogram_equalize(n)
