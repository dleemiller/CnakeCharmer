"""Test connected_components."""

import pytest

from cnake_charmer.py.grpo.connected_components import connected_components


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_connected_components(n):
    assert connected_components(n) == connected_components(n)
