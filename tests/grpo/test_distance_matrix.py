"""Test distance_matrix."""

import pytest

from cnake_charmer.py.grpo.distance_matrix import distance_matrix


@pytest.mark.parametrize("n", [2, 10, 50, 100])
def test_distance_matrix(n):
    result = distance_matrix(n)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result == distance_matrix(n)
