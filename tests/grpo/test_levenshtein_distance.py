"""Test levenshtein_distance."""

import pytest

from cnake_charmer.py.grpo.levenshtein_distance import levenshtein_distance


@pytest.mark.parametrize("n", [1, 10, 50, 200])
def test_levenshtein_distance(n):
    result = levenshtein_distance(n)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result == levenshtein_distance(n)
