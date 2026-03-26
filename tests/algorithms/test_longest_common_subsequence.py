"""Test longest_common_subsequence equivalence."""

import pytest

from cnake_charmer.cy.algorithms.longest_common_subsequence import (
    longest_common_subsequence as cy_lcs,
)
from cnake_charmer.py.algorithms.longest_common_subsequence import (
    longest_common_subsequence as py_lcs,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_common_subsequence_equivalence(n):
    py_result = py_lcs(n)
    cy_result = cy_lcs(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
