"""Test longest_increasing_subseq equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.longest_increasing_subseq import (
    longest_increasing_subseq as cy_func,
)
from cnake_charmer.py.dynamic_programming.longest_increasing_subseq import (
    longest_increasing_subseq as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_increasing_subseq_equivalence(n):
    assert py_func(n) == cy_func(n)
