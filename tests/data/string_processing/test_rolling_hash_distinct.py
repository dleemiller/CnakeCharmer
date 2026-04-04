"""Test rolling_hash_distinct equivalence between Python and Cython."""

import pytest

from cnake_data.cy.string_processing.rolling_hash_distinct import (
    rolling_hash_distinct as cy_rolling_hash_distinct,
)
from cnake_data.py.string_processing.rolling_hash_distinct import (
    rolling_hash_distinct as py_rolling_hash_distinct,
)


@pytest.mark.parametrize("n", [1000, 10000, 50000, 100000])
def test_rolling_hash_distinct_equivalence(n):
    assert py_rolling_hash_distinct(n) == cy_rolling_hash_distinct(n)
