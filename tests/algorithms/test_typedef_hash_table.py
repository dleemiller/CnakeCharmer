"""Test typedef_hash_table equivalence."""

import pytest

from cnake_charmer.cy.algorithms.typedef_hash_table import (
    typedef_hash_table as cy_typedef_hash_table,
)
from cnake_charmer.py.algorithms.typedef_hash_table import (
    typedef_hash_table as py_typedef_hash_table,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_typedef_hash_table_equivalence(n):
    assert py_typedef_hash_table(n) == cy_typedef_hash_table(n)
