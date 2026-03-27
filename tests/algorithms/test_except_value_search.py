"""Test except_value_search equivalence."""

import pytest

from cnake_charmer.cy.algorithms.except_value_search import (
    except_value_search as cy_except_value_search,
)
from cnake_charmer.py.algorithms.except_value_search import (
    except_value_search as py_except_value_search,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_except_value_search_equivalence(n):
    assert py_except_value_search(n) == cy_except_value_search(n)
