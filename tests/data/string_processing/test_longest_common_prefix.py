"""Test longest_common_prefix equivalence."""

import pytest

from cnake_data.cy.string_processing.longest_common_prefix import (
    longest_common_prefix as cy_func,
)
from cnake_data.py.string_processing.longest_common_prefix import (
    longest_common_prefix as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_longest_common_prefix_equivalence(n):
    assert py_func(n) == cy_func(n)
