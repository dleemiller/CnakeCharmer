"""Test longest_common_substring_rolling equivalence."""

import pytest

from cnake_data.cy.string_processing.longest_common_substring_rolling import (
    longest_common_substring_rolling as cy_func,
)
from cnake_data.py.string_processing.longest_common_substring_rolling import (
    longest_common_substring_rolling as py_func,
)


@pytest.mark.parametrize("n", [16, 32, 64, 128])
def test_longest_common_substring_rolling_equivalence(n):
    assert py_func(n) == cy_func(n)
