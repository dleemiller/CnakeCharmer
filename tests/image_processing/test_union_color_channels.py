"""Test union_color_channels equivalence."""

import pytest

from cnake_charmer.cy.image_processing.union_color_channels import (
    union_color_channels as cy_func,
)
from cnake_charmer.py.image_processing.union_color_channels import (
    union_color_channels as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_union_color_channels_equivalence(n):
    assert py_func(n) == cy_func(n)
