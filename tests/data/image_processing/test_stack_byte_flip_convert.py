"""Test stack_byte_flip_convert equivalence."""

import pytest

from cnake_data.cy.image_processing.stack_byte_flip_convert import (
    stack_byte_flip_convert as cy_func,
)
from cnake_data.py.image_processing.stack_byte_flip_convert import (
    stack_byte_flip_convert as py_func,
)


@pytest.mark.parametrize("args", [(8, 6, 1), (16, 8, 3), (32, 12, 4), (64, 24, 3)])
def test_stack_byte_flip_convert_equivalence(args):
    assert py_func(*args) == cy_func(*args)
