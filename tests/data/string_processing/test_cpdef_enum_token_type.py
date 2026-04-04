"""Test cpdef_enum_token_type equivalence."""

import pytest

from cnake_data.cy.string_processing.cpdef_enum_token_type import (
    cpdef_enum_token_type as cy_cpdef_enum_token_type,
)
from cnake_data.py.string_processing.cpdef_enum_token_type import (
    cpdef_enum_token_type as py_cpdef_enum_token_type,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_cpdef_enum_token_type_equivalence(n):
    assert py_cpdef_enum_token_type(n) == cy_cpdef_enum_token_type(n)
