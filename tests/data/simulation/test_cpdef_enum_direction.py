"""Test cpdef_enum_direction equivalence."""

import pytest

from cnake_data.cy.simulation.cpdef_enum_direction import (
    cpdef_enum_direction as cy_cpdef_enum_direction,
)
from cnake_data.py.simulation.cpdef_enum_direction import (
    cpdef_enum_direction as py_cpdef_enum_direction,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_cpdef_enum_direction_equivalence(n):
    assert py_cpdef_enum_direction(n) == cy_cpdef_enum_direction(n)
