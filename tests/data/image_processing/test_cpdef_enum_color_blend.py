"""Test cpdef_enum_color_blend equivalence."""

import pytest

from cnake_data.cy.image_processing.cpdef_enum_color_blend import (
    cpdef_enum_color_blend as cy_cpdef_enum_color_blend,
)
from cnake_data.py.image_processing.cpdef_enum_color_blend import (
    cpdef_enum_color_blend as py_cpdef_enum_color_blend,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_cpdef_enum_color_blend_equivalence(n):
    assert py_cpdef_enum_color_blend(n) == cy_cpdef_enum_color_blend(n)
