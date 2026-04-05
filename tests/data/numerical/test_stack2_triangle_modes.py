import pytest

from cnake_data.cy.numerical.stack2_triangle_modes import stack2_triangle_modes as cy_func
from cnake_data.py.numerical.stack2_triangle_modes import stack2_triangle_modes as py_func


@pytest.mark.parametrize(
    "side_len, mode_code, seed_base",
    [(20, 0, 3), (24, 1, 7), (18, 2, 11), (22, 3, 17)],
)
def test_stack2_triangle_modes_equivalence(side_len, mode_code, seed_base):
    assert py_func(side_len, mode_code, seed_base) == cy_func(side_len, mode_code, seed_base)
