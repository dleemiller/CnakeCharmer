import pytest

from cnake_data.cy.numerical.stack2_rk4_descent import stack2_rk4_descent as cy_func
from cnake_data.py.numerical.stack2_rk4_descent import stack2_rk4_descent as py_func


@pytest.mark.parametrize(
    "start_x_milli, start_y_milli, step_count, step_milli",
    [(220, -140, 800, 12), (1250, -980, 1400, 17), (3300, -2500, 2200, 9)],
)
def test_stack2_rk4_descent_equivalence(start_x_milli, start_y_milli, step_count, step_milli):
    assert py_func(start_x_milli, start_y_milli, step_count, step_milli) == cy_func(
        start_x_milli, start_y_milli, step_count, step_milli
    )
