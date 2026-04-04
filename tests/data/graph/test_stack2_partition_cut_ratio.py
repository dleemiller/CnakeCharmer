import pytest

from cnake_data.cy.graph.stack2_partition_cut_ratio import stack2_partition_cut_ratio as cy_func
from cnake_data.py.graph.stack2_partition_cut_ratio import stack2_partition_cut_ratio as py_func


@pytest.mark.parametrize(
    "node_count, group_mod, seed_tag",
    [(80, 3, 5), (160, 5, 11), (240, 7, 19)],
)
def test_stack2_partition_cut_ratio_equivalence(node_count, group_mod, seed_tag):
    assert py_func(node_count, group_mod, seed_tag) == cy_func(node_count, group_mod, seed_tag)
