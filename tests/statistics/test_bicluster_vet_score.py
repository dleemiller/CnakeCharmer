"""Test bicluster_vet_score equivalence."""

import pytest

from cnake_charmer.cy.statistics.bicluster_vet_score import bicluster_vet_score as cy_func
from cnake_charmer.py.statistics.bicluster_vet_score import bicluster_vet_score as py_func


@pytest.mark.parametrize("args", [(24, 16, 5), (48, 24, 11), (64, 32, 17)])
def test_bicluster_vet_score_equivalence(args):
    p = py_func(*args)
    c = cy_func(*args)
    for pv, cv in zip(p, c, strict=False):
        assert abs(pv - cv) / max(abs(pv), 1.0) < 1e-8
