"""Local push operators for approximate personalized PageRank."""

from __future__ import annotations


def pagerank_limit_push(s, r, w_i, a_i, push_node, rho):
    a_inf = rho * r[push_node]
    b_inf = (1.0 - rho) * r[push_node]
    s[push_node] += a_inf
    r[push_node] = 0.0
    for i in range(len(a_i)):
        r[a_i[i]] += b_inf * w_i[i]


def pagerank_lazy_push(s, r, w_i, a_i, push_node, rho, laziness_factor):
    a = rho * r[push_node]
    b = (1.0 - rho) * (1.0 - laziness_factor) * r[push_node]
    c = (1.0 - rho) * laziness_factor * r[push_node]
    s[push_node] += a
    r[push_node] = c
    for i in range(len(a_i)):
        r[a_i[i]] += b * w_i[i]


def cumulative_pagerank_difference_limit_push(s, r, w_i, a_i, push_node, rho):
    b_inf = (1.0 - rho) * r[push_node]
    r[push_node] = 0.0
    for i in range(len(a_i)):
        cp = b_inf * w_i[i]
        s[a_i[i]] += cp
        r[a_i[i]] += cp
