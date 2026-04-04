"""Gale-Shapley stable marriage algorithm.

Keywords: algorithms, stable marriage, gale shapley, matching, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def stable_marriage(n: int) -> int:
    """Run Gale-Shapley stable matching for n men and n women.

    Man i ranks woman (i + j*7) % n at position j.
    Woman i ranks man (i + j*13) % n at position j.

    Args:
        n: Number of men/women.

    Returns:
        Sum of all men's partner indices.
    """
    # Build preference lists
    man_pref = [[(i + j * 7) % n for j in range(n)] for i in range(n)]
    # Woman ranking: rank[w][m] = position of man m in woman w's list
    woman_rank = [[0] * n for _ in range(n)]
    for w in range(n):
        for j in range(n):
            m = (w + j * 13) % n
            woman_rank[w][m] = j

    # Gale-Shapley
    man_next = [0] * n  # next woman to propose to
    woman_partner = [-1] * n  # current partner of woman
    man_partner = [-1] * n  # current partner of man
    free_men = list(range(n))

    while free_men:
        m = free_men.pop()
        w = man_pref[m][man_next[m]]
        man_next[m] += 1
        if woman_partner[w] == -1:
            woman_partner[w] = m
            man_partner[m] = w
        elif woman_rank[w][m] < woman_rank[w][woman_partner[w]]:
            old_m = woman_partner[w]
            woman_partner[w] = m
            man_partner[m] = w
            man_partner[old_m] = -1
            free_men.append(old_m)
        else:
            free_men.append(m)

    return sum(man_partner)
