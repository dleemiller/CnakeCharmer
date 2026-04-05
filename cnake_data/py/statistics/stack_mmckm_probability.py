"""Compute state probabilities in an M/M/c/K/m-style queue model.

Adapted from The Stack v2 Cython candidate:
- blob_id: 9ef6f7df00318562de207b973704905ad0a83496
- filename: erlang.pyx

Keywords: statistics, queueing, erlang, probability, normalization
"""

from cnake_data.benchmarks import python_benchmark


def _power_faculty(x: float, n: int) -> float:
    res = 1.0
    for i in range(1, n + 1):
        res = res * x / i
    return res


def _cn(lam: float, mu: float, nu: float, agents: int, n: int) -> float:
    if n <= agents:
        return _power_faculty(lam / mu, n)
    res = _power_faculty(lam / mu, agents)
    for i in range(1, n - agents + 1):
        div = agents * mu + i * nu
        if div != 0.0:
            res *= lam / div
    return res


@python_benchmark(args=(55, 31, 11, 12, 700))
def stack_mmckm_probability(
    lam_num: int, mu_num: int, nu_num: int, agents: int, capacity: int
) -> tuple:
    """Return summarized queue-state probabilities for deterministic rate inputs."""
    lam = lam_num / 10.0
    mu = mu_num / 10.0
    nu = nu_num / 20.0

    norm = 0.0
    for n in range(capacity + 1):
        norm += _cn(lam, mu, nu, agents, n)
    p0 = 1.0 / norm if norm > 0 else 0.0

    mid = capacity // 2
    p_mid = _cn(lam, mu, nu, agents, mid) * p0
    p_last = _cn(lam, mu, nu, agents, capacity) * p0

    best_n = 0
    best_p = -1.0
    for n in range(capacity + 1):
        pn = _cn(lam, mu, nu, agents, n) * p0
        if pn > best_p:
            best_p = pn
            best_n = n

    return (int(p0 * 1_000_000), int(p_mid * 1_000_000), int(p_last * 1_000_000), best_n)
